import os
from datetime import datetime

import bentoml.transformers
import numpy as np
import torch
from datasets import Dataset
from torch.optim import AdamW
from transformers import (
    get_scheduler,
)

import constants
import fact_extraction_model.model.bert_two_heads as model_combined
from shared.cas_loader import CASLoader
from shared.dataset_helper import DatasetHelper
from shared.model_helper import ModelHelper
from shared.schema_generator import SchemaGenerator
from shared.wandb_helper import WandbHelper

BATCH_SIZE = 24
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-2
NUM_EPOCHS = 100  # 100
MAX_LENGTH = 512

config = {
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "epochs": NUM_EPOCHS,
    "max_input_length": MAX_LENGTH,
}


wandb_helper = WandbHelper(constants.F_A_EXTRACTION_MODEL_NAME, config)
schema = SchemaGenerator()  # TODO: Impact of sharing modifiers (current implementation)
loader = CASLoader(constants.ANNOTATED_REPORTS_PATH, schema)
NUM_LABELS_FACTS_ANCHORS = len(schema.label2id_anchors)
model_helper = ModelHelper(model_combined, NUM_LABELS_FACTS_ANCHORS, schema, constants.F_A_EXTRACTION_MODEL_NAME)

reports = loader.load_CAS_convert_to_offset_dict()
dataset = Dataset.from_list(reports)

dataset_helper = DatasetHelper(dataset, batch_size=BATCH_SIZE, tokenizer=model_helper.tokenizer)
torch.manual_seed(0)


def get_token_role_in_span(
    token_start: int, token_end: int, span_start: int, span_end: int
):
    """
    Check if the token is inside a span.
    Args:
      - token_start, token_end: Start and end offset of the token
      - span_start, span_end: Start and end of the span
    Returns:
      - "B" if beginning
      - "I" if inner
      - "O" if outer
      - "N" if not valid token (like <SEP>, <CLS>, <UNK>)
    """
    if token_end <= token_start:
        return "N"
    if token_start < span_start or token_end > span_end:
        return "O"
    if token_start > span_start:
        return "I"
    else:
        return "B"


def tokenize_and_adjust_labels(sample):  # TODO: Handle subword tokens -> -100
    """
    Args:
        - sample (dict): {"id": "...", "text": "...", "tags": [{"start": ..., "end": ..., "tag": ...}, ...]
    Returns:
        - The tokenized version of `sample` and the labels of each token.
    """
    # Tokenize the text, keep the start and end positions of tokens with `return_offsets_mapping` option
    # Use max_length and truncation to ajust the text length
    tokenized = model_helper.tokenizer(
        sample["text"],
        return_offsets_mapping=True,
        padding="max_length",
        # max_length=MAX_LENGTH,
        truncation=True,
    )

    # Repeat for all three label dimensions
    # We are doing a multilabel classification task at each token, we create a list of size len(label2id)=13
    # for the 13 labels
    labels_facts = [
        [0 for _ in schema.label2id_facts.keys()] for _ in range(MAX_LENGTH)
    ]

    # Scan all the tokens and spans, assign 1 to the corresponding label if the token lies at the beginning
    # or inside the spans
    previous_word_id = None
    for (token_start, token_end), token_labels in zip(
            tokenized["offset_mapping"], labels_facts
    ):
        for span in sample["fact_tags"]:
            role = get_token_role_in_span(
                token_start, token_end, span["start"], span["end"]
            )
            if role == "B":
                token_labels[schema.label2id_facts[f"B-{span['tag']}"]] = 1
            elif role == "I":
                token_labels[schema.label2id_facts[f"I-{span['tag']}"]] = 1

    # We are doing a multilabel classification task at each token, we create a list of size len(label2id)=13
    # for the 13 labels
    labels_anchors = [[0 for _ in range(1)] for _ in range(MAX_LENGTH)]

    # Scan all the tokens and spans, assign 1 to the corresponding label if the token lies at the beginning
    # or inside the spans
    for (token_start, token_end), token_labels in zip(
            tokenized["offset_mapping"], labels_anchors
    ):
        # for span in sample["tags"]:

        for span in sample["anchor_tags"]: # TODO: anchor_tags, only one annotation per anchor
            role = get_token_role_in_span(
                token_start, token_end, span["start"], span["end"]
            )
            if role == "B":
                token_labels[0] = schema.label2id_anchors[f"B-{span['tag']}"]
            elif role == "I":
                token_labels[0] = schema.label2id_anchors[f"I-{span['tag']}"]

        # token_labels[0] = token_labels[0].item()
    labels_anchors_ = []
    for arr in labels_anchors:
        test = np.asarray(arr)
        labels_anchors_.append(test.item())
    # labels_anchors_ = float(np.asarray(labels_anchors))

    return {
        **tokenized,
        "labels_facts": labels_facts,
        "labels_anchors": labels_anchors_,
    }


tokenized_hf_ds = dataset_helper.apply_tokenization(tokenize_and_adjust_labels)

train_dl = dataset_helper.load_tokenized_dataset(tokenized_hf_ds["train"], shuffle=True)
valid_dl = dataset_helper.load_tokenized_dataset(tokenized_hf_ds["validation"])
test_dl = dataset_helper.load_tokenized_dataset(tokenized_hf_ds["test"])


optimizer = AdamW(model_helper.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

num_training_steps = NUM_EPOCHS * len(train_dl)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)


def do_train(model, train_dl):
    train_loss = 0
    model.train()
    for bid, batch in enumerate(train_dl):
        batch = {k: v.to(model_helper.device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs["loss"].loss
        train_loss += loss.detach().cpu().numpy()
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    return train_loss


def do_eval(model, eval_dl):
    model.eval()
    eval_loss, eval_score, num_batches, f1_anchors = 0, 0, 0, 0
    micro_metrics = {}
    for bid, batch in enumerate(eval_dl):
        batch = {k: v.to(model_helper.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        loss_averaged = outputs["loss"].loss  # TODO: Check calculation

        eval_loss += loss_averaged.detach().cpu().numpy()
        # eval_score += compute_f1_score(batch["labels"], outputs.logits)
        metrics_facts = model_helper.compute_metrics_for_facts(
            [outputs["facts"].logits.cpu(), batch["labels_facts"].cpu()]
        )
        metrics_anchors = model_helper.compute_metrics_for_anchors(
            batch["labels_anchors"].cpu(), outputs["anchors"].logits.cpu()
        )
        eval_score += (metrics_facts["macro_f1_facts"] + metrics_anchors) / 2
        num_batches += 1
        micro_metrics = metrics_facts
        f1_anchors = metrics_anchors

    eval_score /= num_batches

    return eval_loss, eval_score, micro_metrics, f1_anchors

history = []
micro_metrics = []

best_eval_loss = 100
now = datetime.now()
dt_string = now.strftime("%d%m%Y%H%M")
folder_string = constants.F_A_EXTRACTION_MODEL_NAME + "_" + dt_string
path = os.path.join(constants.F_A_EXTRACTION_MODEL_PATH, folder_string)

for epoch in range(NUM_EPOCHS):
    train_loss = do_train(model_helper.model, train_dl)
    eval_loss, eval_score, micro_metrics_batch, f1_anchors = do_eval(model_helper.model, valid_dl)
    micro_metrics.append((epoch + 1, micro_metrics_batch))
    history.append((epoch + 1, train_loss, eval_loss, eval_score))
    print(
        "EPOCH {:d}, train loss: {:.3f}, val loss: {:.3f}, f1-score: {:.5f}".format(
            epoch + 1, train_loss, eval_loss, eval_score
        )
    )
    wandb_helper.log(train_loss=train_loss, eval_loss=eval_loss, eval_score=eval_score, f1_anchors=f1_anchors)
    wandb_helper.log(**micro_metrics_batch)

    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss
        model_helper.model.save_pretrained(path)
        print("Model saved as current eval_loss is: ", best_eval_loss)
    model_helper.save_training_history(history, path)
    model_helper.save_micro_metrics(micro_metrics, path)

bentoml.transformers.save_model(constants.F_A_EXTRACTION_MODEL_NAME, model_helper.model, custom_objects={
    "fact_schema": schema,
    "tokenizer": model_helper.tokenizer
})

model_helper.make_loss_diagram(history, path)

wandb_helper.finish()
