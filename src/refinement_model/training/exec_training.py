import os
from datetime import datetime

import bentoml
import torch
from torch.optim import AdamW
from transformers import get_scheduler

import constants
from shared.cas_loader import CASLoader
from shared.dataset_helper import DatasetHelper
from shared.json_loader import JSONLoader
from shared.model_helper import ModelHelper
from shared.schema_generator import SchemaGenerator
from shared.wandb_helper import WandbHelper
import refinement_model.model.bert_token_classification_refinement as model_rf
from datasets import Dataset

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

wandb_helper = WandbHelper(constants.M_EXTRACTION_MODEL_NAME, config)
schema = SchemaGenerator()  # TODO: Impact of sharing modifiers (current implementation)
loader = JSONLoader()
model_helper = ModelHelper(model_rf, schema, constants.M_EXTRACTION_MODEL_NAME, len(schema.label2id_modifiers))

# TODO: Either load annotated data or output of fact extraction model
#loader = JSONLoader()
#reports = loader.load_json(constants.F_A_EXTRACTION_MODEL_OUTPUT_PATH)

loader = CASLoader(constants.ANNOTATED_REPORTS_PATH, schema)
extracted_facts_with_metadata = loader.load_CAS_convert_to_offset_dict_refinement()

dictlist = []

for entry in extracted_facts_with_metadata:
    dictlist.append(entry[1])

dataset = Dataset.from_list(dictlist)
dataset_helper = DatasetHelper(dataset, batch_size=BATCH_SIZE, tokenizer=model_helper.tokenizer)
torch.manual_seed(0)


def tokenize_and_adjust_labels(examples):
    tokenized_inputs = model_helper.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding="max_length")
    labels_tok = {}
    for label_type in ["modifiers", "isStart", "isEnd"]:
        labels_tok[label_type] = []
        for i, labels_modifiers in enumerate(examples[label_type]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            if labels_modifiers is None or len(labels_modifiers) == 0:
                labels_modifiers = '0'
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the labels_modifiers to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the labels_modifiers for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(labels_modifiers[word_idx])
                # For the other tokens in a word, we set the labels_modifiers to either the current labels_modifiers or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(labels_modifiers[word_idx] if constants.LABEL_ALL_TOKENS else -100)
                previous_word_idx = word_idx

            labels_tok[label_type].append(label_ids)

    #tokenized_inputs["labels_modifiers_tok"] = labels_modifiers_tok

    return {
        **tokenized_inputs,
        "labels_modifiers_tok": labels_tok["modifiers"],
        "labels_isStart_tok": labels_tok["isStart"],
        "labels_isEnd_tok": labels_tok["isEnd"],
    }

print(dataset_helper.dataset["train"][0])

tokenized_hf_ds = dataset_helper.apply_tokenization(tokenize_and_adjust_labels, batched=True)

print(tokenized_hf_ds["train"][0])

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

# TODO: Change to modifier and binary classification loss
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

        metrics_modifiers = model_helper.compute_metrics_for_anchors(
            batch["labels_modifiers_tok"].cpu(), outputs["modifiers"].logits.cpu()
        )

        metrics_isStart = model_helper.compute_metrics_for_anchors(
            batch["labels_isStart_tok"].cpu(), outputs["isStart"].logits.cpu()
        )

        metrics_isEnd = model_helper.compute_metrics_for_anchors(
            batch["labels_isEnd_tok"].cpu(), outputs["isEnd"].logits.cpu()
        )
        eval_score += (metrics_modifiers + metrics_isStart + metrics_isEnd) / 3
        num_batches += 1
        f1_modifiers = metrics_modifiers

    eval_score /= num_batches

    return eval_loss, eval_score, f1_modifiers

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

bentoml.transformers.save_model(constants.M_EXTRACTION_MODEL_NAME, model_helper.model, custom_objects={
    "fact_schema": schema,
    "tokenizer": model_helper.tokenizer
})

model_helper.make_loss_diagram(history, path)

wandb_helper.finish()
