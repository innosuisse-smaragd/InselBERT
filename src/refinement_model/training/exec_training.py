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
NUM_LABELS_FACTS_ANCHORS = len(schema.label2id_anchors)
model_helper = ModelHelper(model_rf, NUM_LABELS_FACTS_ANCHORS, schema, constants.M_EXTRACTION_MODEL_NAME)

# TODO: Either load annotated data or output of fact extraction model
#loader = JSONLoader()
#reports = loader.load_json(constants.F_A_EXTRACTION_MODEL_OUTPUT_PATH)

loader = CASLoader(constants.ANNOTATED_REPORTS_PATH)
reports = loader.load_CAS_convert_to_offset_dict_refinement()

dataset = Dataset.from_list(reports)

dataset_helper = DatasetHelper(dataset, batch_size=BATCH_SIZE, tokenizer=model_helper.tokenizer)

torch.manual_seed(0)


def tokenize_and_adjust_labels(examples):
    tokenized_inputs = model_helper.tokenizer(
        examples["text"], truncation=True, is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(
            batch_index=i
        )  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif (
                word_idx != previous_word_idx
            ):  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


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