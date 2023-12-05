import os
from datetime import datetime

import bentoml.transformers
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from seqeval.metrics import f1_score
from sklearn.metrics import multilabel_confusion_matrix
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    get_scheduler,
)

import constants
import fact_extraction_model.model.bert_two_heads as model_combined
import shared.model_helpers as helper
import wandb
import shared.schema_generator
from shared.cas_loader import CASLoader

# Imports


BATCH_SIZE = 24
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-2
NUM_EPOCHS = 100  # 100
MAX_LENGTH = 512

run = wandb.init(
    # Set the project where this run will be logged
    project="smaragd-llm-01",
    # Track hyperparameters and run metadata
    config={
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "epochs": NUM_EPOCHS,
        "max_input_length": MAX_LENGTH,
    },
)

torch.manual_seed(0)

schema = shared.schema_generator.SchemaGenerator()  # TODO: Impact of sharing modifiers (current implementation)
NUM_LABELS_FACTS_ANCHORS = len(schema.label2id_anchors)

loader = CASLoader(constants.ANNOTATED_REPORTS_PATH)
reports = loader.load_CAS_convert_to_offset_dict()
dataset_unsplit = Dataset.from_list(reports)

print(dataset_unsplit[0])
print("Anchors: ",schema.id2label_anchors)
print("Facts: ",schema.id2label_facts)
print("Mofidiers: ",schema.id2label_modifiers)



# split twice and combine
train_devtest = dataset_unsplit.train_test_split(
    shuffle=True, seed=200, test_size=0.3
)
hf_dev_test = train_devtest["test"].train_test_split(
    shuffle=True, seed=200, test_size=0.50
)
hf_dataset = DatasetDict(
    {
        "train": train_devtest["train"],
        "test": hf_dev_test["test"],
        "validation": hf_dev_test["train"],
    }
)

tokenizer = helper.getTokenizer()


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
    tokenized = tokenizer(
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


tokenized_hf_ds = hf_dataset.map(
    tokenize_and_adjust_labels, remove_columns=hf_dataset["train"].column_names
)

sample = tokenized_hf_ds["train"][0]

print("--------Token---------|--------Labels----------")
for token_id, token_labels, anchor_labels in zip(sample["input_ids"], sample["labels_facts"], sample["labels_anchors"]):
    # Decode the token_id into text
    token_text = tokenizer.decode(token_id)

    # Retrieve all the indices corresponding to the "1" at each token, decode them to label name
    labels = [schema.id2label_facts[label_index] for label_index, value in enumerate(token_labels) if value == 1]

    # Decode those indices into label name
    print(f" {token_text:20} | {labels}"
          f" | {anchor_labels}")

    # Finish when we meet the end of sentence.
    if token_text == "</s>":
        break



# data_collator = DataCollatorWithPadding(tokenizer, padding=True)
data_collator = DataCollatorWithPadding(
    tokenizer, padding="longest", return_tensors="pt"
)

train_dl = DataLoader(
    tokenized_hf_ds["train"],
    shuffle=True,
    # sampler=SubsetRandomSampler(np.random.randint(0, encoded_gmb_dataset["train"].num_rows, 1000).tolist()),
    batch_size=BATCH_SIZE,
    collate_fn=data_collator,
)
valid_dl = DataLoader(
    tokenized_hf_ds["validation"],
    shuffle=False,
    # sampler=SubsetRandomSampler(np.random.randint(0, encoded_gmb_dataset["validation"].num_rows, 200).tolist()),
    batch_size=BATCH_SIZE,
    collate_fn=data_collator,
)
test_dl = DataLoader(
    tokenized_hf_ds["test"],
    shuffle=False,
    #  sampler=SubsetRandomSampler(np.random.randint(0, encoded_gmb_dataset["test"].num_rows, 100).tolist()),
    batch_size=BATCH_SIZE,
    collate_fn=data_collator,
)

# Model instantiation
device = helper.getDevice()
model = helper.getFurtherPretrainedModel(
    modelclass=model_combined,
    num_labels=NUM_LABELS_FACTS_ANCHORS
    # label2id=label2id,
    # id2label=id2label,
)
model = model.to(device)

wandb.watch(model, log_freq=100)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

num_training_steps = NUM_EPOCHS * len(train_dl)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)


# Metrics for single class
def align_predictions(labels_cpu, preds_cpu):
    # remove -100 labels from score computation
    batch_size, seq_len = preds_cpu.shape
    labels_list, preds_list = [], []
    for bid in range(batch_size):
        example_labels, example_preds = [], []
        for sid in range(seq_len):
            # ignore label -100
            if labels_cpu[bid, sid] != -100:
                example_labels.append(schema.id2label_anchors[labels_cpu[bid, sid]])
                example_preds.append(schema.id2label_anchors[preds_cpu[bid, sid]])
        labels_list.append(example_labels)
        preds_list.append(example_preds)
    return labels_list, preds_list


def compute_metrics_for_anchors(labels, logits):
    # convert logits to predictions and move to CPU
    preds_cpu = torch.argmax(logits, dim=-1).cpu().numpy()
    labels_cpu = labels.cpu().numpy()
    labels_list, preds_list = align_predictions(labels_cpu, preds_cpu)
    # seqeval.metrics.f1_score takes list of list of tags
    return f1_score(labels_list, preds_list)


# Metrics for multilabel
def divide(a: int, b: int):
    return a / b if b > 0 else 0


def compute_metrics_for_facts(p):
    """
    Customize the `compute_metrics` of `transformers`
    Args:
        - p (tuple):      2 numpy arrays: predictions and true_labels
    Returns:
        - metrics (dict): f1 score on
    """
    predictions, true_labels = p
    predicted_labels = np.where(
        predictions > 0, np.ones(predictions.shape), np.zeros(predictions.shape)
    )
    metrics = {}

    cm = multilabel_confusion_matrix(
        true_labels.reshape(-1, NUM_LABELS_FACTS_ANCHORS), predicted_labels.reshape(-1, NUM_LABELS_FACTS_ANCHORS)
    )

    for label_idx, matrix in enumerate(cm):
        if label_idx == 0:
            continue  # We don't care about the label "O"
        tp, fp, fn = matrix[1, 1], matrix[0, 1], matrix[1, 0]
        precision = divide(tp, tp + fp)
        recall = divide(tp, tp + fn)
        f1 = divide(2 * precision * recall, precision + recall)
        metrics[f"f1_{schema.id2label_facts[label_idx]}"] = f1

    macro_f1 = sum(list(metrics.values())) / (NUM_LABELS_FACTS_ANCHORS - 1)
    metrics["macro_f1_facts"] = macro_f1

    return metrics


def do_train(model, train_dl):
    train_loss = 0
    model.train()
    for bid, batch in enumerate(train_dl):
        batch = {k: v.to(device) for k, v in batch.items()}
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
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        loss_averaged = outputs["loss"].loss  # TODO: Check calculation

        eval_loss += loss_averaged.detach().cpu().numpy()
        # eval_score += compute_f1_score(batch["labels"], outputs.logits)
        metrics_facts = compute_metrics_for_facts(
            [outputs["facts"].logits.cpu(), batch["labels_facts"].cpu()]
        )
        metrics_anchors = compute_metrics_for_anchors(
            batch["labels_anchors"].cpu(), outputs["anchors"].logits.cpu()
        )
        eval_score += (metrics_facts["macro_f1_facts"] + metrics_anchors) / 2
        num_batches += 1
        micro_metrics = metrics_facts
        f1_anchors = metrics_anchors

    eval_score /= num_batches

    return eval_loss, eval_score, micro_metrics, f1_anchors


def save_checkpoint(model, model_dir, epoch):
    model.save_pretrained(os.path.join(model_dir, "ckpt"))


def save_training_history(history, model_dir, epoch):
    fhist = open(os.path.join(model_dir, "history.tsv"), "w")
    for epoch, train_loss, eval_loss, eval_score in history:
        fhist.write(
            "{:d}\t{:.5f}\t{:.5f}\t{:.5f}\n".format(
                epoch, train_loss, eval_loss, eval_score
            )
        )
    fhist.close()


def save_micro_metrics(micro_metrics, model_dir, epoch):
    fhist = open(os.path.join(model_dir, "fact_metrics.tsv"), "w")
    for epoch, item in micro_metrics:
        fhist.write("{:d}\t".format(epoch))
        for key, value in item.items():
            fhist.write("{}\t{:.5f}\t".format(key, value))
        fhist.write("\n")
    fhist.close()

history = []
micro_metrics = []

best_eval_loss = 100
now = datetime.now()
dt_string = now.strftime("%d%m%Y%H%M")
path = os.path.join(constants.FINETUNED_MODEL_PATH, dt_string)
for epoch in range(NUM_EPOCHS):
    train_loss = do_train(model, train_dl)
    eval_loss, eval_score, micro_metrics_batch, f1_anchors = do_eval(model, valid_dl)
    micro_metrics.append((epoch + 1, micro_metrics_batch))
    history.append((epoch + 1, train_loss, eval_loss, eval_score))
    print(
        "EPOCH {:d}, train loss: {:.3f}, val loss: {:.3f}, f1-score: {:.5f}".format(
            epoch + 1, train_loss, eval_loss, eval_score
        )
    )
    wandb.log(
        {
            "train_loss": train_loss,
            "validation_loss": eval_loss,
            "f1-score": eval_score,
            "f1_anchors": f1_anchors,
        }
    )
    for key, value in micro_metrics_batch.items():
        wandb.log({key: value})
    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss
        save_checkpoint(model, path, epoch + 1)
        print("Model saved as current eval_loss is: ", best_eval_loss)
    save_training_history(history, path, epoch + 1)
    save_micro_metrics(micro_metrics, path, epoch + 1)

bentoml.transformers.save_model(constants.FACT_EXTRACTION_MODEL_NAME, model, custom_objects={
    "fact_schema": schema
})

def make_loss_diagram():
    plt.subplot(2, 1, 1)
    plt.plot([train_loss for _, train_loss, _, _ in history], label="train")
    plt.plot([eval_loss for _, _, eval_loss, _ in history], label="validation")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(loc="best")

    plt.subplot(2, 1, 2)
    plt.plot([eval_score for _, _, _, eval_score in history], label="validation")
    plt.xlabel("epochs")
    plt.ylabel("f1-score")
    plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig(path + "/loss.png", dpi=300)


make_loss_diagram()

wandb.finish()
