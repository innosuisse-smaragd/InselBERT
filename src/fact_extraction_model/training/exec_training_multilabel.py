from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    BertConfig,
    get_scheduler,
)

import numpy as np
from sklearn.metrics import multilabel_confusion_matrix

import fact_extraction_model.model.bert_multilabel_classification as model_multilabel
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
import matplotlib.pyplot as plt
import os

BASE_MODEL = "./serialized_models/medbert_512/"

MAX_LENGTH = 512

BATCH_SIZE = 24
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-2
NUM_EPOCHS = 100
OUTPUT_DIR = "./serialized_models/medbert_pretrained_multilabel/"

# Use method from shared-python
tag2id = {"PER": 1, "ORG": 2, "LOC": 3, "MISC": 4, "NCHUNK": 5, "TIME": 6, "PLACE": 7}
id2tag = {v: k for k, v in tag2id.items()}


label2id = {
    "O": 0,
    **{f"B-{k}": 2 * v - 1 for k, v in tag2id.items()},
    **{f"I-{k}": 2 * v for k, v in tag2id.items()},
}

id2label = {v: k for k, v in label2id.items()}
NUM_LABELS = len(id2label)
print(id2label)

# if not done separately, applying the tokenization function via df.map() fails
train_ds = Dataset.from_json("./data/multilabel.train.jsonlines")
val_ds = Dataset.from_json("./data/multilabel.validation.jsonlines")
test_ds = Dataset.from_json("./data/multilabel.test.jsonlines")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


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


def tokenize_and_adjust_labels(sample):
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
        max_length=MAX_LENGTH,
        truncation=True,
    )

    # We are doing a multilabel classification task at each token, we create a list of size len(label2id)=13
    # for the 13 labels
    labels = [[0 for _ in label2id.keys()] for _ in range(MAX_LENGTH)]

    # Scan all the tokens and spans, assign 1 to the corresponding label if the token lies at the beginning
    # or inside the spans
    for (token_start, token_end), token_labels in zip(
        tokenized["offset_mapping"], labels
    ):
        for span in sample["tags"]:
            role = get_token_role_in_span(
                token_start, token_end, span["start"], span["end"]
            )
            if role == "B":
                token_labels[label2id[f"B-{span['tag']}"]] = 1
            elif role == "I":
                token_labels[label2id[f"I-{span['tag']}"]] = 1

    return {**tokenized, "labels": labels}


# if not done separately, applying the tokenization function via df.map() fails
tokenized_train_ds = train_ds.map(
    tokenize_and_adjust_labels, remove_columns=train_ds.column_names
)
tokenized_val_ds = val_ds.map(
    tokenize_and_adjust_labels, remove_columns=val_ds.column_names
)
tokenized_test_ds = test_ds.map(
    tokenize_and_adjust_labels, remove_columns=val_ds.column_names
)

# data_collator = DataCollatorWithPadding(tokenizer, padding=True)
data_collator = DataCollatorWithPadding(
    tokenizer, padding="longest", return_tensors="pt"
)

train_dl = DataLoader(
    tokenized_train_ds,
    shuffle=True,
    # sampler=SubsetRandomSampler(np.random.randint(0, encoded_gmb_dataset["train"].num_rows, 1000).tolist()),
    batch_size=BATCH_SIZE,
    collate_fn=data_collator,
)
valid_dl = DataLoader(
    tokenized_val_ds,
    shuffle=False,
    # sampler=SubsetRandomSampler(np.random.randint(0, encoded_gmb_dataset["validation"].num_rows, 200).tolist()),
    batch_size=BATCH_SIZE,
    collate_fn=data_collator,
)
test_dl = DataLoader(
    tokenized_test_ds,
    shuffle=False,
    #  sampler=SubsetRandomSampler(np.random.randint(0, encoded_gmb_dataset["test"].num_rows, 100).tolist()),
    batch_size=BATCH_SIZE,
    collate_fn=data_collator,
)

# Model instantiation
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
config = BertConfig.from_pretrained(BASE_MODEL)

model = model_multilabel.BertForMultiLabelClassification.from_pretrained(
    BASE_MODEL, num_labels=NUM_LABELS, label2id=label2id, id2label=id2label
)


# Resize embedding size if additional tokens are added (only for RE)
# model.bert.resize_token_embeddings(len(tokenizer.vocab))
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

num_training_steps = NUM_EPOCHS * len(train_dl)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)


def divide(a: int, b: int):
    return a / b if b > 0 else 0


def compute_metrics(p):
    """
    Customize the `compute_metrics` of `transformers`
    Args:
        - p (tuple):      2 numpy arrays: predictions and true_labels
    Returns:
        - metrics (dict): f1 score on
    """
    # (1)
    predictions, true_labels = p

    # (2)
    predicted_labels = np.where(
        predictions > 0, np.ones(predictions.shape), np.zeros(predictions.shape)
    )
    metrics = {}

    # (3)
    cm = multilabel_confusion_matrix(
        true_labels.reshape(-1, NUM_LABELS), predicted_labels.reshape(-1, NUM_LABELS)
    )

    # (4)
    for label_idx, matrix in enumerate(cm):
        if label_idx == 0:
            continue  # We don't care about the label "O"
        tp, fp, fn = matrix[1, 1], matrix[0, 1], matrix[1, 0]
        precision = divide(tp, tp + fp)
        recall = divide(tp, tp + fn)
        f1 = divide(2 * precision * recall, precision + recall)
        metrics[f"f1_{id2label[label_idx]}"] = f1

    # (5)
    macro_f1 = sum(list(metrics.values())) / (NUM_LABELS - 1)
    metrics["macro_f1"] = macro_f1

    return metrics


def do_train(model, train_dl):
    train_loss = 0
    model.train()
    for bid, batch in enumerate(train_dl):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        train_loss += loss.detach().cpu().numpy()
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    return train_loss


def do_eval(model, eval_dl):
    model.eval()
    eval_loss, eval_score, num_batches = 0, 0, 0
    for bid, batch in enumerate(eval_dl):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss

        eval_loss += loss.detach().cpu().numpy()
        # eval_score += compute_f1_score(batch["labels"], outputs.logits)
        metrics = compute_metrics([outputs.logits.cpu(), batch["labels"].cpu()])
        eval_score += metrics["macro_f1"]
        num_batches += 1

    eval_score /= num_batches

    return eval_loss, eval_score


def save_checkpoint(model, model_dir, epoch):
    model.save_pretrained(os.path.join(OUTPUT_DIR, "ckpt-{:d}".format(epoch)))


def save_training_history(history, model_dir, epoch):
    fhist = open(os.path.join(OUTPUT_DIR, "history.tsv"), "w")
    for epoch, train_loss, eval_loss, eval_score in history:
        fhist.write(
            "{:d}\t{:.5f}\t{:.5f}\t{:.5f}\n".format(
                epoch, train_loss, eval_loss, eval_score
            )
        )
    fhist.close()

    # Training loop


# if os.path.exists(OUTPUT_DIR):
#   shutil.rmtree(OUTPUT_DIR)
#   os.makedirs(OUTPUT_DIR)

history = []

for epoch in range(NUM_EPOCHS):
    train_loss = do_train(model, train_dl)
    eval_loss, eval_score = do_eval(model, valid_dl)
    history.append((epoch + 1, train_loss, eval_loss, eval_score))
    print(
        "EPOCH {:d}, train loss: {:.3f}, val loss: {:.3f}, f1-score: {:.5f}".format(
            epoch + 1, train_loss, eval_loss, eval_score
        )
    )
    save_checkpoint(model, OUTPUT_DIR, epoch + 1)
    save_training_history(history, OUTPUT_DIR, epoch + 1)


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
    plt.savefig(OUTPUT_DIR + "/loss.png", dpi=300)


make_loss_diagram()
