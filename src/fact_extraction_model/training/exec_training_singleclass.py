# Imports

from fact_extraction_model.data.convert_IOB_to_dataset import (
    convert_iob2_to_dataset,
)
import fact_extraction_model.model.bert_token_classification as model_hf
from datasets import Dataset, DatasetDict
from transformers import (
    BertConfig,
    get_scheduler,
    DataCollatorForTokenClassification,
    AutoTokenizer,
)
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from seqeval.metrics import classification_report, f1_score
import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd

import constants


# Constants

FILE = "../../../data/iob2data.txt"
OUTPUT_DIR = "../../../serialized_models/single-class"
BATCH_SIZE = 24
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-2
NUM_EPOCHS = 3


# Data conversion

dataset, label2id = convert_iob2_to_dataset(FILE)

label_list = list(label2id.keys())
id2label = {value: key for key, value in label2id.items()}

NUM_LABELS = len(label_list)


# Data loading and splitting

hf_dataset_unsplit = Dataset.from_list(dataset)

# split twice and combine
train_devtest = hf_dataset_unsplit.train_test_split(
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

# Data preprocessing

tokenizer = AutoTokenizer.from_pretrained(constants.BASE_MODEL_PATH)


def tokenize_and_adjust_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
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


# apply tokenization to whole dataset
encoded_hf_dataset = hf_dataset.map(
    tokenize_and_adjust_labels, batched=True, remove_columns=["ner_tags", "tokens", "id"]
)

collate_fn = DataCollatorForTokenClassification(
    tokenizer, padding="longest", return_tensors="pt"
)

train_dl = DataLoader(
    encoded_hf_dataset["train"],
    shuffle=True,
    # sampler=SubsetRandomSampler(np.random.randint(0, encoded_gmb_dataset["train"].num_rows, 1000).tolist()),
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
)
valid_dl = DataLoader(
    encoded_hf_dataset["validation"],
    shuffle=False,
    # sampler=SubsetRandomSampler(np.random.randint(0, encoded_gmb_dataset["validation"].num_rows, 200).tolist()),
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
)
test_dl = DataLoader(
    encoded_hf_dataset["test"],
    shuffle=False,
    #  sampler=SubsetRandomSampler(np.random.randint(0, encoded_gmb_dataset["test"].num_rows, 100).tolist()),
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
)


# Model instantiation
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
config = BertConfig.from_pretrained(constants.BASE_MODEL_PATH)

model = model_hf.BertForTokenClassification.from_pretrained(
    constants.BASE_MODEL_PATH,
    num_labels=NUM_LABELS,
    label2id=label2id,
    id2label=id2label,
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

# Training set-up


# Reverse of tokenize_and_align_labels()
def align_predictions(labels_cpu, preds_cpu):
    # remove -100 labels from score computation
    batch_size, seq_len = preds_cpu.shape
    labels_list, preds_list = [], []
    for bid in range(batch_size):
        example_labels, example_preds = [], []
        for sid in range(seq_len):
            # ignore label -100
            if labels_cpu[bid, sid] != -100:
                example_labels.append(id2label[labels_cpu[bid, sid]])
                example_preds.append(id2label[preds_cpu[bid, sid]])
        labels_list.append(example_labels)
        preds_list.append(example_preds)
    return labels_list, preds_list


def compute_f1_score(labels, logits):
    # convert logits to predictions and move to CPU
    preds_cpu = torch.argmax(logits, dim=-1).cpu().numpy()
    labels_cpu = labels.cpu().numpy()
    labels_list, preds_list = align_predictions(labels_cpu, preds_cpu)
    # seqeval.metrics.f1_score takes list of list of tags
    return f1_score(labels_list, preds_list)


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
        eval_score += compute_f1_score(batch["labels"], outputs.logits)
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

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

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
