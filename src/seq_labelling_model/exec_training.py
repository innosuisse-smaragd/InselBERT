#https://huggingface.co/docs/transformers/tasks/token_classification
from datetime import datetime

import evaluate
import torch

from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification, \
    TrainingArguments, Trainer, EarlyStoppingCallback

import constants
from shared.cas_loader import CASLoader
from shared.dataset_helper import DatasetHelper
from shared.model_helper import ModelHelper
from shared.schema_generator import SchemaGenerator
from shared.wandb_helper import WandbHelper
from datasets import Dataset
import numpy as np

BATCH_SIZE = 16
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-2
NUM_EPOCHS = 100

config = {
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "epochs": NUM_EPOCHS
}

wandb_helper = WandbHelper(constants.SEQ_LABELLING_MODEL_NAME, config)
schema = SchemaGenerator()  # TODO: Impact of sharing modifiers (current implementation)
model_helper = ModelHelper(AutoModelForTokenClassification, schema, constants.SEQ_LABELLING_MODEL_NAME, len(schema.label2id_combined))



loader = CASLoader(constants.ANNOTATED_REPORTS_PATH, schema)
extracted_facts_with_combined_tags = loader.load_CAS_convert_to_combined_tag_list_seq_labelling()
dictlist = []

for entry in extracted_facts_with_combined_tags:
    dictlist.append(entry[1])


dataset = Dataset.from_list(dictlist)
#https://huggingface.co/docs/datasets/v2.12.0/en/loading#slice-splits
#val_ds = Dataset.from_list(dictlist, split=[f"train[{k}%:{k+10}%]" for k in range(0, 100, 10)])
#train_ds = Dataset.from_list(dictlist, split=[f"train[:{k}%]+train[{k+10}%:]" for k in range(0, 100, 10)])

dataset_helper = DatasetHelper(dataset, batch_size=BATCH_SIZE, tokenizer=model_helper.tokenizer)
torch.manual_seed(0)

print("First entry: ", dataset_helper.dataset["train"][0])
print("Second entry: ", dataset_helper.dataset["train"][1])

def tokenize_and_align_labels(examples):
    tokenized_inputs = model_helper.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

print("First train set entry: ", dataset_helper.dataset["train"][0])

tokenized_hf_ds = dataset_helper.dataset.map(tokenize_and_align_labels, batched=True)


print("First tokenized and shuffled train set entry: ",tokenized_hf_ds["train"][0])

data_collator = DataCollatorForTokenClassification(tokenizer=model_helper.tokenizer)

seqeval = evaluate.load("seqeval")
label_names = list(schema.id2label_combined.values())

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
        **all_metrics
    }

training_args = TrainingArguments(
    output_dir=constants.SEQ_LABELLING_MODEL_PATH,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    report_to="wandb",
    load_best_model_at_end=True,
    metric_for_best_model="eval_precision"
)

trainer = Trainer(
    model=model_helper.model,
    args=training_args,
    train_dataset=tokenized_hf_ds["train"],
    eval_dataset=tokenized_hf_ds["test"],
    tokenizer=model_helper.tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience = 5)]
)

trainer.train()
validation_results = trainer.predict(tokenized_hf_ds["validation"])
trainer.save_metrics(split="validation", metrics=validation_results.metrics, combined=False)

trainer.create_model_card(language="de", tasks="token-classification", model_name="inselbert-sequence-labeller", tags=["seq_labelling", "medical", "german"])
trainer.save_model(constants.SEQ_LABELLING_MODEL_PATH + datetime.now().strftime("%Y%m%d-%H%M%S") + "/")
wandb_helper.finish()
