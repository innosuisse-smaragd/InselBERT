# Based on https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb#scrollTo=DVHs5aCA3l-_

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import (
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


BASE_MODEL = "../../../serialized_models/medbert_512/"
INPUT_DATA = "../../../data/test.csv"
OUTPUT_PATH = "../../../serialized_models/medbert_512_pretrained/"
TOKENIZER_LOC = "../../../serialized_models/medbert_512_pretrained/"
NUM_PROC = 4

csv_dataset = load_dataset(
    "csv", data_files={"train": INPUT_DATA}, column_names=["id", "text", "etc"]
)

datasets = csv_dataset["train"].train_test_split(
    test_size=0.1,
    shuffle=True,
    seed=200,
)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_LOC)


def tokenize_function(examples):
    return tokenizer(examples["text"])


tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    num_proc=NUM_PROC,
    remove_columns=["text", "etc", "id"],
)

print(tokenized_datasets["train"][1])

# block_size = tokenizer.model_max_length
block_size = 128


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=NUM_PROC,
)

print(tokenizer.decode(lm_datasets["train"][1]["input_ids"]))


model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL)

training_args = TrainingArguments(
    OUTPUT_PATH,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm_probability=0.15
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["test"],
    data_collator=data_collator,
)

trainer.train()

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
