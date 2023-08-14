# Based on https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb#scrollTo=DVHs5aCA3l-_

from transformers import AutoTokenizer
from transformers import (
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

import constants
import shared.corpus_loader as loader

corpus_loader = loader.CorpusLoader()

NUM_PROC = 4
BATCHED = True
# block_size = tokenizer.model_max_length
BLOCK_SIZE = 128

reports = corpus_loader.load_corpus(returnTokenized=True)
csv_dataset= corpus_loader.convert_corpus_to_dataset(reports)
print(csv_dataset)


datasets = csv_dataset.train_test_split(
    test_size=0.1,
    shuffle=True,
    seed=200,
)
print(datasets)

tokenizer = AutoTokenizer.from_pretrained(constants.PRETRAINED_MODEL_PATH)


def tokenize_function(examples):
    return tokenizer(text=examples["tokens"], is_split_into_words=True)


tokenized_datasets = datasets.map(
    tokenize_function,
    batched=BATCHED,
    num_proc=NUM_PROC,
)

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
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

#print(tokenizer.decode(lm_datasets["train"][1]["input_ids"]))


model = AutoModelForMaskedLM.from_pretrained(constants.BASE_MODEL_PATH)

training_args = TrainingArguments(
    constants.PRETRAINED_MODEL_PATH,
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
print(f"Perplexity: {eval_results['eval_loss']}")
