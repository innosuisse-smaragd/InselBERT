# Based on https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb#scrollTo=DVHs5aCA3l-_
from datetime import datetime

from transformers import AutoTokenizer
from transformers import (
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

import constants
import shared.corpus_loader as loader


def further_pretrain_model():
    print("Loading corpus")
    corpus_loader = loader.CorpusLoader()
    CORPUS_TYPE = constants.CORPUS_TYPE
    NUM_PROC = 4
    BATCHED = True
    # block_size = tokenizer.model_max_length
    # BLOCK_SIZE = 128
    reports = []

    if CORPUS_TYPE == "MAMMOGRAPHY":
        reports = corpus_loader.load_mammography_corpus()
    elif CORPUS_TYPE == "ALL":
        reports = corpus_loader.load_corpus()
    else:
        raise ValueError(f"Unknown corpus type: {CORPUS_TYPE}")

    csv_dataset = corpus_loader.convert_corpus_to_dataset_text(reports)

    datasets = csv_dataset.train_test_split(
        test_size=0.1,
        shuffle=True,
        seed=200,
    )
    print(datasets)
    print(datasets['train'][0])

    # tokenizer = AutoTokenizer.from_pretrained(constants.PRETRAINED_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(constants.BASE_MODEL_NAME)

    BLOCK_SIZE = tokenizer.model_max_length
    print("Block size", BLOCK_SIZE)

    def tokenize_function(examples):
        return tokenizer(text=examples["text"], is_split_into_words=False)

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=BATCHED,
        num_proc=NUM_PROC,
        remove_columns=["text"]
    )
    print(tokenized_datasets["train"][1])
    print("Finished tokenization")

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=500,  # 1000
        num_proc=NUM_PROC,
    )

    print(tokenizer.decode(lm_datasets["train"][1]["input_ids"]))
    print("Finished grouping texts")

    # print(tokenizer.decode(lm_datasets["train"][1]["input_ids"]))

    # model = AutoModelForMaskedLM.from_pretrained(constants.BASE_MODEL_PATH)
    model = AutoModelForMaskedLM.from_pretrained(constants.BASE_MODEL_NAME)

    output_path = constants.PRETRAINED_MODEL_PATH + datetime.now().strftime("%Y%m%d-%H%M%S") + CORPUS_TYPE + "/"
    training_args = TrainingArguments(
        output_path,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        # num_train_epochs=10,
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

    print("Start training")
    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Perplexity: {eval_results['eval_loss']}")
    trainer.save_metrics("test", eval_results)
    trainer.save_model(output_path)

if __name__ == "__main__":
    further_pretrain_model()
