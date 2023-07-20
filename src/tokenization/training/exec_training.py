# based on https://huggingface.co/learn/nlp-course/chapter6/2
# Tokenization is based on WordPiece


from transformers import AutoTokenizer
import datasets

BASE_MODEL = "../../../serialized_models/medbert_512/"
INPUT_DATA = "../../../data/test.csv"
OUTPUT_PATH = "../../../serialized_models/medbert_512_pretrained"


def pretrain_tokenizer():
    csv_dataset = datasets.load_dataset(
        "csv", data_files={"train": INPUT_DATA}, column_names=["id", "text", "etc"]
    )

    print("number of rows: ", csv_dataset.num_rows)

    training_corpus = get_training_corpus(csv_dataset)

    old_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)

    tokenizer.save_pretrained(OUTPUT_PATH)

    # Evaluation

    example = "Let's test this tokenizer on a pair of sentences."

    new_encoding = tokenizer(example)
    old_encoding = old_tokenizer(example)
    print(len(new_encoding.tokens()))
    print(new_encoding.tokens())
    print(len(old_encoding.tokens()))
    print(old_encoding.tokens())


def get_training_corpus(dataset):
    dataset = dataset["train"]
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]
