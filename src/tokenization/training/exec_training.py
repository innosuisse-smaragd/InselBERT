# based on https://huggingface.co/learn/nlp-course/chapter6/2
# Tokenization is based on WordPiece


from transformers import AutoTokenizer
import shared.corpus_loader as loader
import constants


def pretrain_tokenizer():
    corpus_loader = loader.CorpusLoader()

    reports = corpus_loader.load_corpus()
    csv_dataset = corpus_loader.convert_corpus_to_dataset_text(reports=reports)

    print("number of rows: ", csv_dataset.num_rows)

    training_corpus = get_training_corpus_generator(csv_dataset)

    old_tokenizer = AutoTokenizer.from_pretrained(constants.BASE_MODEL_PATH)

    tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)

    tokenizer.save_pretrained(constants.PRETRAINED_MODEL_PATH)

def get_training_corpus_generator(dataset):
    # dataset = dataset["train"]
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000][constants.REPORTS_CSV_FILE_COLUMN_NAME]


pretrain_tokenizer()
