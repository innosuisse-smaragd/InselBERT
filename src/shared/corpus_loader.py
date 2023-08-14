import pandas as pd
from datasets import Dataset
from smaragd_shared_python.pre_processing.spacy_tokenizer import SpacyTokenizer
from smaragd_shared_python.report.report import Report
from smaragd_shared_python.report.report_parser import ReportParser

from constants import REPORTS_CSV_FILE_PATH

from env import DECRYPTION_KEY


class CorpusLoader:

    def load_corpus(self, returnTokenized = False) -> list[Report]:
        decryption_key = DECRYPTION_KEY if DECRYPTION_KEY != "undefined" else None
        tokenizer = SpacyTokenizer() if returnTokenized else None
        reports = ReportParser.parse_report_csv(
            REPORTS_CSV_FILE_PATH, decryption_key, tokenizer
        )
        return reports

    def convert_corpus_to_dataset(self, reports):
        reports_tokens = [
            {
                "tokens": [
                    report.report_text[token.begin : token.end]
                    for token in report.tokens
                ]
            }
            for report in reports
        ]
        df = pd.DataFrame(reports_tokens)
        dataset = Dataset.from_pandas(df)
        return dataset

    def convert_corpus_to_dataset_text(self, reports):
        reports_texts = [
            {
                "text": report.report_text

            }
            for report in reports
        ]
        df = pd.DataFrame(reports_texts)
        dataset = Dataset.from_pandas(df)
        return dataset



