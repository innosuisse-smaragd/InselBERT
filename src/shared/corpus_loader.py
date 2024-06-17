import csv
from _csv import QUOTE_ALL

import pandas as pd
from datasets import Dataset
from smaragd_shared_python.pre_processing.abstract_tokenizer import AbstractTokenizer
from smaragd_shared_python.pre_processing.spacy_tokenizer import SpacyTokenizer
from smaragd_shared_python.report.report import Report
from smaragd_shared_python.report.report_parser import ReportParser
from smaragd_shared_python.report.token import Token

from constants import REPORTS_CSV_FILE_PATH
from constants import MAMMO_REPORTS_CSV_FILE_PATH
import os

DECRYPTION_KEY = os.environ.get("DECRYPTION_KEY", "undefined")


class CorpusLoader:

    def load_corpus(self, returnTokenized = False) -> list[Report]:
        decryption_key = DECRYPTION_KEY if DECRYPTION_KEY != "undefined" else None
        tokenizer = SpacyTokenizer() if returnTokenized else None
        reports = ReportParser.parse_report_csv(
            REPORTS_CSV_FILE_PATH, decryption_key, tokenizer
        )
        return reports

    def load_mammography_corpus(self, returnTokenized = False) -> list[Report]:
        tokenizer = SpacyTokenizer() if returnTokenized else None
        reports = self.parse_mammography_report_csv(
            MAMMO_REPORTS_CSV_FILE_PATH, tokenizer
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

    # FIXME: Add this method adapted to the mammography data to the ReportParser class in the shared lib
    def parse_mammography_report_csv(self, path: str, tokenizer: AbstractTokenizer = None) -> list[Report]:
        reports = []

        csv_file_to_read = open(path, mode='r', encoding="utf8", newline='')

        reader = csv.reader(csv_file_to_read, delimiter=",", quotechar="\"", quoting=QUOTE_ALL)

        for row in reader:
            identifier = row[1]
            report_text = row[2]
            if tokenizer is None:
                report_tokens = []
            else:
                report_tokens = [Token(token[0], token[1]) for token in tokenizer.tokenize(report_text)]
            reports.append(Report(identifier, report_text, report_tokens))

        csv_file_to_read.close()
        return reports



