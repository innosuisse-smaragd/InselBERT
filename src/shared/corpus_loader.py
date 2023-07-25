from smaragd_shared_python.pre_processing.spacy_tokenizer import SpacyTokenizer
from smaragd_shared_python.report.report_parser import ReportParser
from smaragd_shared_python.report.report import Report

from constants import REPORTS_CSV_FILE_PATH


class CorpusLoader:
    def load_corpus(self) -> list[Report]:
        # functionality added by Jonas in next version
        tokenizer = SpacyTokenizer()
        reports = ReportParser.parse_report_csv(REPORTS_CSV_FILE_PATH, tokenizer)
        return reports

    def convert_corpus_to_dataset(self):
        reports = self.load_corpus()
        tokens_per_report = [
            [report.report_text[token.begin : token.end] for token in report.tokens]
            for report in reports
        ]
        print(tokens_per_report)


loader = CorpusLoader()
loader.convert_corpus_to_dataset()
