import os
from pathlib import Path

from smaragd_shared_python.annotation.document import Document
from smaragd_shared_python.annotation.document_parser import DocumentParser
from smaragd_shared_python.annotation.uima_util import UIMAUtil
from constants import ANNOTATED_REPORTS_PATH


class DatasetEntry:
    def __init__(
        self,
        id: str,
        fact_tags: list[list[str]],
        anchor_tags: list[list[str]],
        modifier_tags: list[list[str]],
        tokens: list[str],
    ):
        self.id = id
        self.fact_tags = fact_tags
        self.anchor_tags = anchor_tags
        self.modifier_tags = modifier_tags
        self.tokens = tokens


class CASLoader:
    def __init__(self, cas_directory: str):
        self.cas_directory = cas_directory

    def load_cas_from_directory(self) -> list[Document]:
        files = os.listdir(self.cas_directory)
        documents = []
        for cas_file in files:
            # jonas adds decryption in load_smaragd_cas()
            cas = UIMAUtil.load_smaragd_cas(Path(ANNOTATED_REPORTS_PATH, cas_file))
            document = DocumentParser.parse_smaragd_cas(cas)
            documents.append(document)
        return documents

    def load_cas_and_convert_to_dict_list(self) -> list[DatasetEntry]:
        documents = self.load_cas_from_directory()
        dictlist = [DatasetEntry]
        for doc in documents:
            tokens = doc.tokens
            fact_tags = []
            anchor_tags = []
            modifier_tags = []
            token_texts = []

            annotation_classes = ["FACT", "ANCHOR_ENTITY", "MODIFIER"]
            prev_annotation = None
            for token in tokens:
                for annoclass in annotation_classes:
                    annotation_tags = []
                    annotations = doc.get_annotation_for_token(token, annoclass)

                    for annotation in annotations:
                        if annoclass == "FACT":
                            value = annotation.fact_value
                        elif annoclass == "ANCHOR_ENTITY":
                            value = annotation.anchor_entity_value
                        elif annoclass == "MODIFIER":
                            value = annotation.modifier_value

                        if prev_annotation != value:
                            annotation_tags.append("B-" + value)
                        else:
                            annotation_tags.append("I-" + value)
                        prev_annotation = value
                    if annoclass == "FACT":
                        fact_tags.append(annotation_tags)
                    elif annoclass == "ANCHOR_ENTITY":
                        anchor_tags.append(annotation_tags)
                    elif annoclass == "MODIFIER":
                        modifier_tags.append(annotation_tags)
                text = doc.get_covered_text(token)
                token_texts.append(text)

            entry = DatasetEntry(
                doc.document_meta_data.document_id,
                fact_tags,
                anchor_tags,
                modifier_tags,
                token_texts,
            )
            dictlist.append(entry)
        return dictlist

    def load_CAS_convert_to_offset_dict(self) -> list[dict]:
        documents = self.load_cas_from_directory()
        dictlist = []
        for doc in documents:
            modifier_annotations = []
            fact_annotations = []
            anchor_annotations = []
            modifier_tags = doc.get_modifier_annotations()
            for modifier in modifier_tags:
                obj = {
                    "start": modifier.begin,
                    "end": modifier.end,
                    "tag": modifier.get_value(),
                }
                modifier_annotations.append(obj)
            anchor_tags = doc.get_anchor_entity_annotations()
            for anchor in anchor_tags:
                obj = {
                    "start": anchor.begin,
                    "end": anchor.end,
                    "tag": anchor.get_value(),
                }
                anchor_annotations.append(obj)
            for fact in doc.facts:
                obj = {
                    "start": fact.begin,
                    "end": fact.end,
                    "tag": fact.get_value(),
                }
                fact_annotations.append(obj)
            entry = {
                "id": doc.document_meta_data.document_id,
                "fact_tags": fact_annotations,
                "anchor_tags": anchor_annotations,
                "modifier_tags": modifier_annotations,
                "text": doc.document_text,
            }
            dictlist.append(entry)
        return dictlist
