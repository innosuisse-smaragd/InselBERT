import os
from pathlib import Path

from dataset_entry import Dataset_Entry

from smaragd_shared_python.annotation.document import Document
from smaragd_shared_python.annotation.document_parser import DocumentParser
from smaragd_shared_python.annotation.uima_util import UIMAUtil
from constants import ANNOTATED_REPORTS_PATH


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

    def load_cas_and_convert_to_dict_list(self) -> list[Dataset_Entry]:
        documents = self.load_cas_from_directory()
        dictlist = [Dataset_Entry]
        for doc in documents:
            tokens = doc.tokens
            fact_tags = []
            anchor_tags = []
            modifier_tags = []
            token_texts = []

            annotation_classes = ["FACT", "ANCHOR_ENTITY", "MODIFIER"]

            prev_fact_annotation = None
            prev_anchor_annotation = None
            prev_modifier_annotation = None
            for token in tokens:
                fact_annotation_tags = []
                fact_annotations = doc.get_annotation_for_token(token, "FACT")

                for annotation in fact_annotations:
                    fact_value = annotation.fact_value
                    if prev_fact_annotation != fact_value:
                        fact_annotation_tags.append("B-" + fact_value)
                    else:
                        fact_annotation_tags.append("I-" + fact_value)
                    print("Value: ", fact_value)
                    print("Prev", prev_fact_annotation)
                    prev_fact_annotation = fact_value

                anchor_annotation_tags = []
                anchor_annotations = doc.get_annotation_for_token(
                    token, "ANCHOR_ENTITY"
                )

                for annotation in anchor_annotations:
                    anchor_entity_value = annotation.anchor_entity_value
                    if prev_anchor_annotation != anchor_entity_value:
                        anchor_annotation_tags.append("B-" + anchor_entity_value)
                    else:
                        anchor_annotation_tags.append("I-" + anchor_entity_value)
                    prev_anchor_annotation = anchor_entity_value

                modifier_annotation_tags = []
                modifier_annotations = doc.get_annotation_for_token(token, "MODIFIER")

                for annotation in modifier_annotations:
                    modifier_value = annotation.modifier_value
                    if prev_modifier_annotation != modifier_value:
                        modifier_annotation_tags.append("B-" + modifier_value)
                    else:
                        modifier_annotation_tags.append("I-" + modifier_value)
                    prev_modifier_annotation = modifier_value
                text = doc.get_covered_text(token)
                token_texts.append(text)
                fact_tags.append(fact_annotation_tags)
                anchor_tags.append(anchor_annotation_tags)
                modifier_tags.append(modifier_annotation_tags)
            entry = Dataset_Entry(
                doc.document_meta_data.document_id,
                fact_tags,
                anchor_tags,
                modifier_tags,
                token_texts,
            )
            dictlist.append(entry)
        return dictlist


loader = CASLoader(ANNOTATED_REPORTS_PATH)
dictlist = loader.load_cas_and_convert_to_dict_list()
print(str(dictlist[1].tokens))
print(str(dictlist[1].fact_tags))
print(str(dictlist[1].anchor_tags))
print(str(dictlist[1].modifier_tags))
