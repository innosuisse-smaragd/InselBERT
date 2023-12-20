import os
from pathlib import Path

from smaragd_shared_python.annotation.document import Document
from smaragd_shared_python.annotation.document_parser import DocumentParser
from smaragd_shared_python.annotation.uima_util import UIMAUtil
from constants import ANNOTATED_REPORTS_PATH
import pandas
import re


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
    def __init__(self, cas_directory: str, schema):
        self.cas_directory = cas_directory
        self.schema = schema

    def load_cas_from_directory(self) -> list[Document]:
        files = os.listdir(self.cas_directory)
        documents = []
        for cas_file in files:
            cas = UIMAUtil.load_smaragd_cas(Path(ANNOTATED_REPORTS_PATH, cas_file))
            document = DocumentParser.parse_smaragd_cas(cas)
            documents.append(document)
        return documents

    def load_cas_and_convert_to_dict_list(self) -> list[dict]:
        documents = self.load_cas_from_directory()
        dictlist = []
        for doc in documents:
            tokens = doc.tokens
            fact_tags = []
            anchor_tags = []
            modifier_tags = []
            token_texts = []

            annotation_classes = ["FACT", "ANCHOR_ENTITY", "MODIFIER"]
            prev_annotations = {annoclass: [] for annoclass in annotation_classes}

            for token in tokens:
                hasFactAnnotations = False
                hasAnchorAnnotations = False
                hasModifierAnnotations = False
                for annoclass in annotation_classes:
                    annotation_tags = []
                    annotations = doc.get_annotation_for_token(token, annoclass)

                    for annotation in annotations:
                        if annoclass == "FACT":
                            value = annotation.fact_value
                            hasFactAnnotations = True
                        elif annoclass == "ANCHOR_ENTITY":
                            value = annotation.anchor_entity_value
                            hasAnchorAnnotations = True
                        elif annoclass == "MODIFIER":
                            value = annotation.modifier_value
                            hasModifierAnnotations = True

                        if value in prev_annotations[annoclass]:
                            annotation_tags.append("I-" + value)
                        else:
                            annotation_tags.append("B-" + value)
                        prev_annotations[annoclass].append(value)

                    if annoclass == "FACT":
                        fact_tags.append(annotation_tags)
                    elif annoclass == "ANCHOR_ENTITY":
                        anchor_tags.append(annotation_tags)
                    elif annoclass == "MODIFIER":
                        modifier_tags.append(annotation_tags)
                if not hasFactAnnotations:
                    prev_annotations["FACT"] = []
                if not hasAnchorAnnotations:
                    prev_annotations["ANCHOR_ENTITY"] = []
                if not hasModifierAnnotations:
                    prev_annotations["MODIFIER"] = []
                text = doc.get_covered_text(token)
                token_texts.append(text)

            entry = {
                "id": doc.document_meta_data.document_id,
                "fact_tags": fact_tags,
                "anchor_tags": anchor_tags,
                "modifier_tags": modifier_tags,
                "text": token_texts,
            }
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

    def load_CAS_convert_to_offset_dict_refinement(self):
        dictlist = self.load_cas_and_convert_to_dict_list()
        extracted_spans = []  # contains list of tuples of fact type and spans
        for index, doc in enumerate(dictlist):
            span_prototypes = {}
            # loop over fact tags and sub lists of tokens and tags
            for i, fact_tags in enumerate(doc["fact_tags"]):
                leading_tokens = doc["text"][i - 20:i]
                trailing_tokens = doc["text"][i:i + 20]
                trailing_padding = [0] * len(trailing_tokens)
                for j, tag in enumerate(fact_tags):
                    fact_type = tag[2:]
                    token = doc["text"][i]
                    anchors = doc["anchor_tags"][i]
                    if len(anchors) == 0:
                        anchors = 0
                    else:
                        anchors = self.schema.label2id_anchors[anchors[0]]
                    modifiers = doc["modifier_tags"][i]
                    if len(modifiers) == 0:
                        modifiers = 0
                    else:
                        modifiers = self.schema.label2id_modifiers[modifiers[0]]
                    if tag.startswith("B-"):  # new fact type
                        if fact_type in span_prototypes.keys():  # there is already a prototype for this fact type
                            # complete previous prototype
                            span_prototypes[fact_type]["tokens"].extend(trailing_tokens)
                            span_prototypes[fact_type]["anchor"].extend(trailing_padding)
                            span_prototypes[fact_type]["modifiers"].extend(trailing_padding)
                            span_prototypes[fact_type]["isStart"][-1] = 1
                            span_prototypes[fact_type]["isStart"].extend([0] * len(trailing_tokens))
                            span_prototypes[fact_type]["isEnd"][-1] = 0
                            span_prototypes[fact_type]["isEnd"].extend([0] * len(trailing_tokens))
                            # add to extracted spans and remove from prototypes

                            extracted_spans.append([fact_type, prototype, doc["id"], doc["text"]])
                            span_prototypes.pop(fact_type)
                        # add token, as well as anchor and modifier tag lists to prototype
                        merged_tokens = [token]
                        merged_tokens[:0] = leading_tokens
                        merged_anchors = [anchors]
                        leading_padding = [0] * len(leading_tokens)
                        merged_anchors[:0] = leading_padding
                        merged_modifiers = [modifiers]
                        merged_modifiers[:0] = leading_padding
                        isStart = [1]
                        isStart[:0] = [0] * len(leading_tokens)
                        isEnd = [0]
                        isEnd[:0] = [0] * len(leading_tokens)
                        span_prototypes[fact_type] = {"tokens": merged_tokens, "anchor": merged_anchors,
                                                      "modifiers": merged_modifiers, "isStart": isStart, "isEnd": isEnd}
                    elif tag.startswith("I-"):
                        # there must be already a fact prototype -> add token, anchors and modifiers
                        span_prototypes[fact_type]["tokens"].append(token)
                        span_prototypes[fact_type]["anchor"].append(anchors)
                        span_prototypes[fact_type]["modifiers"].append(modifiers)
                        span_prototypes[fact_type]["isStart"].append(0)
                        span_prototypes[fact_type]["isEnd"].append(0)

                if len(fact_tags) == 0 and span_prototypes != {}:
                    # no fact tags -> complete all prototypes, add to extracted spans and remove from prototypes
                    for fact_type, prototype in span_prototypes.items():
                        prototype["tokens"].extend(trailing_tokens)
                        prototype["anchor"].extend(trailing_padding)
                        prototype["modifiers"].extend(trailing_padding)
                        prototype["isStart"].extend([0] * len(trailing_tokens))
                        prototype["isEnd"][-1] = 1
                        prototype["isEnd"].extend([0] * len(trailing_tokens))

                        extracted_spans.append([fact_type, prototype, doc["id"], doc["text"]])
                    span_prototypes = {}

            if len(span_prototypes) > 0:
                # end of document -> add prototypes to extracted spans

                # span_prototypes[fact_type]["tokens"].extend(trailing_tokens)
                # span_prototypes[fact_type]["anchor"].extend([None] * len(trailing_tokens))
                # span_prototypes[fact_type]["modifiers"].extend([None] * len(trailing_tokens))
                for fact_type, prototype in span_prototypes.items():
                    prototype["isEnd"][-1] = True
                    extracted_spans.append([fact_type, prototype, doc["id"], doc["text"]])

        return extracted_spans

    def load_CAS_convert_to_offset_dict_qa_single_answer(self, dictlist):
        #dictlist = self.load_CAS_convert_to_offset_dict()
        # Multiplex each entry in dictlist to one entry per fact for training
        dictlist_qa = []
        for doc in dictlist:
            for i, fact in enumerate(doc["fact_tags"]):
                fact_type = fact["tag"]
                fact_answer = doc["text"][fact["start"]:fact["end"]]
                answers = {
                    "text": [fact_answer],
                    "answer_start": [fact["start"]],
                }
                fact_text = doc["text"]
                match = re.search(r'(\d+)\.xmi$', doc["id"])
                fact_id = match.group(1) + "_" + str(i) + "_" + fact_type
                dictlist_qa.append({"id": fact_id, "question": fact_type, "context": fact_text, "answers": answers})

        return dictlist_qa

    def load_CAS_convert_to_offset_dict_qa_multi_answer(self, dictlist):
        #dictlist = self.load_CAS_convert_to_offset_dict()
        # Multiplex each entry in dictlist to one entry per fact for training
        dictlist_qa = []
        for doc in dictlist:
            facts_in_doc = []
            for i, fact in enumerate(doc["fact_tags"]):
                fact_type = fact["tag"]
                fact_text = doc["text"]
                match = re.search(r'(\d+)\.xmi$', doc["id"])
                fact_id = match.group(1) + "_" + str(i) + "_" + fact_type

                fact_answer = doc["text"][fact["start"]:fact["end"]]
                fact_start_index = fact["start"]
                answers = {
                    "text": [fact_answer],
                    "answer_start": [fact_start_index],
                }
                if facts_in_doc == []:
                    facts_in_doc.append(
                        {"id": fact_id, "question": fact_type, "context": fact_text, "answers": answers})
                else:
                    for fact_dict in facts_in_doc:
                        if fact_dict["question"] == fact_type:
                            fact_dict["answers"]["text"].append(fact_answer)
                            fact_dict["answers"]["answer_start"].append(fact_start_index)
                            break
                    else:
                        facts_in_doc.append(
                            {"id": fact_id, "question": fact_type, "context": fact_text, "answers": answers})

            dictlist_qa.extend(facts_in_doc)

        return dictlist_qa

    def load_CAS_convert_to_offset_dict_qa_train_test_split(self):
        dictlist = self.load_CAS_convert_to_offset_dict()
        training = dictlist[:int(len(dictlist) * 0.8)]
        evaluation = dictlist[-int(len(dictlist) * 0.2):]
        train_examples_single = self.load_CAS_convert_to_offset_dict_qa_single_answer(training)
        eval_examples_multi = self.load_CAS_convert_to_offset_dict_qa_multi_answer(evaluation)
        return train_examples_single, eval_examples_multi

