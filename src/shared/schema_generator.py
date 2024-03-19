from smaragd_shared_python.fact_schema.fact_schema_reasoner import FactSchemaReasoner
from smaragd_shared_python.fact_schema.confluence_fact_schema_loader import (
    ConfluenceFactSchemaLoader,
)
from smaragd_shared_python.fact_schema.modifier import Modifier
from smaragd_shared_python.fact_schema.anchor_entity import AnchorEntity
from smaragd_shared_python.fact_schema.fact import Fact
import constants


class SchemaGenerator:

    def __init__(self):
        self.tag2id_facts = self.get_tag2id("facts")
        self.tag2id_anchors = self.get_tag2id("anchors")
        self.tag2id_modifiers = self.get_tag2id("modifiers")
        self.tag2id_anchorsModifiers = self.get_tag2id("combined")

        self.id2tag_facts = self.inverse_2id(self.tag2id_facts)
        self.id2tag_anchors = self.inverse_2id(self.tag2id_anchors)
        self.id2tag_modifiers = self.inverse_2id(self.tag2id_modifiers)
        self.id2tag_combined = self.inverse_2id(self.tag2id_anchorsModifiers)

        self.label2id_facts = self.convert_tags_to_labels(self.tag2id_facts)
        self.label2id_anchors = self.convert_tags_to_labels(self.tag2id_anchors)
        self.label2id_modifiers = self.convert_tags_to_labels(self.tag2id_modifiers)
        self.label2id_combined = self.convert_tags_to_labels(self.tag2id_anchorsModifiers)

        self.id2label_facts = self.inverse_2id(self.label2id_facts)
        self.id2label_anchors = self.inverse_2id(self.label2id_anchors)
        self.id2label_modifiers = self.inverse_2id(self.label2id_modifiers)
        self.id2label_combined = self.inverse_2id(self.label2id_combined)

    @staticmethod
    def get_tag2id(entity):
        tag_names = []
        result = {}
        # Load fact schema
        fact_schema_loader = ConfluenceFactSchemaLoader(fact_schema_path=constants.FACT_SCHEMA_PATH)
        fact_schema = fact_schema_loader.load_fact_schema()
        reasoner = FactSchemaReasoner(fact_schema)

        if entity == "facts":
            facts = reasoner.get_entities(Fact.get_type())
            tag_names = [fact.class_name for fact in facts]
        elif entity == "anchors":
            anchors = reasoner.get_entities(AnchorEntity.get_type())
            tag_names = [anchor.class_name for anchor in anchors]
        elif entity == "modifiers":
            modifiers = reasoner.get_entities(Modifier.get_type())
            tag_names = [modifier.class_name for modifier in modifiers]
        elif entity == "combined":
            anchors = reasoner.get_entities(AnchorEntity.get_type())
            anchor_names = [anchor.class_name for anchor in anchors]
            modifiers = reasoner.get_entities(Modifier.get_type())
            modifier_names = [modifier.class_name for modifier in modifiers]
            tag_names = anchor_names + modifier_names
        if constants.FILTER_ENTITIES:
            tag_names = [tag for tag in tag_names if tag not in constants.ENTITIES_TO_IGNORE]
        tag_names = set(tag_names) # remove duplicates
        i = 1
        for index, element in enumerate(tag_names):
            if element != "":
                result[element] = i
                i += 1
        return result

    @staticmethod
    def convert_tags_to_labels(tag2id: dict):
        return {
            "O": 0,
            **{f"B-{k}": 2 * v - 1 for k, v in tag2id.items()},
            **{f"I-{k}": 2 * v for k, v in tag2id.items()},
        }

    @staticmethod
    def inverse_2id(mapping: dict):
        return {v: k for k, v in mapping.items()}

if __name__ == "__main__":
    schema = SchemaGenerator()
    print(schema.tag2id_anchorsModifiers)
    print(schema.id2label_combined)
