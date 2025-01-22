from smaragd_shared_python.fact_schema.confluence_fact_schema_loader import ConfluenceFactSchemaLoader

import constants
from ollama import Client
from typing import List, Dict
from pydantic import BaseModel, Field, RootModel

fact_schema_loader = ConfluenceFactSchemaLoader(fact_schema_path=constants.FACT_SCHEMA_PATH)
fact_schema = fact_schema_loader.load_fact_schema()

fact_schema_templates = []

for fact in fact_schema.facts:
    template = {}
    template["fact_class"] = fact.class_name
    template["text"] = ""
    template["entities"] = []
    template["entities"].append({fact.anchor_entity.class_name: ""})
    for modifier in fact.modifiers:
        template["entities"].append({modifier.class_name: ""})
    fact_schema_templates.append(template)

print(fact_schema_templates[0])


class Entity(RootModel[Dict[str, str]]):
    """Flexible model for entity with dynamic keys."""
    pass

class ExtractedFact(BaseModel):
    """Model for a single extracted fact."""
    fact_class: str = Field(..., description="The classification of the extracted fact.")
    entities: List[Entity] = Field(..., description="List of entities with dynamic key-value pairs.")

class RootModelSchema(BaseModel):
    """Root model for the entire JSON structure."""
    extracted_facts: List[ExtractedFact] = Field(..., description="List of extracted facts.")

prompt = "You are an experienced radiologist. Your task is to fill in this structured report template based on the following text: "

example_report = """
"MAMMOGRAPHIE IN ZWEI EBENEN VOM 25.06.2022

Fragestellung/ Indikation:
Mastodynie.
Unauffällige senologische Familienanamnese. Unauffällige senologische Untersuchung.
Auffälligkeiten?

Befund:
Erstuntersuchung.

Cutis und Subcutis unauffällig. Unauffällige Mamillenregion beidseits. Sehr dichtes Drüsengewebe.
Beurteilung bezüglich Herdbefunden nicht möglich. Kein gruppierter, polymorpher Mikrokalk. Keine Architekturstörung, keine retrahierten Areale. 
Keine suspekten Lymphknoten.

Beurteilung:
ACR Typ D beidseits.
BIRADS 0 beidseits.

Weitere Abklärung mittels Sonographie empfohlen."""

prompt = prompt + example_report + "This is the report template: " + str(fact_schema_templates[0]) + " . Only return the specified fact class instances. Fill the attributes with the exact same string as in the text. If attributes are not present in the text, leave the string empty. Return a JSON object."

client = Client(
    host='http://localhost:11434',
    headers={'x-some-header': 'some-value'}
)
response = client.generate(model='llama3:latest', prompt=prompt,format=RootModelSchema.model_json_schema(), stream=False)
response_validated = RootModelSchema.model_validate_json(response['response'])
print(response['response'])
