import ast
import csv
import json
import os

from bs4 import BeautifulSoup
from smaragd_shared_python.fact_schema.confluence_fact_schema_loader import ConfluenceFactSchemaLoader

import constants
from ollama import Client
from typing import List, Dict
from pydantic import BaseModel, Field, RootModel

from llama_evaluation.example_template import EXAMPLE_TEMPLATE
from llama_evaluation.prompt_template import PROMPT_TEMPLATE
from shared.cas_loader import CASLoader
from shared.schema_generator import SchemaGenerator


class Entity(RootModel[Dict[str, str]]):
    """Flexible model for entity with dynamic keys."""
    pass

class ExtractedFact(BaseModel):
    """Model for a single extracted fact."""
    fact_text: str = Field(..., description="The text of the extracted fact.")
    entities: List[Entity] = Field(..., description="List of entities with dynamic key-value pairs.")

class RootModelSchema(BaseModel):
    """Root model for the entire JSON structure."""
    extracted_fact_instances: List[ExtractedFact] = Field(..., description="List of extracted facts.")
    fact_class: str = Field(..., description="The name of the extracted fact.")

class TemplateFillingEvaluator:
    def __init__(self,):
        self.client = Client(host='http://eris.ti.bfh.ch:11434')
        self.fact_schema = self.generate_fact_schema()
        self.reports = self.read_reports_from_csv()

    def evaluate_reports(self):
        # Initialize results to aggregate the results for all reports
        all_results = {}

        for report in self.reports:
            report_text = report["content"]
            results_facts = []

            # Iterate over the fact schema with better variable naming
            for index, fact_template in enumerate(self.fact_schema):
                # Prepare the prompt for the current fact template
                prompt = self.prepare_prompt(report_text, fact_template)

                # Generate the response using the client
                response = self.client.generate(
                    model='deepseek-r1:70b',
                    prompt=prompt,
                    format=RootModelSchema.model_json_schema(),
                    stream=False,
                    options={"num_ctx": 10000}
                )

                ## TODO: llama3.3, weniger shots, ohne few-shot learning, evaluation-script

                try:
                    response_json = json.loads(response['response'])  # Ensure response is a valid JSON object
                except json.JSONDecodeError:
                    response_json = {"error": "Invalid JSON response", "raw_response": response['response']}
                # Debug: Print the raw response
                results_facts.append({"prompt": prompt,"report_text": report_text,"response": response_json})

            # Add the facts results for the current report to the final results
            all_results[report["id"]] = results_facts

            # Write results to file for the current report
            self.write_results_to_file(results_facts, report["id"])

        # Return all results aggregated
        return all_results

    def write_results_to_file(self, results, filename:str):
        output_dir = "./data/output/llm_evaluation/"
        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

        file_path = os.path.join(output_dir, f"{filename}.json")
        with open(file_path, "w") as file:
            json.dump(results, file,ensure_ascii=False, indent=4)

    @staticmethod
    def generate_fact_schema():
        fact_schema_loader = ConfluenceFactSchemaLoader(fact_schema_path=constants.FACT_SCHEMA_PATH)
        fact_schema = fact_schema_loader.load_fact_schema()

        fact_schema_templates = []

        for fact in fact_schema.facts: # TODO: if fact not in list_to_ignore
            # Create the new template structure
            template = {
                "fact_class": fact.class_name,
                "extracted_fact_instances": [
                    {
                        "fact_text": "",  # Replaces "text"
                        "entities": []  # Entities remain in a list
                    }
                ]
            }

            # Add the anchor entity to the entities list
            template["extracted_fact_instances"][0]["entities"].append({fact.anchor_entity.class_name: ""})

            # Add all modifiers to the entities list
            for modifier in fact.modifiers:
                template["extracted_fact_instances"][0]["entities"].append({modifier.class_name: ""})
            fact_schema_templates.append(template)

        return fact_schema_templates



    def prepare_prompt(self, document: str, fact_template: str):
        examples = self.get_examples_for(fact_template,5)

        prompt = PROMPT_TEMPLATE.format(
            EXAMPLES=examples,
            DOCUMENT=document,
            REPORT_TEMPLATE=fact_template,
            FACT_CLASS=fact_template["fact_class"]
        )
        print("Prompting for fact: ", str(fact_template["fact_class"]))
        return prompt

    def get_examples_for(self, fact_template: str, shots=5): #TODO: Adhere to samee train/test split for quantitative evaluation
        schema = SchemaGenerator()
        loader = CASLoader(constants.ANNOTATED_REPORTS_PATH, schema)
        fact_name = fact_template["fact_class"]
        extracted_facts_with_combined_tags = loader.load_CAS_convert_to_combined_tag_list_seq_labelling()
        filtered_examples = [entry for entry in extracted_facts_with_combined_tags if entry[0] == fact_name]
        filtered_examples = filtered_examples[:shots]

        examples_str = ""
        for example in filtered_examples:
            new_instance = fact_template.copy()  # Copy the fact_template
            extracted_fact_instance = new_instance["extracted_fact_instances"][0].copy()  # Copy the instance template

            tokens = example[1]["tokens"]
            tags = example[1]["tags"]

            # Reset the entities in the copied extracted_fact_instance
            for entity in extracted_fact_instance["entities"]:
                for key in entity.keys():
                    entity[key] = ""  # Reset all fields in the entity to empty strings

            # Populate fact_text
            extracted_fact_instance["fact_text"] = " ".join(tokens)

            # Update entities
            for index, tag in enumerate(tags):

                tag_name = schema.id2label_combined[tag]
                tag_name = tag_name[2:]  # Remove the "B-" or "I-" prefix
                for entity in extracted_fact_instance["entities"]:
                    if tag_name in entity:
                        entity[tag_name] += f" {tokens[index]}"

            # Replace the extracted_fact_instances list with the new instance
            new_instance["extracted_fact_instances"] = [extracted_fact_instance]
            report_text = " ".join(example[3])
            # Serialize and append
            result = json.dumps(new_instance, ensure_ascii=False)
            formatted_example = EXAMPLE_TEMPLATE.format(
                EXAMPLE=result,
                REPORT_TEXT=report_text,
                TEMPLATE=fact_template,
                FACT_CLASS=fact_template["fact_class"]
            )
            examples_str += formatted_example + "\n"

        return examples_str

    def read_reports_from_csv(self):
        with open("./data/synthetic_reports/Smaragd_synthetic_reports.csv", 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader, None)  # skip the headers
            reports = []
            for row in csv_reader:
                if len(row) < 3:
                    continue  # skip rows that don't have at least 3 columns

                id_padded = str(row[0]).zfill(3)
                # Extract the first two columns for the filename
                filename = f"SwissMammo_{id_padded}_BR{row[1]}.txt"

                # The third column is the file content
                content = row[2]
                reports.append({"id": id_padded, "filename": filename, "content": content})
        return reports

if __name__ == '__main__':
    evaluator = TemplateFillingEvaluator()
    results = evaluator.evaluate_reports()
    print(results)



