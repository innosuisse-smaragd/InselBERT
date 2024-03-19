# https://huggingface.co/docs/evaluate/base_evaluator#token-classification

from datasets import load_dataset, Dataset
from evaluate import evaluator
from transformers import AutoModelForTokenClassification

import constants
from shared.cas_loader import CASLoader
from shared.dataset_helper import DatasetHelper
from shared.model_helper import ModelHelper
from shared.schema_generator import SchemaGenerator
import matplotlib.pyplot as plt


schema = SchemaGenerator()
model_helper = ModelHelper(AutoModelForTokenClassification, schema, constants.SEQ_LABELLING_MODEL_NAME, len(schema.label2id_combined))


loader = CASLoader(constants.ANNOTATED_REPORTS_PATH, schema)
extracted_facts_with_combined_tags = loader.load_CAS_convert_to_combined_tag_list_seq_labelling(encode=False)
dictlist = []

for entry in extracted_facts_with_combined_tags:
    dictlist.append(entry[1])



dataset = Dataset.from_list(dictlist)
dataset_helper = DatasetHelper(dataset, tokenizer=model_helper.tokenizer)

task_evaluator = evaluator("token-classification")

eval_results = task_evaluator.compute(
    model_or_pipeline=constants.SEQ_LABELLING_MODEL_PATH + "checkpoint-610",
    data=dataset_helper.dataset["validation"],
    metric="seqeval",
    label_column="tags",
    #strategy="bootstrap",
    #n_resamples=30,
)

print(eval_results)
data = eval_results


# Separate keys and values for better visualization
bar_keys = [key for key in data if isinstance(data[key], dict)]
bar_values = [data[key]['f1'] if isinstance(data[key], dict) and 'f1' in data[key] else None for key in bar_keys]


# Create bar plot for F1 scores
plt.bar(bar_keys, bar_values, color='skyblue')
plt.xlabel('Key')
plt.ylabel('F1 Score')
plt.title('F1 Scores for Different Keys')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("f1_scores.png")

