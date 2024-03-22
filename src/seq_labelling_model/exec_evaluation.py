# https://huggingface.co/docs/evaluate/base_evaluator#token-classification
import pandas as pd
from datasets import load_dataset, Dataset
from evaluate import evaluator
from transformers import AutoModelForTokenClassification

import constants
from shared.cas_loader import CASLoader
from shared.dataset_helper import DatasetHelper
from shared.model_helper import ModelHelper
from shared.schema_generator import SchemaGenerator
import plotly.graph_objs as go


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
    model_or_pipeline=constants.SEQ_LABELLING_MODEL_PATH + "20240322-162537_medbert/",
    data=dataset_helper.dataset["validation"],
    metric="seqeval",
    label_column="tags",
    #strategy="bootstrap",
    #n_resamples=30,
)


# Assuming eval_results is a dictionary containing evaluation results
# Convert it to a DataFrame for better handling
df = pd.DataFrame(eval_results)

# Assuming eval_results contains the F1 scores as 'f1' keys
# Extract keys and corresponding F1 scores
bar_keys = [key for key in eval_results if isinstance(eval_results[key], dict) and 'f1' in eval_results[key]]
bar_values = [eval_results[key]['f1'] for key in bar_keys]

# Sort the keys and values based on F1 scores in descending order
sorted_data = sorted(zip(bar_keys, bar_values), key=lambda x: x[1], reverse=True)
sorted_keys, sorted_values = zip(*sorted_data)

# Create a Plotly bar chart
fig = go.Figure(data=[go.Bar(x=sorted_keys, y=sorted_values, marker_color='skyblue')])
fig.update_layout(xaxis=dict(tickangle=-45),
                  yaxis=dict(title='F1 Score'),
                  title='F1 Scores for Different Keys',
                  plot_bgcolor='rgba(0,0,0,0)')
fig.show()
fig.write_html(constants.SEQ_LABELLING_MODEL_PATH + "evaluation_results/" + "f1_scores.html")


