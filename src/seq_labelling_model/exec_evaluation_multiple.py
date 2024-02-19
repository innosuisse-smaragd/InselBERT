import pandas as pd
import plotly.graph_objs as go
from datasets import Dataset
from evaluate import evaluator
from transformers import AutoModelForTokenClassification

import constants
from shared.cas_loader import CASLoader
from shared.dataset_helper import DatasetHelper
from shared.model_helper import ModelHelper
from shared.schema_generator import SchemaGenerator

schema = SchemaGenerator()
model_helper = ModelHelper(AutoModelForTokenClassification, schema, constants.SEQ_LABELLING_MODEL_NAME, len(schema.label2id_combined))
loader = CASLoader(constants.ANNOTATED_REPORTS_PATH, schema)
extracted_facts_with_combined_tags = loader.load_CAS_convert_to_combined_tag_list_seq_labelling(encode=False)
dictlist = []

for entry in extracted_facts_with_combined_tags:
    dictlist.append(entry[1])
dataset = Dataset.from_list(dictlist)
dataset_helper = DatasetHelper(dataset, tokenizer=model_helper.tokenizer)

models = [
    constants.SEQ_LABELLING_MODEL_PATH + "checkpoint-366/",
    constants.SEQ_LABELLING_MODEL_PATH + "checkpoint-488/",
    constants.SEQ_LABELLING_MODEL_PATH + "checkpoint-610/",
]

task_evaluator = evaluator("token-classification")

results = []
for model in models:
    results.append(
        task_evaluator.compute(
            model_or_pipeline=model, data=dataset_helper.dataset["validation"], metric="seqeval", label_column="tags",
            )
        )

df = pd.DataFrame(results, index=models)

df = df[["overall_f1", "overall_accuracy", "total_time_in_seconds", "samples_per_second", "latency_in_seconds"]]
df.to_csv(constants.SEQ_LABELLING_MODEL_PATH + "evaluation_results/" + "results.csv")

# Create a list of colors for better visualization
colors = ['blue', 'orange', 'green', 'red', 'purple']

for col, color in zip(df.columns, colors):
    fig = go.Figure()
    for model, row in df.iterrows():
        fig.add_trace(go.Bar(x=[model], y=[row[col]], name=model, marker_color=color))
    fig.update_layout(title=f'Stacked Bar Chart for {col}', barmode='stack')
    fig.write_html(constants.SEQ_LABELLING_MODEL_PATH + "evaluation_results/" + f"stacked_bar_chart_{col}.html")


