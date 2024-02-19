import pandas as pd
from evaluate import evaluator

import plotly.graph_objs as go
import constants
from shared.cas_loader import CASLoader
from shared.dataset_helper import DatasetHelper
from shared.schema_generator import SchemaGenerator

schema = SchemaGenerator()
loader = CASLoader(constants.ANNOTATED_REPORTS_PATH, schema)

train_dictlist, eval_test_dictlist = loader.load_CAS_convert_to_offset_dict_qa_train_test_split()
dataset_qa_train_test_valid = DatasetHelper.create_data_splits_qa(train_dictlist, eval_test_dictlist)
validation_set = dataset_qa_train_test_valid["validation"]

models = [
    constants.QA_HF_MODEL_PATH + "checkpoint-hf",
    constants.QA_MODEL_PATH + "checkpoint-283",
    constants.QA_MODEL_PATH + "checkpoint-849",

]

task_evaluator = evaluator("question-answering")

results = []
for model in models:
    results.append(
        task_evaluator.compute(
            model_or_pipeline=model, data=validation_set, metric="squad_v2", squad_v2_format = True
            )
        )

df = pd.DataFrame(results, index=models)

df = df[["exact", "f1", "total", "HasAns_exact", "HasAns_f1", "HasAns_total","best_exact", "best_exact_thresh","best_f1", "best_f1_thresh","total_time_in_seconds", "samples_per_second", "latency_in_seconds"]]
df.to_csv(constants.QA_HF_MODEL_PATH + "evaluation_results/" + "results.csv")

for col in df.columns:
    fig = go.Figure()
    for model, row in df.iterrows():
        fig.add_trace(go.Bar(x=[model], y=[row[col]], name=model))
    fig.update_layout(title=f'Stacked Bar Chart for {col}', barmode='stack')
    fig.write_html(constants.QA_HF_MODEL_PATH + "evaluation_results/" + f"stacked_bar_chart_{col}.html")


