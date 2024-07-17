from datetime import datetime

import pandas as pd
from evaluate import evaluator

import plotly.graph_objs as go
import plotly.express as px
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
    constants.INSELBERT_MULTI_QA_PATH,
    constants.MEDBERT_DE_QA_PATH,
    constants.INSELBERT_MAMMO_QA_PATH_03,
    constants.INSELBERT_MAMMO_QA_PATH_10
]

task_evaluator = evaluator("question-answering")

results = []
# 599 is based on Wilcox, R. R. (2010). Fundamentals of modern statistical methods: Substantially improving power and accuracy. Springer.
for model in models:
    results.append(
        task_evaluator.compute(
            model_or_pipeline=model, data=validation_set, metric="squad_v2", squad_v2_format = True, strategy="bootstrap",
            n_resamples=599 # final: 599
            )
        )

df = pd.DataFrame(results, index=models)
df.to_csv(constants.EVAL_OUTPUT_PATH + datetime.now().strftime("%Y%m%d-%H%M%S") + "evaluation_results_qa.csv")

df = df[["exact", "f1", "total", "HasAns_exact", "HasAns_f1", "HasAns_total","best_exact", "best_exact_thresh","best_f1", "best_f1_thresh","total_time_in_seconds", "samples_per_second", "latency_in_seconds"]]
df.to_csv(constants.QA_HF_MODEL_PATH + "evaluation_results/" + "results.csv")


df.drop(columns=["total", "HasAns_total", "best_exact_thresh", "best_f1_thresh","total_time_in_seconds", "samples_per_second", "latency_in_seconds"], inplace=True)

models = df.index
metrics = df.columns


short_model_names = {
    './serialized_models/inselbert_qa_hf/inselbert_qa_hf_240319': 'InselBERT',
    './serialized_models/inselbert_qa_hf/medbert_de_qa_hf_240319': 'medBERT-de'
}
# Define color mapping for models
color_mapping = {'./serialized_models/inselbert_qa_hf/inselbert_qa_hf_240319': 'blue', './serialized_models/inselbert_qa_hf/medbert_de_qa_hf_240319': 'red'}

# Reshape data for Plotly Express
df_melted = df.reset_index().melt(id_vars='index', var_name='Metric', value_name='Value')

# Replace long model names with shorter ones
df_melted['index'] = df_melted['index'].map(short_model_names)

# Update color mapping with shorter model names
color_mapping_short = {short_model_names[k]: v for k, v in color_mapping.items()}

# Plot figure with customizations
fig = px.bar(df_melted, x='Metric', y='Value', color='index', barmode='group', color_discrete_map=color_mapping_short,
             category_orders={"index": short_model_names.values()},
             labels={'index': 'Model', 'Value': 'Metric Values', 'Metric': 'Metrics'})

# Adjust legend position and font size
fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12)),
                  xaxis_title='Metrics', yaxis_title='Metric Values', font=dict(size=12))

# Adjust plot size
fig.update_layout(width=800, height=500)

fig.show()

fig.write_html("grouped_bar_chart.html")






