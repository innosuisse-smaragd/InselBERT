# https://huggingface.co/docs/evaluate/base_evaluator

from datasets import Dataset, DatasetDict
from evaluate import evaluator

import constants
from shared.cas_loader import CASLoader
from shared.schema_generator import SchemaGenerator

schema = SchemaGenerator()
loader = CASLoader(constants.ANNOTATED_REPORTS_PATH, schema)
train_set, eval_set = loader.load_CAS_convert_to_offset_dict_qa_train_test_split()

dataset = Dataset.from_list(eval_set)
dataset = DatasetDict(
    {
        "validation": dataset
    }
)

task_evaluator = evaluator("question-answering")

eval_results = task_evaluator.compute(
    model_or_pipeline="./serialized_models/inselbert_qa_hf/medbert_de_qa_hf_240319",
    data=dataset["validation"],
    metric="squad_v2",
    squad_v2_format=True,
    strategy="bootstrap",
    n_resamples=30
)

print(eval_results)

data = eval_results
