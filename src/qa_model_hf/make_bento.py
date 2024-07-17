from transformers import pipeline, AutoTokenizer
import bentoml

import constants
from shared.schema_generator import SchemaGenerator

schema = SchemaGenerator()
tokenizer = AutoTokenizer.from_pretrained(constants.BASE_MODEL_NAME)
model_checkpoint = "serialized_models/inselbert_qa/inselbert_qa_hf_mammo_03_240620"
# model_checkpoint = constants.QA_HF_MODEL_PATH
question_answerer = pipeline("question-answering", model=model_checkpoint, handle_impossible_answer=True, tokenizer=tokenizer)
bentoml.transformers.save_model(name=constants.QA_HF_MODEL_NAME, pipeline=question_answerer, custom_objects={
    "fact_schema": schema
})