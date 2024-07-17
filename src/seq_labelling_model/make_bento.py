from transformers import pipeline, AutoTokenizer
import bentoml

import constants
from shared.schema_generator import SchemaGenerator

tokenizer = AutoTokenizer.from_pretrained(constants.BASE_MODEL_NAME)
schema = SchemaGenerator()
# model_checkpoint = constants.SEQ_LABELLING_MODEL_PATH
model_checkpoint = "serialized_models/inselbert_seq_labelling/20240624-140900_inselbert_mammo_03/20240624-115954_CV0"
sequence_labeller = pipeline("ner", model=model_checkpoint, tokenizer=tokenizer, aggregation_strategy="simple")
bentoml.transformers.save_model(name=constants.SEQ_LABELLING_MODEL_NAME, pipeline=sequence_labeller,custom_objects={
    "fact_schema": schema
})