from transformers import pipeline, AutoTokenizer
import bentoml

import constants

tokenizer = AutoTokenizer.from_pretrained(constants.BASE_MODEL_NAME)
model_checkpoint = constants.SEQ_LABELLING_MODEL_PATH
sequence_labeller = pipeline("ner", model=model_checkpoint, tokenizer=tokenizer, aggregation_strategy="simple")
bentoml.transformers.save_model(name=constants.SEQ_LABELLING_MODEL_NAME, pipeline=sequence_labeller)