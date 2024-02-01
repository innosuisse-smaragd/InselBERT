from transformers import pipeline
import bentoml

model_checkpoint = "./serialized_models/inselbert_qa_hf/"
question_answerer = pipeline("question-answering", model=model_checkpoint, handle_impossible_answer=True)
bentoml.transformers.save_model(name="inselbert_extract_f_qa", pipeline=question_answerer)