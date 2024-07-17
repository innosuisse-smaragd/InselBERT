from transformers import pipeline

import constants

text="Keine Herdläsion sichtbar."
classifier = pipeline("ner", model='./serialized_models/inselbert_seq_labelling/20240624-160000_inselbert_mammo_10/20240624-150516_CV0/')
print(classifier(text))