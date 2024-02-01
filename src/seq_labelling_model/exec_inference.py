from transformers import pipeline

import constants

text="Keine Herdläsion sichtbar."
classifier = pipeline("ner", model=constants.SEQ_LABELLING_MODEL_PATH)
print(classifier(text))