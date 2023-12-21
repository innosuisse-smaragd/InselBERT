from transformers import pipeline

import constants
import pprint

# Replace this with your own checkpoint
model_checkpoint = constants.QA_MODEL_PATH + "checkpoint-963"
question_answerer = pipeline("question-answering", model=model_checkpoint, handle_impossible_answer=True)

context = """
Klinische Angaben:

Screening-Mammographie
Befund:

Bilaterale Mammographie wurde durchgeführt.
Keine auffälligen Mikroverkalkungen oder Gruppierungen identifiziert.
Keine pathologischen Massen oder asymmetrischen Dichten in beiden Brüsten.
Architektur der Brustgewebe erscheint unauffällig.
Beurteilung:
Zwei kleine Mikroverkalkungen im oberen rechten Quadranten.
Die Mammographiebefunde sind im Rahmen des Normalen. Keine Anzeichen von malignen Läsionen oder anderen pathologischen Veränderungen.
Empfehlung: Regelmäßiges Mammographiescreening gemäß den aktuellen Leitlinien.
BIRADS-level: 6. 
"""
question = "ACR Klassifizierung erwähnt"
answer = question_answerer(question=question, context=context, top_k=3)
pprint.pprint(answer)