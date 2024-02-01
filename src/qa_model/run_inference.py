from transformers import pipeline

import pprint
import bentoml

# Replace this with your own checkpoint
model_checkpoint = "./serialized_models/inselbert_qa_hf/"
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
ACR-Klassifizierung: 2
"""
question = "ACR Klassifizierung erwähnt"
answer = question_answerer(question=question, context=context, top_k=3)
pprint.pprint(answer)