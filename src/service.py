import itertools

import bentoml
from bentoml.io import JSON
from pydantic import BaseModel

import constants


class InferenceRequest(BaseModel):
    reportText: str
    fact: str


class InferenceFact(BaseModel):
    id: str
    span: str
    start: int
    end: int
    score: float
    alternatives: str
    extracted_tokens: object = {}
    merged_tokens: object = {}
    extracted_entities: object = {}



class InferenceResponse(BaseModel):
    message: str
    facts: list[InferenceFact]


qa_model = bentoml.models.get(constants.QA_HF_MODEL_NAME + ":latest")
qa_runner = qa_model.to_runner()
seq_labelling_runner = bentoml.models.get(constants.SEQ_LABELLING_MODEL_NAME + ":latest").to_runner()

schema = qa_model.custom_objects.get("fact_schema")
svc = bentoml.Service("inselbert_extract", runners=[qa_runner, seq_labelling_runner])


def words_overlap(slice1, slice2):
    if slice1[0] < slice2[0]:  # slice1 is leftmost
        return slice2[0] < slice1[1]  # slice2 ends before slice1 starts
    else:
        return slice1[0] < slice2[1]


@svc.api(input=JSON(pydantic_model=InferenceRequest),
         output=JSON(pydantic_model=InferenceResponse))
async def do_inference(request: InferenceRequest) -> InferenceResponse:
    facts = []
    # Step 1: Extract facts from the report by extractive question answering
    answers = await qa_runner.async_run(question=request.fact, context=request.reportText, top_k=3)
    print("Raw answers: ",answers)

    alternatives = []
    if answers[0]['answer'] == "":
        return InferenceResponse(message="The report does not contain the specified fact.", facts=[])
    else:
        pairs_of_results = itertools.permutations(answers, 2)
        for pair in pairs_of_results:

            if words_overlap((pair[0]['start'], pair[0]['end']), (pair[1]['start'], pair[1]['end'])):
                if pair[0]['score'] > pair[1]['score']:
                    if pair[1] in answers:
                        answers.remove(pair[1])
                        alternatives.append(pair[1])
                else:
                    if pair[0] in answers:
                        answers.remove(pair[0])
                        alternatives.append(pair[0])
        for key, answer in enumerate(answers):
            if answer['score'] < 0.2:  # TODO: maybe make this a parameter
                alternatives.append(answer)
            else:
                facts.append(InferenceFact(id=request.fact, span=answer['answer'], score=answer['score'], start=answer['start'], end=answer['end'], alternatives=str(alternatives)))

    # Step 2: Labelling of anchors and modifiers

    for fact in facts:
        print("Fact: ", fact.id)
        extracted_tokens = await seq_labelling_runner.async_run(inputs=fact.span)
        fact.extracted_tokens = extracted_tokens

# TODO: Maybe sort out tags with low scores

        # Merge subword-tokens
        results = []
        prev_tok = None
        prev_end = 0
        prev_start = 0
        for tag in extracted_tokens:
            if tag['word'].startswith("##") and prev_tok is not None:
                prev_tok += tag['word'][2:]
                prev_end = tag['end']
            else:
                if prev_tok is not None:
                    results.append({"word": prev_tok, "tag": prev_tag, "start": prev_start, "end": prev_end})
                prev_tok = tag['word']
                prev_end = tag['end']
                prev_tag = tag['entity']
                prev_start = tag['start']
        if prev_tok is not None:
            results.append({"word": prev_tok, "tag": prev_tag, "start": prev_start, "end": prev_end})
        fact.merged_tokens = results

        # Merge B- and I- tags to one tag
        merged_tags = []
        prev_word = None
        prev_end = 0
        prev_start = 0
        for tag in fact.merged_tokens:
            if tag['tag'].startswith("I-") and prev_word is not None:
                prev_word += " " + tag['word']
                prev_end = tag['end']
            else:
                if prev_word is not None:
                    merged_tags.append({"word": prev_word, "tag": prev_tag[2:], "start": prev_start, "end": prev_end})
                prev_word = tag['word']
                prev_tag = tag['tag']
                prev_end = tag['end']
                prev_start = tag['start']
        if prev_word is not None:
            merged_tags.append({"word": prev_word, "tag": prev_tag[2:], "start": prev_start, "end": prev_end})
        fact.extracted_entities = merged_tags

    return InferenceResponse(message="", facts=facts)


