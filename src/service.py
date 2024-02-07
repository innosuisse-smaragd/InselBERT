import itertools

import bentoml
import pandas as pd
import torch
from bentoml._internal.io_descriptors import JSON
from pydantic import BaseModel

import constants



class InferenceRequest(BaseModel):
    reportText: str
    fact: str


class InferenceModifier(BaseModel):
    id: str
    span: str


class InferenceFact(BaseModel):
    id: str
    span: str
    start: int
    end: int
    score: float
    alternatives: str
    #anchorEntity: str
    #modifiers: list[InferenceModifier]


class InferenceResponse(BaseModel):
    message: str
    facts: list[InferenceFact]


qa_runner = bentoml.models.get(constants.QA_HF_MODEL_NAME + ":latest").to_runner()
svc = bentoml.Service("inselbert_extract", runners=[qa_runner])


def words_overlap(slice1, slice2):
    if slice1[0] < slice2[0]:  # slice1 is leftmost
        return slice2[0] < slice1[1]  # slice2 ends before slice1 starts
    else:
        return slice1[0] < slice2[1]


@svc.api(input=JSON(pydantic_model=InferenceRequest),
         output=JSON(pydantic_model=InferenceResponse))
async def do_inference(request: InferenceRequest) -> InferenceResponse:
    answers = await qa_runner.async_run(question=request.fact, context=request.reportText, top_k=3)
    print("Raw answers: ",answers)
    facts = []
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
    return InferenceResponse(message="", facts=facts)


