import constants
from transformers import AutoTokenizer, BertConfig
import torch


def getTokenizer():
    tokenizer = AutoTokenizer.from_pretrained(constants.BASE_MODEL_NAME) # TODO: Adapt!
    return tokenizer

def getDevice():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def getFurtherPretrainedModel(modelclass, num_labels):
    config = BertConfig.from_pretrained(constants.PRETRAINED_MODEL_PATH)  # TODO: Adapt?

    model = modelclass.BertForFactAndAnchorClassification.from_pretrained(
        constants.BASE_MODEL_NAME,
        num_labels=num_labels
       # label2id=label2id,
       # id2label=id2label,  # TODO: add other mappings- but how?
    )
    return model


def getFinetunedModel(modelclass):
    model = modelclass.BertForFactAndAnchorClassification.from_pretrained(
        constants.F_A_EXTRACTION_MODEL_PATH,
    )
    return model

