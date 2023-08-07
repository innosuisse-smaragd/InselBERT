import constants
from transformers import AutoTokenizer, BertConfig
import torch


def getTokenizer():
    tokenizer = AutoTokenizer.from_pretrained(constants.BASE_MODEL_PATH)
    return tokenizer


def getDevice():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def getModel(modelclass, NUM_LABELS, label2id, id2label):
    config = BertConfig.from_pretrained(constants.BASE_MODEL_PATH)  # TODO: Adapt?

    model = modelclass.BertForFactAndAnchorClassification.from_pretrained(
        constants.BASE_MODEL_PATH,
        num_labels=NUM_LABELS,
        label2id=label2id,
        id2label=id2label,  # TODO: add other mappings- but how?
    )
    return model


def getPretrainedModel(modelclass):
    model = modelclass.BertForFactAndAnchorClassification.from_pretrained(
        constants.FINETUNED_MODEL_01_PATH,
    )
    return model
