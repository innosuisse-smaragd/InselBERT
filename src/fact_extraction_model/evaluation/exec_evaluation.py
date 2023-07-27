import torch
import fact_extraction_model.model.bert_two_heads as model_combined

from fact_extraction_model.shared import BASE_MODEL, OUTPUT_DIR

from transformers import AutoTokenizer
import pandas as pd
from seqeval.metrics import classification_report

model = model_combined.BertForFactAndAnchorClassification.from_pretrained(OUTPUT_DIR)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
id2label_anchors = {}  # TODO:
test_dl = {}  # TODO:

# Single-class evaluation


def get_label_and_predicted_tags(batch):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    preds_cpu = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
    labels_cpu = batch["labels"].cpu().numpy()
    labels_list, preds_list = align_predictions(labels_cpu, preds_cpu)
    return labels_list, preds_list


id2label_anchors[-100] = "IGN"
test_labels_list, test_preds_list = [], []
for batch in test_dl:
    labels_list, preds_list = get_label_and_predicted_tags(batch)
    for labels, preds in zip(labels_list, preds_list):
        test_labels_list.append(labels)
        test_preds_list.append(preds)

report = classification_report(test_labels_list, test_preds_list, output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_json(path_or_buf=OUTPUT_DIR + "/classification_report.json")


# TODO: Multi-class evaluation
