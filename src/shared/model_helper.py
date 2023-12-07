import os

import numpy as np
from matplotlib import pyplot as plt
from seqeval.metrics import f1_score
from sklearn.metrics import multilabel_confusion_matrix

import constants
from transformers import AutoTokenizer, BertConfig
import torch


class ModelHelper:

    def __init__(self, model_def, num_labels, schema):
        self.tokenizer = self.get_tokenizer()
        self.device = self.get_device()
        self.model_def = model_def
        self.num_labels = num_labels
        self.model = self.get_further_pretrained_model()
        self.model.to(self.device)
        self.schema = schema

    @staticmethod
    def get_tokenizer():
        tokenizer = AutoTokenizer.from_pretrained(constants.BASE_MODEL_NAME) # TODO: Adapt!
        return tokenizer

    @staticmethod
    def get_device():
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def get_further_pretrained_model(self):
        config = BertConfig.from_pretrained(constants.PRETRAINED_MODEL_PATH)  # TODO: Adapt?

        model = self.model_def.BertForFactAndAnchorClassification.from_pretrained(
            constants.BASE_MODEL_NAME,
            num_labels=self.num_labels
           # label2id=label2id,
           # id2label=id2label,  # TODO: add other mappings- but how?
        )
        return model

    def get_finetuned_model(self):
        model = self.model_def.BertForFactAndAnchorClassification.from_pretrained(
            constants.F_A_EXTRACTION_MODEL_PATH,
        )
        return model

    # Metrics for single class
    def align_predictions(self,labels_cpu, preds_cpu):
        # remove -100 labels from score computation
        batch_size, seq_len = preds_cpu.shape
        labels_list, preds_list = [], []
        for bid in range(batch_size):
            example_labels, example_preds = [], []
            for sid in range(seq_len):
                # ignore label -100
                if labels_cpu[bid, sid] != -100:
                    example_labels.append(self.schema.id2label_anchors[labels_cpu[bid, sid]])
                    example_preds.append(self.schema.id2label_anchors[preds_cpu[bid, sid]])
            labels_list.append(example_labels)
            preds_list.append(example_preds)
        return labels_list, preds_list


    def compute_metrics_for_anchors(self,labels, logits):
        # convert logits to predictions and move to CPU
        preds_cpu = torch.argmax(logits, dim=-1).cpu().numpy()
        labels_cpu = labels.cpu().numpy()
        labels_list, preds_list = self.align_predictions(labels_cpu, preds_cpu)
        # seqeval.metrics.f1_score takes list of list of tags
        return f1_score(labels_list, preds_list)

    # Metrics for multilabel
    @staticmethod
    def divide(a: int, b: int):
        return a / b if b > 0 else 0

    def compute_metrics_for_facts(self,p):
        """
        Customize the `compute_metrics` of `transformers`
        Args:
            - p (tuple):      2 numpy arrays: predictions and true_labels
        Returns:
            - metrics (dict): f1 score on
        """
        predictions, true_labels = p
        predicted_labels = np.where(
            predictions > 0, np.ones(predictions.shape), np.zeros(predictions.shape)
        )
        metrics = {}

        cm = multilabel_confusion_matrix(
            true_labels.reshape(-1, self.num_labels), predicted_labels.reshape(-1, self.num_labels)
        )

        for label_idx, matrix in enumerate(cm):
            if label_idx == 0:
                continue  # We don't care about the label "O"
            tp, fp, fn = matrix[1, 1], matrix[0, 1], matrix[1, 0]
            precision = self.divide(tp, tp + fp)
            recall = self.divide(tp, tp + fn)
            f1 = self.divide(2 * precision * recall, precision + recall)
            metrics[f"f1_{self.schema.id2label_facts[label_idx]}"] = f1

        macro_f1 = sum(list(metrics.values())) / (self.num_labels - 1)
        metrics["macro_f1_facts"] = macro_f1

        return metrics

    @staticmethod
    def save_training_history(history, model_dir):
        fhist = open(os.path.join(model_dir, "history.tsv"), "w")
        for epoch, train_loss, eval_loss, eval_score in history:
            fhist.write(
                "{:d}\t{:.5f}\t{:.5f}\t{:.5f}\n".format(
                    epoch, train_loss, eval_loss, eval_score
                )
            )
        fhist.close()

    @staticmethod
    def save_micro_metrics(micro_metrics, model_dir):
        fhist = open(os.path.join(model_dir, "fact_metrics.tsv"), "w")
        for epoch, item in micro_metrics:
            fhist.write("{:d}\t".format(epoch))
            for key, value in item.items():
                fhist.write("{}\t{:.5f}\t".format(key, value))
            fhist.write("\n")
        fhist.close()

    @staticmethod
    def make_loss_diagram(history, path):
        plt.subplot(2, 1, 1)
        plt.plot([train_loss for _, train_loss, _, _ in history], label="train")
        plt.plot([eval_loss for _, _, eval_loss, _ in history], label="validation")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend(loc="best")

        plt.subplot(2, 1, 2)
        plt.plot([eval_score for _, _, _, eval_score in history], label="validation")
        plt.xlabel("epochs")
        plt.ylabel("f1-score")
        plt.legend(loc="best")

        plt.tight_layout()
        plt.savefig(path + "/loss.png", dpi=300)








