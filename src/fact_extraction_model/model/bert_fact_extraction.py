import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class BertForFactExtraction(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForFactExtraction, self).__init__(config)
        self.num_labels = num_labels
        # body
        self.bert = BertModel(config)
        # head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size * 2)
        self.linear = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.init_weights()

    def forward(
        self, input_ids, token_type_ids, attention_mask, span_idxs, labels=None
    ):
        outputs = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        ).last_hidden_state

        sub_maxpool, obj_maxpool = [], []
        for bid in range(outputs.size(0)):
            # span includes entity markers, maxpool across span
            sub_span = torch.max(
                outputs[bid, span_idxs[bid, 0] : span_idxs[bid, 1] + 1, :],
                dim=0,
                keepdim=True,
            ).values
            obj_span = torch.max(
                outputs[bid, span_idxs[bid, 2] : span_idxs[bid, 3] + 1, :],
                dim=0,
                keepdim=True,
            ).values
            sub_maxpool.append(sub_span)
            obj_maxpool.append(obj_span)

        sub_emb = torch.cat(sub_maxpool, dim=0)
        obj_emb = torch.cat(obj_maxpool, dim=0)
        rel_input = torch.cat((sub_emb, obj_emb), dim=-1)

        rel_input = self.layer_norm(rel_input)
        rel_input = self.dropout(rel_input)
        logits = self.linear(rel_input)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            return SequenceClassifierOutput(loss, logits)
        else:
            return SequenceClassifierOutput(None, logits)
