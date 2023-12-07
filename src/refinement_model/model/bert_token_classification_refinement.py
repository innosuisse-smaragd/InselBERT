# Source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py


from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import (
    TokenClassifierOutput,
)


class BertForTokenClassificationRefinement(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels_modifiers = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout_modifiers = nn.Dropout(classifier_dropout)
        self.classifier_modifiers = nn.Linear(config.hidden_size, self.num_labels_modifiers)

        self.dropout_is_start = nn.Dropout(classifier_dropout)
        self.binary_classifier_is_start = nn.Linear(config.hidden_size, 2)

        self.dropout_is_end = nn.Dropout(classifier_dropout)
        self.binary_classifier_is_end = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        # Split into three label tensors
        labels_modifiers: Optional[torch.Tensor] = None,
        labels_isStart: Optional[torch.Tensor] = None,
        labels_isEnd: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[
            0
        ]  # same as outputs.last_hidden_state() -> all other attributes are None

        sequence_output_modifiers = self.dropout_modifiers(sequence_output)
        logits_modifiers = self.classifier_modifiers(sequence_output_modifiers)

        sequence_output_isStart = self.dropout_is_start(sequence_output)
        logits_isStart = self.classifier_isStart(sequence_output_isStart)

        sequence_output_isEnd = self.dropout_is_end(sequence_output)
        logits_isEnd = self.classifier_isEnd(sequence_output_isEnd)

        loss_modifiers = None
        if labels_modifiers is not None:
            loss_fct_modifiers = nn.CrossEntropyLoss()
            loss_modifiers = loss_fct_modifiers(
                logits_modifiers.view(-1, self.num_labels), labels_modifiers.view(-1)
            )

        loss_isStart = None
        if labels_isStart is not None:
            loss_fct_isStart = nn.BCELoss()  # Binary Cross Entropy loss
            loss_isStart = loss_fct_isStart(
                logits_isStart.view(-1, 2), labels_modifiers.view(-1)
            )

        loss_isEnd = None
        if labels_isEnd is not None:
            loss_fct_isEnd = nn.BCELoss()  # Binary Cross Entropy loss
            loss_isEnd = loss_fct_isEnd(
                logits_isEnd.view(-1, 2), labels_modifiers.view(-1)
            )

        loss_averaged = None
        if loss_modifiers or loss_isStart or loss_isEnd:  # TODO: Improve?
            loss_averaged = (loss_modifiers + loss_isStart + loss_isEnd) / 3

        averaged_output = TokenClassifierOutput(loss=loss_averaged)

        modifiers_output = TokenClassifierOutput(
            loss=loss_modifiers,
            logits=logits_modifiers,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        isStart_output = TokenClassifierOutput(
            loss=loss_isStart,
            logits=logits_isStart,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        isEnd_output = TokenClassifierOutput(
            loss=loss_isEnd,
            logits=logits_isEnd,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        return_dict = {
            "loss": averaged_output,
            "modifiers": modifiers_output,
            "isStart": isStart_output,
            "isEnd": isEnd_output,
        }

        return return_dict