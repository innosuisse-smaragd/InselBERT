from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import (
    TokenClassifierOutput,
)


class BertForFactAndAnchorClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels  # same for facts and anchors

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout_facts = nn.Dropout(classifier_dropout)
        self.classifier_facts = nn.Linear(config.hidden_size, config.num_labels)

        self.dropout_anchors = nn.Dropout(classifier_dropout)
        self.classifier_anchors = nn.Linear(config.hidden_size, config.num_labels)

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
        # Split into two label tensors
        labels_anchors: Optional[torch.Tensor] = None,
        labels_facts: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

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

        sequence_output_facts = self.dropout_facts(sequence_output)
        logits_facts = self.classifier_facts(sequence_output_facts)

        sequence_output_anchors = self.dropout_anchors(sequence_output)
        logits_anchors = self.classifier_anchors(sequence_output_anchors)

        loss_averaged = None
        loss_facts = None
        if labels_facts is not None:
            loss_fct_facts = nn.BCEWithLogitsLoss()  # Binary Cross Entropy loss
            loss_facts = loss_fct_facts(logits_facts, labels_facts.float())

        loss_anchors = None
        if labels_anchors is not None:
            loss_fct_anchors = nn.CrossEntropyLoss()
            loss_anchors = loss_fct_anchors(
                logits_anchors.view(-1, self.num_labels), labels_anchors.view(-1)
            )

        if loss_facts or loss_anchors:
            loss_averaged = (loss_facts + loss_anchors) / 2

        facts_output = TokenClassifierOutput(
            loss=loss_facts,
            logits=logits_facts,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        anchors_output = TokenClassifierOutput(
            loss=loss_anchors,
            logits=logits_anchors,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        return_dict = {
            "loss": loss_averaged,
            "anchors": anchors_output,
            "facts": facts_output,
        }

        return return_dict
