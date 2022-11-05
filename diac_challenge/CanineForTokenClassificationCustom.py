from transformers import CanineForTokenClassification, CanineForSequenceClassification, AdamW, CaninePreTrainedModel, CanineModel
from transformers.modeling_outputs import TokenClassifierOutput
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import copy
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

labels = [
    "no_diac", 
    "ă", 
    'î',
    "â",
    "ș",
    "ț",
    'Î',
    'Â',
    'Ă',
    'Ș',
    'Ț',
]

chars_with_diacritics = ['a','t','i','s', 'A', 'T', 'I', 'S']

id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}

class CanineForTokenClassificationCustom(CaninePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.canine = CanineModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.canine(
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

        for idx, out in enumerate(outputs):
            print(idx, out)
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(weight=torch.tensor(per_label_weights, device="cuda"), reduction='none').cuda()
            # loss_fct = CrossEntropyLoss(weight=torch.tensor(per_label_weights, device="cpu"), reduction='none')
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = loss * attention_mask.flatten()
            loss = loss.sum() / (attention_mask.sum() + 1e-15)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

model = CanineForTokenClassificationCustom.from_pretrained('google/canine-s', 
                                                                     num_labels=len(labels),
                                                                     id2label=id2label,
                                                                     label2id=label2id)

# model(**batch)

import pytorch_lightning as pl
from transformers import CanineForSequenceClassification, AdamW
import torch.nn as nn

class CanineReviewClassifier(pl.LightningModule):
    def __init__(self):
        super(CanineReviewClassifier, self).__init__()
        self.model = CanineForTokenClassificationCustom.from_pretrained('google/canine-s', 
                                                                     num_labels=len(labels),
                                                                     id2label=id2label,
                                                                     label2id=label2id)
    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                             labels=labels)

        return outputs
        
    def common_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        logits = outputs.logits

        predictions = logits.argmax(-1)
        correct = (predictions == batch['labels']).sum().item()
        accuracy = correct/batch['input_ids'].shape[0]

        return loss, accuracy
      
    def training_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        self.log("training_accuracy", accuracy)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_accuracy", accuracy, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     

        return loss

    def configure_optimizers(self):
        # We could make the optimizer more fancy by adding a scheduler and specifying which parameters do
        # not require weight_decay but just using AdamW out-of-the-box works fine
        return AdamW(self.parameters(), lr=LR)

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return test_dataloader