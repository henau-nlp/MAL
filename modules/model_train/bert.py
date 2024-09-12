import torch
import os
import sys
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from modules.model_train.crf import CRF

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transformers import BertModel, BertPreTrainedModel
from torch.nn import CrossEntropyLoss

import torch.nn.functional as F


# class BertCrfForNer(BertPreTrainedModel):
#     def __init__(self, config,tag_to_ix):
#         super(BertCrfForNer, self).__init__(config)
#         self.tag_to_ix=tag_to_ix
#         self.bert = BertModel(config)
#
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)
#         """
#                :param in_features: number of features for the input
#                :param tag_to_ix: number of tags. Including [CLS] and [SEP], DO NOT Include START, STOP
#                """
#
#         self.crf = CRF(config.hidden_size, self.tag_to_ix)
#         # self.softmax = nn.Softmax(config.num_labels)
#     def __build_features(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
#
#         # masks = input_ids.gt(0)
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         # 取出最后一层的hidden states,shape=[32,seq_len,768]
#         sequence_output = outputs[0]
#
#         return sequence_output, attention_mask
#
#     def loss(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
#         features, masks = self.__build_features(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
#         loss = self.crf.loss(features, labels, masks=masks)
#         return loss
#
#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
#         features, masks = self.__build_features(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
#         scores, tag_seq, probs = self.crf(features, masks)
#         return scores, tag_seq, probs

class BertCrfForNer(BertPreTrainedModel):
    def __init__(self, config, tag_to_ix):
        super(BertCrfForNer, self).__init__(config)
        self.tag_to_ix = tag_to_ix
        self.bert = BertModel(config)
        # self.bilstm = nn.LSTM(768, 768 // 2, num_layers=1,
        #                       bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.5)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        """
               :param in_features: number of features for the input
               :param tag_to_ix: number of tags. Including [CLS] and [SEP], DO NOT Include START, STOP
               """

        self.crf = CRF(config.hidden_size, self.tag_to_ix)
        unfreeze_layers = ['encoder.layer.11', 'pooler']
        # for name, param in self.bert.named_parameters():
        # if "encoder.layer.11" in name or "pooler" in name:
        #     param.requires_grad = True
        # print(name, param.requires_grad)

        # 冻结bert前面层的参数
        # param.requires_grad = False
        # for ele in unfreeze_layers:
        #     if ele in name:
        #         param.requires_grad = True
        #         break
        # print(name, param.requires_grad)

    def __build_features(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        # masks = input_ids.gt(0)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 取出最后一层的hidden states,shape=[32,seq_len,768]
        sequence_output = outputs[0]
        # 取出[CLS],shape=[32,768]
        cls = outputs[1]
        sequence_output = self.dropout(sequence_output)
        # sequence_output = self.bilstm(sequence_output)
        # sequence_output = self.dropout(sequence_output[0])

        return sequence_output, attention_mask, cls

    def loss(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        features, masks, cls = self.__build_features(input_ids, token_type_ids=token_type_ids,
                                                     attention_mask=attention_mask, labels=labels)
        loss = self.crf.loss(features, labels, masks=masks)

        return loss

    def get_cls(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        features, masks, cls = self.__build_features(input_ids, token_type_ids=token_type_ids,
                                                     attention_mask=attention_mask, labels=labels)
        return cls

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        features, masks, cls = self.__build_features(input_ids, token_type_ids=token_type_ids,
                                                     attention_mask=attention_mask, labels=labels)
        scores, tag_seq, probs = self.crf(features, masks)

        return scores, tag_seq, probs, cls

# class BertSoftmaxForNer(BertPreTrainedModel):
#     def __init__(self, config):
#         super(BertSoftmaxForNer, self).__init__(config)
#         self.num_labels = config.num_labels
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)
#         self.init_weights()
#
#         """
#                :param in_features: number of features for the input
#                :param tag_to_ix: number of tags. Including [CLS] and [SEP], DO NOT Include START, STOP
#                """
#
#         self.softmax = nn.Softmax(config.num_labels)
#     def __build_features(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
#
#         # masks = input_ids.gt(0)
#
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#
#         sequence_output = outputs[0]
#
#         return sequence_output, attention_mask
#
#     def loss(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         sequence_output = outputs[0]
#         sequence_output = self.dropout(sequence_output)
#         logits = self.classifier(sequence_output)
#
#         loss_fct = CrossEntropyLoss(ignore_index=0)
#         # Only keep active parts of the loss
#         if attention_mask is not None:
#             active_loss = attention_mask.view(-1) == 1
#             active_logits = logits.view(-1, self.num_labels)[active_loss]
#             active_labels = labels.view(-1)[active_loss]
#             loss = loss_fct(active_logits, active_labels)
#         else:
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#         return loss
#
#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
#         outputs = self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
#         sequence_output = outputs[0]
#         sequence_output = self.dropout(sequence_output)
#         logits = self.classifier(sequence_output)
#         tag_seq = self.softmax(logits)
#         return tag_seq
#
# class BertSoftmax1ForNer(BertPreTrainedModel):
#     def __init__(self, config):
#         super(BertSoftmax1ForNer, self).__init__(config)
#         self.num_labels = config.num_labels
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)
#         self.loss_type = config.loss_type
#         self.init_weights()
#
#     def forward(self, input_ids, attention_mask=None, token_type_ids=None,labels=None):
#         outputs = self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
#         sequence_output = outputs[0]
#         sequence_output = self.dropout(sequence_output)
#         logits = self.classifier(sequence_output)
#         outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
#         if labels is not None:
#             assert self.loss_type in ['lsr', 'focal', 'ce']
#             if self.loss_type == 'lsr':
#                 loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
#             elif self.loss_type == 'focal':
#                 loss_fct = FocalLoss(ignore_index=0)
#             else:
#                 loss_fct = CrossEntropyLoss(ignore_index=0)
#             # Only keep active parts of the loss
#             if attention_mask is not None:
#                 active_loss = attention_mask.view(-1) == 1
#                 active_logits = logits.view(-1, self.num_labels)[active_loss]
#                 active_labels = labels.view(-1)[active_loss]
#                 loss = loss_fct(active_logits, active_labels)
#             else:
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             outputs = (loss,) + outputs
#         return outputs  # (loss), scores, (hidden_states), (attentions)
