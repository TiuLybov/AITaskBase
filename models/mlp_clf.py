import torch
import torch.nn as nn
from transformers import AutoModel

import logging

logger = logging.getLogger(__name__)


class MLPTaskClassifier(nn.Module):
    def __init__(self,
                 num_classes,
                 bert_path,
                 linear_shape,
                 dropout=0.3):
        super(MLPTaskClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_path)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(linear_shape, num_classes)

    def forward(self,
                ids,
                attention_mask):
        outputs = self.bert(
            input_ids=ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.pooler_output
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits


