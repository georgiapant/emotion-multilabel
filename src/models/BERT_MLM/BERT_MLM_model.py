import torch.nn as nn
# from src.config import BERT_MODEL, num_labels, MAX_LEN
from transformers import BertModel, AutoModel, AutoModelForMaskedLM
import torch
import numpy as np
import torch.nn.functional as F

class BertClassifier_MLM(nn.Module):

    def __init__(self, num_labels, BERT_MODEL, freeze_bert=False):

        super(BertClassifier_MLM, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 100, num_labels

        # Instantiate BERT model
        self.bert = AutoModelForMaskedLM.from_pretrained(BERT_MODEL, output_hidden_states=True) #problem_type="multi_label_classification"

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(H, D_out),
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False


    def forward(self, input_ids, token_type_ids, attention_mask):

        outputs = self.bert(input_ids=input_ids, token_type_ids = token_type_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state_cls = outputs.hidden_states[-1][:, 0]
        logits = self.classifier(last_hidden_state_cls)

        return logits


    def mlmForward(self, input_ids, token_type_ids, attention_mask, labels):
        # BERT forward
        outputs = self.bert(input_ids, token_type_ids, attention_mask, labels=labels)

        return outputs.loss
