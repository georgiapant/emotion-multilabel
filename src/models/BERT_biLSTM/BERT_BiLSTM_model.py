import torch.nn as nn
# from src.config import BERT_MODEL, num_labels
from transformers import BertModel
import torch


class BertClassifier(nn.Module):

    def __init__(self, num_labels, BERT_MODEL, bidirectional, freeze_bert=False):

        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 200, num_labels  # orig hidden layers 50

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained(BERT_MODEL,
                                              problem_type="multi_label_classification")  # ,  config=config)

        self.bert_drop = nn.Dropout(0.3)

        if bidirectional == True:
            self.LSTM = nn.LSTM(D_in, H, num_layers=1, bidirectional=True, batch_first=True)
            self.linear = nn.Linear(2 * H, D_out)
        else:
            self.LSTM = nn.LSTM(D_in, H, num_layers=1, bidirectional=False, batch_first=True)
            self.linear = nn.Linear(H, D_out)

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):

        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        lstm, _ = self.LSTM(output)
        drop = self.bert_drop(lstm[:, 0, :])  # do not take the
        logits = self.linear(drop)

        return logits