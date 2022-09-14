import torch.nn as nn
# from src.config import BERT_MODEL, num_labels
from transformers import BertForSequenceClassification
import torch
import torch.nn.functional as F


class BertClassifier(nn.Module):

    def __init__(self, num_labels, BERT_MODEL, freeze_bert=False):

        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 384, num_labels

        # Instantiate BERT model
        self.bert = BertForSequenceClassification.from_pretrained(BERT_MODEL, problem_type="multi_label_classification")

        self.emo_freq = nn.Linear(10, D_in)
        self.vad_linear = nn.Linear(3, D_in)
        self.fc = nn.Linear(D_in, H)
        self.label = nn.Linear(H, D_out)
        self.dropout = nn.Dropout(0.3)

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def attention_net(self, input_matrix, final_output_cls):
        hidden = final_output_cls
        attn_weights = torch.bmm(input_matrix, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(input_matrix.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state

    def forward(self, input_ids, attention_mask, nrc_feats, vad_vec):
        bert_out = \
            self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]
        last_hidden_state_cls = bert_out[:, 0, :]

        nrc = F.relu(self.emo_freq(nrc_feats))
        vad = F.relu(self.vad_linear(vad_vec))
        combine = torch.cat((bert_out, nrc, vad), dim=1)

        output = self.attention_net(combine, last_hidden_state_cls)
        output = F.relu(self.fc(output))
        output = self.dropout(output)
        logits = self.label(output)

        return logits