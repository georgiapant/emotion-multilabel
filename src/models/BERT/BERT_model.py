import torch.nn as nn
# from src.config import BERT_MODEL, num_labels
from transformers import BertModel
import torch


class BertClassifier(nn.Module):

    def __init__(self, num_labels, BERT_MODEL, freeze_bert=False):

        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, num_labels

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained(BERT_MODEL, problem_type="multi_label_classification", output_hidden_states=True)

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

    def forward(self, input_ids, attention_mask):

        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)


        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)


        return logits