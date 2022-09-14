import sys

sys.path.append("./")
import json
from sklearn.model_selection import train_test_split
# from src.config import BATCH_SIZE, MAX_LEN, EPOCHS, patience, BERT_MODEL, RANDOM_SEED, project_root_path
import torch
from transformers import BertModel, BertTokenizer, XLMRobertaTokenizer, \
    XLMRobertaModel  # ,BertForSequenceClassification
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
import time
import torch.nn.functional as F
import pandas as pd
import numpy as np
import gc
from collections import Counter
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, classification_report, jaccard_score, precision_recall_curve
from src.features.preprocess_feature_creation import create_dataloaders_BERT
from src.pytorchtools import EarlyStopping
from src.helpers import format_time, set_seed
from src.dataset.create_dataset import CreateDataset
from src.models.BERT_biLSTM import BERT_BiLSTM_model


# sys.path.append(project_root_path)
gc.collect()
np.seterr(divide='ignore', invalid='ignore')


class BertBilstm:

    def __init__(self, dataset, drop_neutral, weighted_loss, thresholds_opt, BATCH_SIZE, MAX_LEN, EPOCHS, patience,
                 BERT_MODEL, bidirectional, mlm_weight, RANDOM_SEED, project_root_path):
        self.device = torch.device('cuda')
        print('GPU:', torch.cuda.get_device_name(0))

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
        self.tokenizer.save_pretrained(project_root_path+"/models/tokenizer_simple/")

        self.EPOCHS = EPOCHS
        self.patience = patience
        self.MAX_LEN = MAX_LEN
        self.BATCH_SIZE = BATCH_SIZE
        self.BERT_MODEL = BERT_MODEL
        self.bidirectional = bidirectional
        self.RANDOM_SEED = RANDOM_SEED
        self.weighted_loss = weighted_loss
        self.thresholds_opt = thresholds_opt
        self.project_root_path = project_root_path
        create_dataset = CreateDataset(self.project_root_path)

        if dataset == 'goemotions':
            self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, self.labels, self.weights \
                = create_dataset.goemotions(drop_neutral=drop_neutral, with_weights=weighted_loss)
        # elif dataset == 'ec':
        #     self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test \
        #         = create_dataset.ec(drop_neutral=drop_neutral)
        else:
            print("No dataset with the name {}".format(dataset))

        self.num_labels = len(self.labels)

    def get_weighted_loss(self, logits, labels_true, weights):
        # loss_fn = nn.BCEWithLogitsLoss()
        weights = torch.from_numpy(weights).to(self.device)

        zero_cls = weights[:, 0] ** (1 - labels_true)
        one_cls = weights[:, 1] ** labels_true
        loss = self.loss_fn(logits, labels_true)
        weighted_loss = torch.mean((zero_cls * one_cls) * loss)

        return weighted_loss

    def initialize_model(self, train_dataloader, epochs, num_labels, BERT_MODEL, bidirectional):
        """
        Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
        """
        # Instantiate Bert Classifier
        # classifier = model() #freeze_bert=False)
        classifier = BERT_BiLSTM_model.BertClassifier(num_labels, BERT_MODEL, bidirectional, freeze_bert=False)

        # Tell PyTorch to run the model on GPU
        classifier.to(self.device)

        # Create the optimizer
        optimizer = AdamW(classifier.parameters(),
                          lr=5e-5,  # Default learning rate
                          eps=1e-8  # Default epsilon value
                          )

        # Total number of training steps
        total_steps = len(train_dataloader) * epochs

        # Set up the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value
                                                    num_training_steps=total_steps)

        return classifier, optimizer, scheduler

    def monitor_metrics(self, logits, labels_true):
        if labels_true is None:
            return {}

        probs = torch.sigmoid(logits)
        probs = probs.cpu().detach().numpy()
        labels_true = labels_true.cpu().detach().numpy()

        fpr_micro, tpr_micro, thresholds = metrics.roc_curve(labels_true.ravel(), probs.ravel())
        auc_micro = metrics.auc(fpr_micro, tpr_micro)

        thresholds = []

        for i in range(self.num_labels):
            p, r, th = precision_recall_curve(labels_true[:, i], probs[:, i])
            # print(f"p, r: {p}, {r}")
            f1 = np.nan_to_num((2 * p * r) / (p + r), copy=False)
            f1_max = f1.argmax()
            thresholds.append(th[f1_max])
            # f1_scores.append(f1[f1_max])

        if self.thresholds_opt:
            y_pred = probs > np.asarray(thresholds)
        else:
            y_pred = self.logits_to_labels(probs)

        accuracy = jaccard_score(labels_true, y_pred, average='samples')

        return auc_micro, accuracy, thresholds

    def logits_to_labels(self, logits, threshold=0.3):
        y_pred_labels = np.zeros_like(logits)

        for i in range(logits.shape[0]):
            for j in range(logits.shape[1]):
                if logits[i][j] > threshold:
                    y_pred_labels[i][j] = 1
                else:
                    y_pred_labels[i][j] = 0

        return y_pred_labels

    def evaluate(self, logits, y_true, thresholds=None):
        logits = torch.sigmoid(logits)
        logits = logits.cpu().detach().numpy()

        roc_metrics = []
        j = 0

        for i in self.labels:
            roc = metrics.roc_auc_score(y_true[i].values, logits[:, j])
            roc_metrics.append(roc)
            j = +1

        s = pd.Series(roc_metrics, index=self.labels)
        print(f'AUC metrics for all classes:\n {s}', flush=True)

        if thresholds is not None:
            y_pred = logits > np.asarray(thresholds)
        else:
            y_pred = self.logits_to_labels(logits)

        print(f'Classification report:\n {classification_report(y_true, y_pred)}', flush=True)

        # jaccard score for multilabel classification
        accuracy = jaccard_score(y_true, y_pred, average='samples')
        fscore_micro = f1_score(y_true, y_pred, average='micro')
        fscore_macro = f1_score(y_true, y_pred, average='macro')
        print('Accuracy: %f F1-score micro: %f F1-score macro: %f' % (accuracy, fscore_micro, fscore_macro))

    def validation(self, model, val_dataloader):
        """
        After the completion of each training epoch, measure the model's performance
        on our validation set.
        """
        # Put the model into the evaluation mode. The dropout layers are disabled during
        # the test time.
        model.eval()

        # Tracking variables
        val_accuracy = []
        val_loss = []
        val_auc_micro = []
        val_thresholds = []

        # For each batch in our validation set...
        for batch in val_dataloader:
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(self.device) for t in batch)

            # Compute logits
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask)

            # Compute loss

            # b_labels = b_labels.type(torch.LongTensor).to(self.device)
            b_labels = b_labels.float().to(self.device)
            if self.weighted_loss:
                loss = self.get_weighted_loss(logits, b_labels, self.weights)
            else:
                loss = self.loss_fn(logits, b_labels)

            val_loss.append(loss.item())

            # Get the predictions
            auc_micro, accuracy, thresholds = self.monitor_metrics(logits, b_labels)
            val_auc_micro.append(auc_micro)
            val_accuracy.append(accuracy)
            val_thresholds.append(thresholds)

        # Compute the average accuracy and loss over the validation set.
        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)
        val_auc = np.mean(val_auc_micro)
        val_thresholds = np.mean(val_thresholds, axis=0)

        return val_loss, val_accuracy, val_auc, val_thresholds

    def train(self, model, train_dataloader, val_dataloader=None, optimizer=None, scheduler=None,
              evaluation=False):
        """
        Train the BertClassifier model.
        """
        # Start training loop
        print("Start training...\n", flush=True)
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)
        t0 = time.time()
        threshold_list = []
        thresholds_fin = []

        for epoch_i in range(self.EPOCHS):
            gc.collect()
            # =======================================
            #               Training
            # =======================================
            # Print the header of the result table
            print(
                f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Val Auc': ^9}| {'Elapsed':^12} | {'Elapsed Total':^12}",
                flush=True)
            print("-" * 100, flush=True)

            # Measure the elapsed time of each epoch
            t0_epoch, t0_batch = time.time(), time.time()

            # Reset tracking variables at the beginning of each epoch
            total_loss, batch_loss, batch_counts = 0, 0, 0

            # Put the model into the training mode
            model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):
                batch_counts += 1
                # Load batch to GPU
                b_input_ids, b_attn_mask, b_labels = tuple(t.to(self.device) for t in batch)
                # Zero out any previously calculated gradients
                model.zero_grad()

                # Perform a forward pass. This will return logits.
                logits = model(b_input_ids, b_attn_mask)

                # Compute loss and accumulate the loss values
                b_labels = b_labels.float().to(self.device)

                if self.weighted_loss:
                    loss = self.get_weighted_loss(logits, b_labels, self.weights)
                else:
                    loss = self.loss_fn(logits, b_labels)

                batch_loss += loss.item()
                total_loss += loss.item()

                # Perform a backward pass to calculate gradients
                loss.backward()

                # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and the learning rate
                optimizer.step()
                scheduler.step()

                # Print the loss values and time elapsed for every 20 batches
                if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                    # Calculate time elapsed for 20 batches
                    time_elapsed_batch = format_time(time.time() - t0_batch)
                    time_elapsed_total = format_time(time.time() - t0_epoch)
                    # Print training results
                    print(
                        f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {'-':^9}|{time_elapsed_batch:^12} | {time_elapsed_total:^12}",
                        flush=True)

                    # Reset batch tracking variables
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

            # Calculate the average loss over the entire training data
            avg_train_loss = total_loss / len(train_dataloader)

            print("-" * 100)
            # =======================================
            #               Evaluation
            # =======================================
            if evaluation == True:
                # After the completion of each training epoch, measure the model's performance
                # on our validation set.
                val_loss, val_accuracy, val_auc, thresholds = self.validation(model, val_dataloader)

                # Print performance over the entire training data
                time_elapsed = format_time(time.time() - t0_epoch)
                threshold_list.append(thresholds)

                print(
                    f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Val Auc': ^9}| {'Elapsed':^12} | {'Elapsed Total':^12}",
                    flush=True)
                print("-" * 100, flush=True)
                print(
                    f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {val_auc:^9.6f}| | {'-':^12}| {time_elapsed:^12}",
                    flush=True)
                print("-" * 100)
                early_stopping(val_loss, model)

                if early_stopping.early_stop:
                    print("Early stopping")
                    thresholds_fin = threshold_list[epoch_i - self.patience + 1]
                    break
                else:
                    thresholds_fin = threshold_list[epoch_i]
            print("\n")

        model.load_state_dict(torch.load('checkpoint.pt'))
        print(f"Total training time: {format_time(time.time() - t0)}", flush=True)

        print("Training complete!", flush=True)
        return thresholds_fin

    def bert_predict(self, X_test, y_test=None):
        """
        Perform a forward pass on the trained BERT model to predict probabilities
        on the test set.
        """
        # Put the model into the evaluation mode. The dropout layers are disabled during
        # the test time.
        t0 = time.time()

        if self.bidirectional:
            if self.weighted_loss:
                model = torch.jit.load(self.project_root_path + '/models/model_scripted_BERT_BiLSTM_weighted_loss.pt')
            else:
                model = torch.jit.load(self.project_root_path + '/models/model_scripted_BERT_BiLSTM.pt')
        else:
            if self.weighted_loss:
                model = torch.jit.load(self.project_root_path + '/models/model_scripted_BERT_LSTM_weighted_loss.pt')
            else:
                model = torch.jit.load(self.project_root_path + '/models/model_scripted_BERT_LSTM.pt')

        model.eval()

        tokenizer = BertTokenizer.from_pretrained(self.project_root_path + "/models/tokenizer_simple/")
        test_dataloader = create_dataloaders_BERT(X_test, y_test, tokenizer, self.MAX_LEN, self.BATCH_SIZE, sampler='sequential',
                                                  token_type=False, concept=False)
        all_logits = []

        # For each batch in our test set...
        for batch in test_dataloader:
            # Load batch to GPU
            b_input_ids, b_attn_mask = tuple(t.to(self.device) for t in batch)[:2]

            # Compute logits
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask)
            all_logits.append(logits)

        # Concatenate logits from each batch
        all_logits = torch.cat(all_logits, dim=0)

        # Apply softmax to calculate probabilities
        # probs = F.softmax(all_logits, dim=1).cpu().numpy()

        print(f"Total prediction time: {format_time(time.time() - t0)}", flush=True)

        return all_logits

    def main(self):
        t0=time.time()
        # X_train, y_train, X_val, y_val, X_test, y_test = health_bin()

        train_dataloader = create_dataloaders_BERT(self.X_train, self.y_train, self.tokenizer, self.MAX_LEN, self.BATCH_SIZE,
                                                   sampler='random', token_type=False, concept=False)
        val_dataloader = create_dataloaders_BERT(self.X_val, self.y_val, self.tokenizer, self.MAX_LEN, self.BATCH_SIZE,
                                                 sampler='sequential', token_type=False, concept=False)

        set_seed(self.RANDOM_SEED)  # Set seed for reproducibility

        bert_classifier, optimizer, scheduler = self.initialize_model(train_dataloader,
                                                                      epochs=self.EPOCHS,
                                                                      num_labels=self.num_labels,
                                                                      BERT_MODEL=self.BERT_MODEL,
                                                                      bidirectional=self.bidirectional)

        thresholds = self.train(bert_classifier, train_dataloader, val_dataloader, optimizer, scheduler, evaluation=True)

        for batch in train_dataloader:
            b_input_ids, b_attn_mask = tuple(t.to(self.device) for t in batch)[:2]
            model_scripted = torch.jit.trace(bert_classifier, (b_input_ids, b_attn_mask))

            if self.bidirectional:
                if self.weighted_loss:
                    torch.jit.save(model_scripted, self.project_root_path + '/models/model_scripted_BERT_BiLSTM_weighted_loss.pt')
                    break
                else:
                    torch.jit.save(model_scripted, self.project_root_path + '/models/model_scripted_BERT_BiLSTM.pt')
                    break

            else:
                if self.weighted_loss:
                    torch.jit.save(model_scripted, self.project_root_path + '/models/model_scripted_BERT_LSTM_weighted_loss.pt')
                    break
                else:
                    torch.jit.save(model_scripted, self.project_root_path + '/models/model_scripted_BERT_LSTM.pt')
                    break

        print("Validation set", flush=True)
        logits_val = self.bert_predict(self.X_val)
        if self.thresholds_opt:
            print("The thresholds are: {}".format(thresholds), flush=True)
            self.evaluate(logits_val, self.y_val, thresholds)

        else:
            self.evaluate(logits_val, self.y_val)

        print("Test set", flush=True)
        logits_test = self.bert_predict(self.X_test)
        if self.thresholds_opt:
            print("The thresholds are: {}".format(thresholds), flush=True)
            self.evaluate(logits_test, self.y_test, thresholds)
        else:
            self.evaluate(logits_test, self.y_test)

        print(f"Total training and prediction time: {format_time(time.time() - t0)}", flush=True)
