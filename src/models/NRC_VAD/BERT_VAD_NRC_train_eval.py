import torch
from transformers import BertTokenizer
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
import time
import pandas as pd
import numpy as np
import gc
from sklearn import metrics
from sklearn.metrics import f1_score, classification_report, jaccard_score, precision_recall_curve
from src.features.preprocess_feature_creation import create_dataloaders_BERT, nrc_feats, vad_feats
from src.pytorchtools import EarlyStopping
from src.helpers import format_time, set_seed, LinearLR, MyCosineAnnealingWarmupRestarts
from src.dataset.create_dataset import CreateDataset
from src.models.NRC_VAD import BERT_VAD_NRC_model

gc.collect()
np.seterr(divide='ignore', invalid='ignore')


class BertVadNrc:

    def __init__(self, dataset, drop_neutral, weighted_loss, thresholds_opt, BATCH_SIZE, MAX_LEN, EPOCHS, patience,
                 BERT_MODEL, bidirectional, mlm_weight, RANDOM_SEED, project_root_path, es, scheduler, use_sparsemax):

        self.device = torch.device('cuda')
        print('GPU:', torch.cuda.get_device_name(0))

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
        self.tokenizer.save_pretrained(project_root_path + "/models/tokenizer/")

        self.es = es
        self.EPOCHS = EPOCHS
        self.patience = patience
        self.MAX_LEN = MAX_LEN
        self.BATCH_SIZE = BATCH_SIZE
        self.BERT_MODEL = BERT_MODEL
        self.RANDOM_SEED = RANDOM_SEED
        self.weighted_loss = weighted_loss
        self.thresholds_opt = thresholds_opt
        self.scheduler = scheduler
        self.project_root_path = project_root_path
        self.use_sparsemax = use_sparsemax
        create_dataset = CreateDataset(self.project_root_path)

        if dataset == 'goemotions':
            self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, self.labels, self.weights \
                = create_dataset.goemotions(drop_neutral=drop_neutral, with_weights=weighted_loss)
        elif dataset == 'ec':
            self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, self.labels, self.weights \
                = create_dataset.ec(with_weights=weighted_loss)
        else:
            print("No dataset with the name {}".format(dataset))

        self.num_labels = len(self.labels)

    def get_weighted_loss(self, logits, labels_true, weights):
        weights = torch.Tensor(weights).to(self.device)
        zero_cls = weights[:, 0] ** (1 - labels_true)
        one_cls = weights[:, 1] ** labels_true
        loss = self.loss_fn(logits, labels_true)
        weighted_loss = torch.mean((zero_cls * one_cls) * loss)
        return weighted_loss

    def initialize_model(self, train_dataloader, epochs, num_labels, BERT_MODEL):
        """
        Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
        """
        # Instantiate Bert Classifier
        # classifier = model() #freeze_bert=False)
        classifier = BERT_VAD_NRC_model.BertClassifier(num_labels, BERT_MODEL,
                                                       freeze_bert=False, use_sparsemax=self.use_sparsemax)

        # Tell PyTorch to run the model on GPU
        classifier.to(self.device)

        # Create the optimizer
        optimizer = AdamW(classifier.parameters(),
                          lr=5e-5,  # Default learning rate (default: 5e-5)
                          eps=1e-8  # Default epsilon value
                          )

        # Total number of training steps
        total_steps = len(train_dataloader) * epochs

        if self.scheduler == 'linear':
            # Set up the learning rate scheduler
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=0,  # Default value
                                                        num_training_steps=total_steps)
            return classifier, optimizer, scheduler

        elif self.scheduler == 'chained':
            scheduler = LinearLR(optimizer, start_factor=1, end_factor=0, total_iters=len(train_dataloader) * 10)
            scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_dataloader), gamma=0.75)

            return classifier, optimizer, scheduler, scheduler2

        elif self.scheduler == 'cosine':
            scheduler = MyCosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=len(train_dataloader),
                                                        cycle_mult=1.0, max_lr=5e-5, min_lr=0, warmup_steps=0,
                                                        gamma=0.75)
            return classifier, optimizer, scheduler
        else:
            print("No known scheduler. Scheduler options are: linear, chained, cosine")

        # return classifier, optimizer, scheduler

    def monitor_metrics(self, logits, labels_true):

        if labels_true is None:
            return {}

        probs = torch.sigmoid(logits)
        probs = probs.cpu().detach().numpy()
        labels_true = labels_true.cpu().detach().numpy()

        fpr_micro, tpr_micro, _ = metrics.roc_curve(labels_true.ravel(), probs.ravel())
        auc_micro = metrics.auc(fpr_micro, tpr_micro)

        thresholds = []

        for i in range(self.num_labels):
            p, r, th = precision_recall_curve(labels_true[:, i], probs[:, i])
            # print(f"p, r: {p}, {r}")
            f1 = np.nan_to_num((2 * p * r) / (p + r), copy=False)
            f1_max = f1.argmax()
            thresholds.append(th[f1_max])

        if self.thresholds_opt:
            y_pred = probs > np.asarray(thresholds)
        else:
            y_pred = self.logits_to_labels(probs)

        accuracy = jaccard_score(labels_true, y_pred, average='samples')
        fscore_macro = f1_score(labels_true, y_pred, average='macro')

        return auc_micro, accuracy, thresholds, fscore_macro

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
        probs = torch.sigmoid(logits)
        probs = probs.cpu().detach().numpy()

        roc_metrics = []
        j = 0

        for i in self.labels:
            roc = metrics.roc_auc_score(y_true[i].values, probs[:, j])
            roc_metrics.append(roc)
            j = +1

        s = pd.Series(roc_metrics, index=self.labels)
        print(f'AUC metrics for all classes:\n {s}', flush=True)

        if thresholds is not None:
            y_pred = probs > np.asarray(thresholds)
        else:
            y_pred = self.logits_to_labels(probs)
        # print(y_true)
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
        # val_accuracy = []
        val_loss = []
        # val_auc_micro = []
        # val_thresholds = []

        logits_all = []
        b_labels_all = []

        # For each batch in our validation set...
        for batch in val_dataloader:
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(self.device) for t in batch)
            lex_feats = nrc_feats(b_input_ids, self.tokenizer).to(self.device)
            vad = vad_feats(b_input_ids, self.tokenizer, self.MAX_LEN, self.project_root_path).to(self.device)

            # Compute logits
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask, lex_feats, vad)

            # Compute loss

            # b_labels = b_labels.type(torch.LongTensor).to(self.device)
            b_labels = b_labels.float().to(self.device)

            if self.weighted_loss:
                loss = self.get_weighted_loss(logits, b_labels, self.weights)
            else:
                loss = self.loss_fn(logits, b_labels)

            val_loss.append(loss.item())

            # Get the predictions
            # auc_micro, accuracy, thresholds= self.monitor_metrics(logits, b_labels)
            # val_auc_micro.append(auc_micro)
            # val_accuracy.append(accuracy)
            # val_thresholds.append(thresholds)

            logits_all.append(logits)
            b_labels_all.append(b_labels)

        # all_logits.append(logits)

        # Concatenate logits from each batch
        logits_all = torch.cat(logits_all, dim=0)
        b_labels_all = torch.cat(b_labels_all, dim=0)

        val_auc, val_accuracy, val_thresholds, val_f1 = self.monitor_metrics(logits_all, b_labels_all)

        # Compute the average accuracy and loss over the validation set.
        val_loss = np.mean(val_loss)
        # val_accuracy = np.mean(val_accuracy)
        # val_auc = np.mean(val_auc_micro)
        # val_thresholds = np.mean(val_thresholds, axis=0)

        return val_loss, val_accuracy, val_auc, val_thresholds, val_f1

    def train(self, model, train_dataloader, val_dataloader=None, optimizer=None, scheduler=None,
              evaluation=False, scheduler2=None):
        """
        Train the BertClassifier model.
        """
        # Start training loop
        print("Start training...\n", flush=True)

        early_stopping = EarlyStopping(metric=self.es, patience=self.patience, verbose=True)
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
                f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9}"
                f" | {'Val Auc': ^9}| {'Elapsed':^12} | {'Elapsed Total':^12}",
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
                lex_feats = nrc_feats(b_input_ids, self.tokenizer).to(self.device)
                vad = vad_feats(b_input_ids, self.tokenizer, self.MAX_LEN, self.project_root_path).to(self.device)
                # Zero out any previously calculated gradients
                model.zero_grad()

                # Perform a forward pass. This will return logits.
                logits = model(b_input_ids, b_attn_mask, lex_feats, vad)

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
                if self.scheduler == 'chained':
                    scheduler2.step()

                # Print the loss values and time elapsed for every 20 batches
                if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                    # Calculate time elapsed for 20 batches
                    time_elapsed_batch = format_time(time.time() - t0_batch)
                    time_elapsed_total = format_time(time.time() - t0_epoch)
                    # Print training results
                    print(
                        f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} "
                        f"| {'-':^9}|{time_elapsed_batch:^12} | {time_elapsed_total:^12}",
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
            if evaluation:
                # After the completion of each training epoch, measure the model's performance
                # on our validation set.
                val_loss, val_accuracy, val_auc, thresholds, val_f1 = self.validation(model, val_dataloader)

                # Print performance over the entire training data
                time_elapsed = format_time(time.time() - t0_epoch)
                threshold_list.append(thresholds)

                print(
                    f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} "
                    f"| {'Val Auc': ^9}| {'Elapsed':^12} | {'Elapsed Total':^12}",
                    flush=True)
                print("-" * 100, flush=True)
                print(
                    f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} "
                    f"| {val_auc:^9.6f}| | {'-':^12}| {time_elapsed:^12}",
                    flush=True)
                print("The validation f1_score is: {}".format(val_f1))
                print("-" * 100)

                if self.es == 'f1':
                    early_stopping(val_f1, model)
                elif self.es == 'loss':
                    early_stopping(val_loss, model)
                else:
                    torch.save(model.state_dict(), self.project_root_path + '/models/checkpoint.pt')

                if early_stopping.early_stop:
                    print("Early stopping")
                    thresholds_fin = threshold_list[epoch_i - self.patience + 1]
                    break
                else:
                    thresholds_fin = threshold_list[epoch_i]

        model.load_state_dict(torch.load('checkpoint.pt'))

        print("\n")
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

        tokenizer = BertTokenizer.from_pretrained(self.project_root_path + "/models/tokenizer/")
        test_dataloader = create_dataloaders_BERT(X_test, y_test, tokenizer, self.MAX_LEN, self.BATCH_SIZE,
                                                  sampler='sequential', token_type=False, concept=False)

        if self.use_sparsemax:
            if self.scheduler=='chained':
                model, optimizer, scheduler, scheduler2 = self.initialize_model(test_dataloader, epochs=self.EPOCHS,
                                                                    num_labels=self.num_labels,
                                                                    BERT_MODEL=self.BERT_MODEL)
            else:
                model, optimizer, scheduler = self.initialize_model(test_dataloader, epochs=self.EPOCHS,
                                                                    num_labels=self.num_labels, BERT_MODEL=self.BERT_MODEL)

            if self.weighted_loss:
                model.load_state_dict(torch.load(self.project_root_path + '/models/model_sparse_BERT_vad_nrc_weighted_loss.pt'))
            else:
                model.load_state_dict(torch.load(self.project_root_path + '/models/model_sparse_BERT_vad_nrc.pt'))

        else:
            if self.weighted_loss:
                model = torch.jit.load(self.project_root_path + '/models/model_scripted_BERT_vad_nrc_weighted_loss.pt')

            else:
                model = torch.jit.load(self.project_root_path + '/models/model_scripted_BERT_vad_nrc.pt')

        model.eval()

        all_logits = []

        # For each batch in our test set...
        for batch in test_dataloader:
            # Load batch to GPU
            b_input_ids, b_attn_mask = tuple(t.to(self.device) for t in batch)[:2]
            lex_feats = nrc_feats(b_input_ids, self.tokenizer).to(self.device)
            vad = vad_feats(b_input_ids, self.tokenizer, self.MAX_LEN, self.project_root_path).to(self.device)

            # Compute logits
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask, lex_feats, vad)
            all_logits.append(logits)

        # Concatenate logits from each batch
        all_logits = torch.cat(all_logits, dim=0)
        print(f"Total prediction time: {format_time(time.time() - t0)}", flush=True)

        return all_logits

    def main(self):
        t0 = time.time()

        train_dataloader = create_dataloaders_BERT(self.X_train, self.y_train, self.tokenizer, self.MAX_LEN,
                                                   self.BATCH_SIZE, sampler='random', token_type=False, concept=False)
        val_dataloader = create_dataloaders_BERT(self.X_val, self.y_val, self.tokenizer, self.MAX_LEN, self.BATCH_SIZE,
                                                 sampler='sequential', token_type=False, concept=False)

        set_seed(self.RANDOM_SEED)  # Set seed for reproducibility

        if self.scheduler == 'chained':
            bert_classifier, optimizer, scheduler, scheduler2 = self.initialize_model(train_dataloader,
                                                                                      epochs=self.EPOCHS,
                                                                                      num_labels=self.num_labels,
                                                                                      BERT_MODEL=self.BERT_MODEL)
            thresholds = self.train(bert_classifier, train_dataloader, val_dataloader, optimizer, scheduler,
                                    evaluation=True, scheduler2=scheduler2)

        else:
            bert_classifier, optimizer, scheduler = self.initialize_model(train_dataloader,
                                                                          epochs=self.EPOCHS,
                                                                          num_labels=self.num_labels,
                                                                          BERT_MODEL=self.BERT_MODEL)

            thresholds = self.train(bert_classifier, train_dataloader, val_dataloader, optimizer, scheduler,
                                    evaluation=True)

        # self.train(bert_classifier, train_dataloader, val_dataloader, optimizer, scheduler, evaluation=True)

        if self.use_sparsemax:
            if self.weighted_loss:
                torch.save(bert_classifier.state_dict(),
                           self.project_root_path + '/models/model_sparse_BERT_vad_nrc_weighted_loss.pt')

            else:
                torch.save(bert_classifier.state_dict(),
                           self.project_root_path + '/models/model_sparse_BERT_vad_nrc.pt')
        else:
            for batch in train_dataloader:
                b_input_ids, b_attn_mask = tuple(t.to(self.device) for t in batch)[:2]
                lex_feats = nrc_feats(b_input_ids, self.tokenizer).to(self.device)
                vad = vad_feats(b_input_ids, self.tokenizer, self.MAX_LEN, self.project_root_path).to(self.device)

                model_scripted = torch.jit.trace(bert_classifier, (b_input_ids, b_attn_mask, lex_feats, vad))

                if self.weighted_loss:
                    torch.jit.save(model_scripted,
                                   self.project_root_path + '/models/model_scripted_BERT_vad_nrc_weighted_loss.pt')
                    break
                else:
                    torch.jit.save(model_scripted, self.project_root_path + '/models/model_scripted_BERT_vad_nrc.pt')
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
