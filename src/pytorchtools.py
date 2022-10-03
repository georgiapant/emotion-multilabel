import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, metric= 'loss', patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_f1_max = 0
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.metric = metric

    def __call__(self, number, model):

        if self.metric =='loss':
            score = -number

            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint_loss(number, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint_loss(number, model)
                self.counter = 0
        elif self.metric=='f1':
            score = number
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint_f1(number, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint_f1(number, model)
                self.counter = 0
        else:
            print("Invalid early stopping metric")

    def save_checkpoint_loss(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


    def save_checkpoint_f1(self, f1, model):
        '''Saves model when validation f1 increases.'''
        if self.verbose:
            self.trace_func(f'Validation f1 increased ({self.val_f1_max:.6f} --> {f1:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_f1_max= f1

