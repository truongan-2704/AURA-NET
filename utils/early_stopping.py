"""
Early Stopping mechanism for training
"""

import numpy as np
import torch


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving

    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss, 'max' for accuracy/mAP
        verbose: Print messages
    """

    def __init__(self, patience=10, min_delta=0.0, mode='max', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            raise ValueError(f'mode {mode} is unknown!')

    def __call__(self, score, epoch):
        """
        Check if should stop

        Args:
            score: Current metric score
            epoch: Current epoch

        Returns:
            should_stop: Whether to stop training
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.verbose:
                print(f'EarlyStopping: Initial score: {score:.4f}')
            return False

        if self.monitor_op(score - self.min_delta, self.best_score):
            # Improvement
            if self.verbose:
                print(f'EarlyStopping: Score improved from {self.best_score:.4f} to {score:.4f}')
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping: No improvement for {self.counter}/{self.patience} epochs')

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'EarlyStopping: Stopping training at epoch {epoch}')
                    print(f'EarlyStopping: Best score {self.best_score:.4f} at epoch {self.best_epoch}')
                return True

        return False

    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
