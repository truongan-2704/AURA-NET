"""
Learning Rate Finder for optimal learning rate selection
Based on the method from "Cyclical Learning Rates for Training Neural Networks"
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import copy


class LRFinder:
    """
    Learning Rate Finder

    Gradually increases learning rate and records loss to find optimal LR

    Usage:
        lr_finder = LRFinder(model, optimizer, criterion, device)
        lr_finder.range_test(train_loader, start_lr=1e-7, end_lr=10, num_iter=100)
        lr_finder.plot()
        optimal_lr = lr_finder.get_best_lr()
    """

    def __init__(self, model, optimizer, criterion, device='cuda'):
        """
        Args:
            model: PyTorch model
            optimizer: Optimizer
            criterion: Loss function
            device: Device to use
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        # Save initial state
        self.model_state = copy.deepcopy(model.state_dict())
        self.optimizer_state = copy.deepcopy(optimizer.state_dict())

        # Results
        self.lrs = []
        self.losses = []
        self.best_lr = None

    def range_test(self, train_loader, start_lr=1e-7, end_lr=10, num_iter=100,
                   smooth_f=0.05, diverge_th=5):
        """
        Perform learning rate range test

        Args:
            train_loader: Training data loader
            start_lr: Starting learning rate
            end_lr: Ending learning rate
            num_iter: Number of iterations
            smooth_f: Smoothing factor for loss
            diverge_th: Threshold for divergence detection
        """
        print(f'LR Finder: Testing learning rates from {start_lr} to {end_lr}')

        # Reset to initial state
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)

        # Set model to training mode
        self.model.train()

        # Calculate learning rate schedule
        lrs = np.logspace(np.log10(start_lr), np.log10(end_lr), num_iter)

        # Initialize
        self.lrs = []
        self.losses = []
        best_loss = float('inf')
        avg_loss = 0.0
        beta = 1 - smooth_f

        # Create iterator
        iterator = iter(train_loader)

        pbar = tqdm(range(num_iter), desc='LR Finder')

        for iteration in pbar:
            # Get batch
            try:
                imgs, targets, _ = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                imgs, targets, _ = next(iterator)

            imgs = imgs.to(self.device)

            # Set learning rate
            lr = lrs[iteration]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # Forward pass
            self.optimizer.zero_grad()

            predictions, evidence_info = self.model(imgs, return_evidence=True)

            loss, _ = self.criterion(
                predictions, targets,
                evidence_map=evidence_info['evidence_map'],
                route_weights=evidence_info['route_weights']
            )

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Record
            loss_value = loss.item()

            # Smooth loss
            if iteration == 0:
                avg_loss = loss_value
            else:
                avg_loss = beta * avg_loss + (1 - beta) * loss_value

            smoothed_loss = avg_loss / (1 - beta ** (iteration + 1))

            # Track best loss
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss

            # Store
            self.lrs.append(lr)
            self.losses.append(smoothed_loss)

            # Update progress bar
            pbar.set_postfix({'lr': f'{lr:.2e}', 'loss': f'{smoothed_loss:.4f}'})

            # Check for divergence
            if smoothed_loss > diverge_th * best_loss:
                print(f'\nLR Finder: Stopping early due to divergence at LR={lr:.2e}')
                break

        # Restore initial state
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)

        print('LR Finder: Range test completed')

    def plot(self, save_path=None, skip_start=10, skip_end=5):
        """
        Plot learning rate vs loss

        Args:
            save_path: Path to save plot
            skip_start: Skip first N points
            skip_end: Skip last N points
        """
        if len(self.lrs) == 0:
            print('LR Finder: No data to plot. Run range_test() first.')
            return

        # Skip start and end
        lrs = self.lrs[skip_start:-skip_end] if skip_end > 0 else self.lrs[skip_start:]
        losses = self.losses[skip_start:-skip_end] if skip_end > 0 else self.losses[skip_start:]

        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses, linewidth=2)
        plt.xscale('log')
        plt.xlabel('Learning Rate', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Learning Rate Finder', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # Mark suggested LR
        if self.best_lr:
            plt.axvline(self.best_lr, color='red', linestyle='--',
                       label=f'Suggested LR: {self.best_lr:.2e}')
            plt.legend()

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'LR Finder: Plot saved to {save_path}')

        plt.show()

    def get_best_lr(self, skip_start=10, skip_end=5):
        """
        Get suggested learning rate

        Uses the learning rate with steepest negative gradient

        Args:
            skip_start: Skip first N points
            skip_end: Skip last N points

        Returns:
            best_lr: Suggested learning rate
        """
        if len(self.lrs) == 0:
            print('LR Finder: No data. Run range_test() first.')
            return None

        # Skip start and end
        lrs = self.lrs[skip_start:-skip_end] if skip_end > 0 else self.lrs[skip_start:]
        losses = self.losses[skip_start:-skip_end] if skip_end > 0 else self.losses[skip_start:]

        # Find steepest gradient
        gradients = np.gradient(losses)
        min_gradient_idx = np.argmin(gradients)

        self.best_lr = lrs[min_gradient_idx]

        print(f'LR Finder: Suggested learning rate: {self.best_lr:.2e}')

        return self.best_lr

    def get_lr_at_min_loss(self, skip_start=10, skip_end=5):
        """
        Get learning rate at minimum loss

        Args:
            skip_start: Skip first N points
            skip_end: Skip last N points

        Returns:
            lr_at_min: Learning rate at minimum loss
        """
        if len(self.lrs) == 0:
            print('LR Finder: No data. Run range_test() first.')
            return None

        # Skip start and end
        lrs = self.lrs[skip_start:-skip_end] if skip_end > 0 else self.lrs[skip_start:]
        losses = self.losses[skip_start:-skip_end] if skip_end > 0 else self.losses[skip_start:]

        min_loss_idx = np.argmin(losses)
        lr_at_min = lrs[min_loss_idx]

        print(f'LR Finder: Learning rate at minimum loss: {lr_at_min:.2e}')

        return lr_at_min
