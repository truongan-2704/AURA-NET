"""
TensorBoard logger for AURA-Net training
"""

import torch
from pathlib import Path
import numpy as np


class TensorBoardLogger:
    """
    TensorBoard logger for training visualization
    """

    def __init__(self, log_dir='runs/train', enabled=True):
        """
        Args:
            log_dir: Directory to save TensorBoard logs
            enabled: Whether to enable TensorBoard logging
        """
        self.enabled = enabled
        self.writer = None

        if self.enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.log_dir = Path(log_dir)
                self.log_dir.mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(str(self.log_dir))
                print(f'TensorBoard: Start with "tensorboard --logdir {log_dir}"')
            except ImportError:
                print('Warning: TensorBoard not available. Install with: pip install tensorboard')
                self.enabled = False

    def log_scalar(self, tag, value, step):
        """Log a scalar value"""
        if self.enabled and self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag, tag_scalar_dict, step):
        """Log multiple scalars"""
        if self.enabled and self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_image(self, tag, img, step):
        """
        Log an image

        Args:
            tag: Image tag
            img: Image tensor (C, H, W) or numpy array
            step: Global step
        """
        if self.enabled and self.writer:
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)

            # Ensure correct shape (C, H, W)
            if img.ndim == 2:
                img = img.unsqueeze(0)
            elif img.ndim == 3 and img.shape[0] not in [1, 3, 4]:
                img = img.permute(2, 0, 1)

            self.writer.add_image(tag, img, step)

    def log_images(self, tag, imgs, step, max_images=16):
        """
        Log a batch of images

        Args:
            tag: Image tag
            imgs: Image tensor (B, C, H, W) or (B, H, W, C)
            step: Global step
            max_images: Maximum number of images to log
        """
        if self.enabled and self.writer:
            if isinstance(imgs, np.ndarray):
                imgs = torch.from_numpy(imgs)

            # Ensure correct shape (B, C, H, W)
            if imgs.ndim == 4 and imgs.shape[1] not in [1, 3, 4]:
                imgs = imgs.permute(0, 3, 1, 2)

            # Limit number of images
            imgs = imgs[:max_images]

            # Create grid
            from torchvision.utils import make_grid
            grid = make_grid(imgs, nrow=4, normalize=True)
            self.writer.add_image(tag, grid, step)

    def log_histogram(self, tag, values, step):
        """Log histogram of values"""
        if self.enabled and self.writer:
            self.writer.add_histogram(tag, values, step)

    def log_model_graph(self, model, input_tensor):
        """Log model graph"""
        if self.enabled and self.writer:
            try:
                self.writer.add_graph(model, input_tensor)
            except Exception as e:
                print(f'Warning: Could not log model graph: {e}')

    def log_pr_curve(self, tag, labels, predictions, step):
        """
        Log precision-recall curve

        Args:
            tag: Curve tag
            labels: Ground truth labels (N,)
            predictions: Prediction scores (N,)
            step: Global step
        """
        if self.enabled and self.writer:
            self.writer.add_pr_curve(tag, labels, predictions, step)

    def log_text(self, tag, text, step):
        """Log text"""
        if self.enabled and self.writer:
            self.writer.add_text(tag, text, step)

    def log_hyperparameters(self, hparams, metrics):
        """
        Log hyperparameters and metrics

        Args:
            hparams: Dictionary of hyperparameters
            metrics: Dictionary of metrics
        """
        if self.enabled and self.writer:
            self.writer.add_hparams(hparams, metrics)

    def log_confusion_matrix(self, tag, confusion_matrix, class_names, step):
        """
        Log confusion matrix as image

        Args:
            tag: Tag for confusion matrix
            confusion_matrix: Confusion matrix array (N, N)
            class_names: List of class names
            step: Global step
        """
        if self.enabled and self.writer:
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                confusion_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax
            )
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title('Confusion Matrix')

            # Convert to image
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = torch.from_numpy(img).permute(2, 0, 1)

            self.writer.add_image(tag, img, step)
            plt.close(fig)

    def log_learning_rate(self, optimizer, step):
        """Log current learning rate"""
        if self.enabled and self.writer:
            for i, param_group in enumerate(optimizer.param_groups):
                lr = param_group['lr']
                self.writer.add_scalar(f'Learning_Rate/group_{i}', lr, step)

    def log_model_weights(self, model, step):
        """Log model weights histograms"""
        if self.enabled and self.writer:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.writer.add_histogram(f'Weights/{name}', param.data, step)
                    if param.grad is not None:
                        self.writer.add_histogram(f'Gradients/{name}', param.grad, step)

    def close(self):
        """Close the writer"""
        if self.enabled and self.writer:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
