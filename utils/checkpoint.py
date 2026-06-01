"""
Checkpoint utilities
"""

import torch
from pathlib import Path


def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_path, is_best=False,
                   config=None, class_names=None, metrics=None):
    """
    Save training checkpoint

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        loss: Current loss
        save_path: Path to save checkpoint
        is_best: Whether this is the best model so far
        config: Model configuration dict
        class_names: List of class names
        metrics: Dictionary of metrics
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'config': config,
        'class_names': class_names,
        'metrics': metrics
    }

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)

    if is_best:
        best_path = Path(save_path).parent / 'best.pt'
        torch.save(checkpoint, best_path)


def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None, device='cpu'):
    """
    Load training checkpoint
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load to
    
    Returns:
        epoch: Epoch number from checkpoint
        loss: Loss from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0.0)
    
    return epoch, loss


def load_model_weights(model, weights_path, device='cpu'):
    """
    Load only model weights (for inference)
    
    Args:
        model: Model to load weights into
        weights_path: Path to weights file
        device: Device to load to
    """
    checkpoint = torch.load(weights_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model
