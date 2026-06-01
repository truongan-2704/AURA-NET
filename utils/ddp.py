"""
Multi-GPU training with Distributed Data Parallel (DDP)
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def setup_ddp(rank, world_size):
    """
    Setup DDP environment

    Args:
        rank: Process rank
        world_size: Total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize process group
    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        rank=rank,
        world_size=world_size
    )

    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup DDP"""
    dist.destroy_process_group()


def is_main_process(rank=None):
    """Check if current process is main process"""
    if rank is None:
        rank = dist.get_rank() if dist.is_initialized() else 0
    return rank == 0


def get_rank():
    """Get current process rank"""
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size():
    """Get world size"""
    return dist.get_world_size() if dist.is_initialized() else 1


def reduce_tensor(tensor, world_size=None):
    """
    Reduce tensor across all processes

    Args:
        tensor: Tensor to reduce
        world_size: World size (default: auto-detect)

    Returns:
        reduced_tensor: Reduced tensor
    """
    if not dist.is_initialized():
        return tensor

    if world_size is None:
        world_size = get_world_size()

    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size

    return rt


def reduce_dict(input_dict, world_size=None):
    """
    Reduce dictionary of tensors across all processes

    Args:
        input_dict: Dictionary of tensors
        world_size: World size (default: auto-detect)

    Returns:
        reduced_dict: Dictionary of reduced tensors
    """
    if not dist.is_initialized():
        return input_dict

    if world_size is None:
        world_size = get_world_size()

    reduced_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, torch.Tensor):
            reduced_dict[key] = reduce_tensor(value, world_size)
        else:
            reduced_dict[key] = value

    return reduced_dict


def create_ddp_model(model, rank, find_unused_parameters=False):
    """
    Wrap model with DDP

    Args:
        model: Model to wrap
        rank: Process rank
        find_unused_parameters: Whether to find unused parameters

    Returns:
        ddp_model: DDP wrapped model
    """
    if torch.cuda.is_available():
        model = model.to(rank)
        ddp_model = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=find_unused_parameters
        )
    else:
        ddp_model = DDP(model, find_unused_parameters=find_unused_parameters)

    return ddp_model


def create_ddp_dataloader(dataset, batch_size, rank, world_size, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=False):
    """
    Create dataloader with DistributedSampler

    Args:
        dataset: Dataset
        batch_size: Batch size per GPU
        rank: Process rank
        world_size: World size
        shuffle: Whether to shuffle
        num_workers: Number of workers
        pin_memory: Whether to pin memory
        drop_last: Whether to drop last incomplete batch

    Returns:
        dataloader: DataLoader with DistributedSampler
    """
    from torch.utils.data import DataLoader

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=drop_last
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )

    return dataloader


def save_checkpoint_ddp(model, optimizer, scheduler, epoch, loss, save_path,
                        is_best=False, rank=0, **kwargs):
    """
    Save checkpoint in DDP training (only on main process)

    Args:
        model: DDP model
        optimizer: Optimizer
        scheduler: Scheduler
        epoch: Current epoch
        loss: Current loss
        save_path: Path to save checkpoint
        is_best: Whether this is best model
        rank: Process rank
        **kwargs: Additional items to save
    """
    if rank != 0:
        return

    from utils.checkpoint import save_checkpoint

    # Unwrap DDP model
    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    # Create checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        **kwargs
    }

    # Save
    from pathlib import Path
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)

    if is_best:
        best_path = Path(save_path).parent / 'best.pt'
        torch.save(checkpoint, best_path)


def synchronize():
    """Synchronize all processes"""
    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()
    if world_size == 1:
        return

    dist.barrier()


class DDPLogger:
    """
    Logger that only logs on main process
    """

    def __init__(self, logger, rank=0):
        """
        Args:
            logger: Base logger
            rank: Process rank
        """
        self.logger = logger
        self.rank = rank

    def info(self, msg):
        """Log info message"""
        if self.rank == 0:
            self.logger.info(msg)

    def warning(self, msg):
        """Log warning message"""
        if self.rank == 0:
            self.logger.warning(msg)

    def error(self, msg):
        """Log error message"""
        if self.rank == 0:
            self.logger.error(msg)

    def debug(self, msg):
        """Log debug message"""
        if self.rank == 0:
            self.logger.debug(msg)


def print_ddp(msg, rank=None):
    """
    Print message only on main process

    Args:
        msg: Message to print
        rank: Process rank (default: auto-detect)
    """
    if rank is None:
        rank = get_rank()

    if rank == 0:
        print(msg)


def gather_tensors(tensor):
    """
    Gather tensors from all processes

    Args:
        tensor: Tensor to gather

    Returns:
        gathered: List of tensors from all processes (only on rank 0)
    """
    if not dist.is_initialized():
        return [tensor]

    world_size = get_world_size()
    rank = get_rank()

    # Gather sizes
    local_size = torch.tensor([tensor.size(0)], device=tensor.device)
    size_list = [torch.tensor([0], device=tensor.device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)

    # Gather tensors
    max_size = max([int(size.item()) for size in size_list])

    # Pad tensor to max size
    if tensor.size(0) < max_size:
        padding = torch.zeros(
            max_size - tensor.size(0),
            *tensor.size()[1:],
            dtype=tensor.dtype,
            device=tensor.device
        )
        tensor = torch.cat([tensor, padding], dim=0)

    # Gather
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)

    # Trim padding
    if rank == 0:
        trimmed_list = []
        for t, size in zip(tensor_list, size_list):
            trimmed_list.append(t[:int(size.item())])
        return trimmed_list

    return []


def all_gather_object(obj):
    """
    Gather objects from all processes

    Args:
        obj: Object to gather

    Returns:
        gathered: List of objects from all processes
    """
    if not dist.is_initialized():
        return [obj]

    world_size = get_world_size()

    # Gather
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, obj)

    return gathered
