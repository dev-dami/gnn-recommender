import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class DistributedManager:
    """Manages multi-GPU training via DistributedDataParallel.

    Supports single-node multi-GPU and provides helpers for
    process group initialization and cleanup.
    """

    def __init__(self, backend: str = "nccl"):
        self.backend = backend
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.initialized = False

    def setup(self, rank: int = 0, world_size: int = 1):
        """Initialize the process group."""
        if self.initialized:
            return

        if world_size > 1:
            import os
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "12355")
            dist.init_process_group(backend=self.backend, rank=rank, world_size=world_size)
            torch.cuda.set_device(rank)
            self.initialized = True

        self.rank = rank
        self.world_size = world_size
        self.local_rank = rank

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model with DDP if multi-GPU."""
        if self.world_size > 1:
            model = model.to(self.local_rank)
            model = DDP(model, device_ids=[self.local_rank])
        return model

    def cleanup(self):
        """Destroy process group."""
        if self.initialized:
            dist.destroy_process_group()
            self.initialized = False

    def is_main(self) -> bool:
        return self.rank == 0

    def broadcast_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Broadcast loss from rank 0 to all ranks."""
        if self.world_size > 1:
            dist.broadcast(loss, src=0)
        return loss

    @property
    def status(self) -> dict:
        return {
            "backend": self.backend,
            "rank": self.rank,
            "world_size": self.world_size,
            "initialized": self.initialized,
            "gpu_count": torch.cuda.device_count(),
        }
