import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau


def create_scheduler(optimizer: optim.Optimizer, scheduler_type: str = "cosine",
                     T_max: int = 100, step_size: int = 30, gamma: float = 0.5,
                     lr_min: float = 1e-6):
    if scheduler_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=lr_min)
    elif scheduler_type == "step":
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, verbose=True)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
