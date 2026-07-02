import torch
import torch.nn as nn
from contextlib import nullcontext


class MixedPrecisionTrainer:
    """Handles mixed-precision (FP16/BF16) training for recommendation models.

    Automatically falls back to FP32 on CPU or unsupported hardware.
    """

    def __init__(self, enabled: bool = True, dtype: str = "fp16", device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.enabled = enabled and device.startswith("cuda")

        if dtype == "bf16" and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
        elif dtype == "fp16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float16

        self.scaler = torch.amp.GradScaler(enabled=self.enabled)

    def get_context(self):
        """Return the appropriate autocast context manager."""
        if self.enabled:
            return torch.amp.autocast(device_type="cuda", dtype=self.dtype)
        return nullcontext()

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed-precision backward pass."""
        return self.scaler.scale(loss) if self.enabled else loss

    def step(self, optimizer: torch.optim.Optimizer):
        """Unscale gradients and step optimizer."""
        if self.enabled:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    @property
    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "dtype": str(self.dtype),
            "device": self.device,
            "scaler_scale": self.scaler.get_scale() if self.enabled else 1.0,
        }
