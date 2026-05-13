import logging

import torch

LOGGER = logging.getLogger(__name__)


def get_device() -> torch.device:
    """
    Get the best available device for PyTorch operations.

    Priority order: CUDA > MPS (Apple Silicon) > CPU

    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

        device = torch.device("cuda")
        LOGGER.info(f"Device: CUDA ({torch.cuda.get_device_name(0)})")

        return device

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        LOGGER.info("Device: MPS (Apple Silicon)")

        return device

    device = torch.device("cpu")
    LOGGER.warning("Device: CPU (fallback)")

    return device
