"""
Device selection module with Apple Silicon MPS prioritization.
"""
import torch


def get_device(force_cpu: bool = False, verbose: bool = True) -> torch.device:
    """
    Get the best available device for PyTorch operations.
    
    Priority order: CUDA > MPS (Apple Silicon) > CPU
    
    Args:
        force_cpu: If True, always return CPU device.
        verbose: If True, print device selection info.
    
    Returns:
        torch.device: The selected device.
    """
    if force_cpu:
        device = torch.device("cpu")
        if verbose:
            print("Device: CPU (forced)")
        return device
    
    # Check for CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
        return device
    
    # Check for Apple Silicon MPS
    if torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            device = torch.device("mps")
            if verbose:
                print("Device: MPS (Apple Silicon)")
            return device
    
    # Fallback to CPU
    device = torch.device("cpu")
    if verbose:
        print("Device: CPU")
    return device


def get_device_from_config(config: dict, verbose: bool = True) -> torch.device:
    """
    Get device based on configuration settings.
    
    Args:
        config: Configuration dictionary with 'hardware.device' key.
        verbose: If True, print device selection info.
    
    Returns:
        torch.device: The selected device.
    """
    device_setting = config.get("hardware", {}).get("device", "auto")
    
    if device_setting == "auto":
        return get_device(force_cpu=False, verbose=verbose)
    elif device_setting == "cpu":
        return get_device(force_cpu=True, verbose=verbose)
    elif device_setting == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            print("Warning: MPS requested but not available. Falling back to CPU.")
            return torch.device("cpu")
    elif device_setting == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
    else:
        print(f"Warning: Unknown device '{device_setting}'. Using auto-detection.")
        return get_device(force_cpu=False, verbose=verbose)


def optimize_for_device(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """
    Apply device-specific optimizations to a model.
    
    Args:
        model: PyTorch model to optimize.
        device: Target device.
    
    Returns:
        Optimized model on the target device.
    """
    model = model.to(device)
    
    # Enable TF32 for CUDA if available (faster matmuls)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    return model
