import torch


def auto_device_map() -> torch.device:
    """Auto map device based on machine."""
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            device = "mps"

    print(f"[INFO::utils::auto_device_map] device: {device}")
    return torch.device(device)
