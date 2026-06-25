def resolve_device_and_dtype(accelerator: str, precision_id: str) -> tuple:
    import torch

    if accelerator == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda", 0)
    elif accelerator == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    dtype = getattr(torch, precision_id, torch.float32)
    return device, dtype
