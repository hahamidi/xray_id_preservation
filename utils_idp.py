import torch
import torch.nn as nn

def fmt_params(n: int) -> str:
    """Format numbers into human-readable units (K, M, B)."""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)

def report_module(name: str, module: nn.Module):
    """Pretty-print info about a module: device, dtype, params, trainable/frozen."""
    if module is None:
        print(f"[REPORT] {name:<15}: None")
        return
    
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    trainable = trainable_params > 0
    try:
        device = next(module.parameters()).device
        dtype = next(module.parameters()).dtype
    except StopIteration:
        device, dtype = "N/A", "N/A"  # in case the module has no params

    print(f"[REPORT] {name:<15}: {module.__class__.__name__:<30} "
          f"on {device} dtype={dtype} "
          f"params={fmt_params(trainable_params)}/{fmt_params(total_params)} "
          f"{'trainable' if trainable else 'frozen'}")