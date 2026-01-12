import torch

def topk_sparsify(t: torch.Tensor, k_ratio: float):
    flat = t.view(-1)
    k = max(1, int(flat.numel() * k_ratio))
    # pick largest magnitude
    vals, idx = torch.topk(flat.abs(), k, sorted=False)
    sel_idx = idx
    sel_val = flat[sel_idx]
    return sel_idx, sel_val, flat.numel()
