import torch

def quantize_symmetric(t: torch.Tensor, bits: int):
    # symmetric uniform quant
    qmax = (2 ** (bits - 1)) - 1
    maxv = t.abs().max()
    scale = (maxv / qmax).clamp(min=1e-12)
    qt = torch.round(t / scale).clamp(-qmax, qmax).to(torch.int16)  # store in int16
    return qt, scale
