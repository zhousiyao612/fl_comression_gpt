from fl.compression.base import Compressor
from fl.compression.ef import ErrorFeedbackBuffer
from fl.compression.topk import topk_sparsify
from fl.compression.quant import quantize_symmetric
import torch

class NoCompression(Compressor):
    def compress_uplink(self, delta: dict, client_id: int):
        bits = self.estimate_full_bits(delta)
        return delta, None, bits

    def decompress_uplink(self, payload, meta=None):
        return payload

class TopKCompressor(Compressor):
    def __init__(self, topk_ratio=0.01, use_ef=False):
        self.topk_ratio = topk_ratio
        self.use_ef = use_ef
        self.ef = ErrorFeedbackBuffer() if use_ef else None

    def compress_uplink(self, delta: dict, client_id: int):
        comp = {}
        meta = {"type": "topk", "topk_ratio": self.topk_ratio}
        bits = 0

        for name, t in delta.items():
            t = t.detach()
            if self.use_ef:
                e = self.ef.get(f"{client_id}:{name}", t.shape, t.device)
                t = t + e

            idx, val, n = topk_sparsify(t, self.topk_ratio)
            comp[name] = (idx.cpu(), val.cpu(), n)

            # bits estimate:
            # idx: store as int32 -> 32 bits each
            # val: float32 -> 32 bits each
            k = idx.numel()
            bits += k * 32 + k * 32

            if self.use_ef:
                # reconstruct sparse update and update error
                recon = torch.zeros(n, device=t.device)
                recon[idx] = val
                recon = recon.view(t.shape)
                new_e = (t - recon).detach()
                self.ef.set(f"{client_id}:{name}", new_e)

        return comp, meta, bits

    def decompress_uplink(self, payload, meta=None):
        out = {}
        for name, pack in payload.items():
            idx, val, n = pack
            idx = idx.to(torch.long)
            val = val.to(torch.float32)
            dense = torch.zeros(n, dtype=torch.float32)
            dense[idx] = val
            out[name] = dense
        return out

class Quant8Compressor(Compressor):
    def __init__(self, bits=8, use_ef=False):
        self.bits = bits
        self.use_ef = use_ef
        self.ef = ErrorFeedbackBuffer() if use_ef else None

    def compress_uplink(self, delta: dict, client_id: int):
        comp = {}
        meta = {"type": "quant", "bits": self.bits}
        bits_total = 0

        for name, t in delta.items():
            t = t.detach()
            if self.use_ef:
                e = self.ef.get(f"{client_id}:{name}", t.shape, t.device)
                t = t + e

            qt, scale = quantize_symmetric(t, self.bits)
            comp[name] = (qt.cpu(), float(scale.cpu().item()), t.numel(), list(t.shape))

            # bits estimate:
            # qt: bits per element
            bits_total += t.numel() * self.bits
            # scale: float32
            bits_total += 32

            if self.use_ef:
                # dequant recon
                recon = (qt.to(torch.float32) * scale).to(t.device).view(t.shape)
                self.ef.set(f"{client_id}:{name}", (t - recon).detach())

        return comp, meta, bits_total

    def decompress_uplink(self, payload, meta=None):
        out = {}
        for name, pack in payload.items():
            qt, scale, n, shape = pack
            qt = qt.to(torch.int16)
            dense = (qt.to(torch.float32) * scale).view(shape)
            out[name] = dense
        return out

def build_compressor(cfg: dict, model):
    if not cfg["compression"]["enabled"]:
        return NoCompression()

    method = cfg["compression"]["method"].lower()
    topk_ratio = cfg["compression"].get("topk_ratio", 0.01)
    qbits = cfg["compression"].get("quant_bits", 8)

    if method == "none":
        return NoCompression()
    if method == "topk":
        return TopKCompressor(topk_ratio=topk_ratio, use_ef=False)
    if method == "topk_ef":
        return TopKCompressor(topk_ratio=topk_ratio, use_ef=True)
    if method == "quant8":
        return Quant8Compressor(bits=qbits, use_ef=False)
    if method == "quant8_ef":
        return Quant8Compressor(bits=qbits, use_ef=True)

    raise ValueError(f"Unknown compression method: {method}")
