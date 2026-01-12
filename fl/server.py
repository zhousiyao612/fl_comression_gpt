import random
import torch
from copy import deepcopy
from fl.utils import get_model_state, set_model_state
from fl.metrics import estimate_full_model_bits

class Server:
    def __init__(self, model, compressor, device):
        self.model = model
        self.device = device
        self.compressor = compressor

    def sample_clients(self, clients, k: int):
        return random.sample(clients, k)

    def broadcast(self):
        state = get_model_state(self.model)
        # 下行通常不压缩也行；这里也支持压缩（可选）
        payload, bits = self.compressor.compress_downlink(state)
        return payload, bits

    def aggregate(self, client_payloads):
        # FedAvg on (delta) updates
        # client_payloads: list of { "delta": compressed, "weight": n_samples, "meta": ... }
        total = sum(p["weight"] for p in client_payloads)
        agg_delta = None

        for p in client_payloads:
            weight = p["weight"] / total
            delta = self.compressor.decompress_uplink(p["delta"], p.get("meta", None))
            if agg_delta is None:
                agg_delta = {k: v * weight for k, v in delta.items()}
            else:
                for k in agg_delta:
                    agg_delta[k] += delta[k] * weight

        # apply update
        state = get_model_state(self.model)
        for k in state:
            state[k] = state[k] + agg_delta[k].to(state[k].device)
        set_model_state(self.model, state)

    @torch.no_grad()
    def evaluate(self, test_loader):
        self.model.eval()
        correct, total = 0, 0
        for x, y in test_loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
        return correct / max(total, 1)
