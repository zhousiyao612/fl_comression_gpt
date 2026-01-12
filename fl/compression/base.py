from abc import ABC, abstractmethod

class Compressor(ABC):
    @abstractmethod
    def compress_uplink(self, delta: dict, client_id: int):
        pass

    @abstractmethod
    def decompress_uplink(self, payload, meta=None):
        pass

    def compress_downlink(self, state_dict: dict):
        # default: no compression on downlink
        bits = self.estimate_full_bits(state_dict)
        return {"state_dict": state_dict, "compressor_ref": self}, bits

    def estimate_full_bits(self, state_dict: dict):
        # float32 by default
        total_elems = 0
        for _, v in state_dict.items():
            total_elems += v.numel()
        return total_elems * 32
