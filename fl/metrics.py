class BitMeter:
    def __init__(self):
        self.bits = {"uplink": 0, "downlink": 0}

    def add(self, key, nbits):
        self.bits[key] += int(nbits)

    def summary(self, reset=False):
        out = dict(self.bits)
        out["total"] = out["uplink"] + out["downlink"]
        if reset:
            self.bits = {"uplink": 0, "downlink": 0}
        return out
