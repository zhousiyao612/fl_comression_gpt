import torch

class ErrorFeedbackBuffer:
    def __init__(self):
        self.buf = {}

    def get(self, name, shape, device):
        if name not in self.buf:
            self.buf[name] = torch.zeros(shape, device=device)
        return self.buf[name]

    def set(self, name, value):
        self.buf[name] = value
