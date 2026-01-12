import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_model_state(model):
    # keep tensors on CPU for payload transport simulation
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def set_model_state(model, state_dict):
    model.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in state_dict.items()}, strict=True)

def model_delta(new_state, old_state):
    return {k: (new_state[k] - old_state[k]) for k in new_state}
