import torch
from copy import deepcopy
from fl.utils import set_model_state, get_model_state, model_delta
from fl.models import build_model_for_client

class Client:
    def __init__(self, cid, device, cfg, train_loader):
        self.cid = cid
        self.device = device
        self.cfg = cfg
        self.train_loader = train_loader
        self.n_samples = len(train_loader.dataset)

        # 为客户端构建模型（结构与server一致）
        self.model = build_model_for_client(cfg).to(device)

    def train_one_round(self, global_payload):
        # 1) load global weights
        global_state = global_payload["state_dict"]
        set_model_state(self.model, global_state)

        # 2) local train
        self.model.train()
        opt_name = self.cfg["federated"]["optimizer"]
        lr = self.cfg["federated"]["lr"]

        if opt_name.lower() == "sgd":
            opt = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=self.cfg["federated"].get("momentum", 0.0))
        else:
            opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        loss_fn = torch.nn.CrossEntropyLoss()
        local_epochs = self.cfg["federated"]["local_epochs"]

        for _ in range(local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                logits = self.model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()

        # 3) compute delta = local - global
        new_state = get_model_state(self.model)
        delta = model_delta(new_state, global_state)

        # 4) compress uplink update
        compressor = global_payload["compressor_ref"]
        comp_delta, meta, bits = compressor.compress_uplink(delta, client_id=self.cid)

        return {
            "delta": comp_delta,
            "meta": meta,
            "weight": self.n_samples
        }, bits
