from tqdm import trange
import torch

from fl.data import build_datasets, build_dataloaders_for_clients
from fl.models import build_model
from fl.server import Server
from fl.client import Client
from fl.compression import build_compressor
from fl.metrics import BitMeter

def run_federated(cfg: dict):
    device = torch.device(cfg.get("device", "cpu"))

    # 1) data
    train_ds, test_loader = build_datasets(cfg)
    client_loaders = build_dataloaders_for_clients(train_ds, cfg)

    # 2) model
    global_model = build_model(cfg).to(device)

    # 3) server / clients
    compressor = build_compressor(cfg, global_model)
    server = Server(global_model, compressor, device=device)

    clients = []
    for cid, loader in enumerate(client_loaders):
        clients.append(Client(
            cid=cid,
            device=device,
            cfg=cfg,
            train_loader=loader,
        ))

    # 4) bits meter
    bitmeter = BitMeter()

    rounds = cfg["federated"]["rounds"]
    for r in trange(rounds, desc="Federated Rounds"):
        selected = server.sample_clients(clients, cfg["federated"]["clients_per_round"])

        # downlink: broadcast global weights
        payload_down, down_bits = server.broadcast()
        bitmeter.add("downlink", down_bits)

        client_updates = []
        uplink_bits_total = 0

        for c in selected:
            # client trains locally
            update_payload, uplink_bits = c.train_one_round(payload_down)
            client_updates.append(update_payload)
            uplink_bits_total += uplink_bits

        bitmeter.add("uplink", uplink_bits_total)

        # server aggregates
        server.aggregate(client_updates)

        # eval
        acc = server.evaluate(test_loader)
        stats = bitmeter.summary(reset=False)
        print(f"[Round {r+1}/{rounds}] acc={acc:.4f} bits={stats}")

    print("Done. Final bits:", bitmeter.summary(reset=False))
