from tqdm import trange
import torch
import json
import os
from typing import Optional, List, Dict, Any

from fl.data import build_datasets, build_dataloaders_for_clients
from fl.models import build_model
from fl.server import Server
from fl.client import Client
from fl.compression import build_compressor
from fl.metrics import BitMeter

def run_federated(cfg: dict, yaml_path: Optional[str] = None, filename_suffix: str = ""):
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

    # Initialize training data collection
    rounds_data: List[Dict[str, Any]] = []
    training_data = {
        "config": cfg,
        "rounds": rounds_data
    }

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
        
        # Store round data
        round_data = {
            "round": r + 1,
            "accuracy": float(acc),
            "bits": stats,
            "selected_clients": len(selected)
        }
        rounds_data.append(round_data)
        
        print(f"[Round {r+1}/{rounds}] acc={acc:.4f} bits={stats}")

    final_stats = bitmeter.summary(reset=False)
    training_data["final_bits"] = final_stats
    print("Done. Final bits:", final_stats)

    # Save training data to JSON file
    if yaml_path:
        yaml_name = os.path.splitext(os.path.basename(yaml_path))[0]
        json_filename = f"{yaml_name}{filename_suffix}.json"
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        print(f"Training data saved to {json_filename}")
