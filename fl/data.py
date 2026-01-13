import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from fl.partition import dirichlet_partition

def build_datasets(cfg):
    name = cfg["task"]["dataset"].upper()

    if name == "CIFAR10":
        tfm = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_ds = datasets.CIFAR10(root="/root/autodl-pub/cifar-10", train=True, download=False, transform=tfm)
        test_ds = datasets.CIFAR10(root="/root/autodl-pub/cifar-10", train=False, download=False, transform=tfm)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)
        return train_ds, test_loader

    raise ValueError(f"Unknown dataset: {name}")

def build_dataloaders_for_clients(train_ds, cfg):
    num_clients = cfg["federated"]["num_clients"]
    batch_size = cfg["federated"]["batch_size"]

    if cfg["noniid"]["enabled"]:
        labels = train_ds.targets
        alpha = cfg["noniid"]["alpha"]
        min_size = cfg["noniid"]["min_size"]
        parts = dirichlet_partition(labels, num_clients, alpha=alpha, min_size=min_size, seed=cfg.get("seed", 42))
    else:
        # IID split
        n = len(train_ds)
        idx = torch.randperm(n).tolist()
        parts = [idx[i::num_clients] for i in range(num_clients)]

    loaders = []
    for k in range(num_clients):
        subset = Subset(train_ds, parts[k])
        loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2))
    return loaders
