import numpy as np

def dirichlet_partition(labels, num_clients, alpha=0.5, min_size=10, seed=42):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    num_classes = labels.max() + 1

    idx_by_class = [np.where(labels == c)[0] for c in range(num_classes)]
    for c in range(num_classes):
        rng.shuffle(idx_by_class[c])

    client_indices = [[] for _ in range(num_clients)]

    while True:
        client_indices = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            idxs = idx_by_class[c]
            if len(idxs) == 0:
                continue

            proportions = rng.dirichlet([alpha] * num_clients)
            # split points
            splits = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
            parts = np.split(idxs, splits)
            for k in range(num_clients):
                client_indices[k].extend(parts[k].tolist())

        sizes = [len(ci) for ci in client_indices]
        if min(sizes) >= min_size:
            break

    for k in range(num_clients):
        rng.shuffle(client_indices[k])
    return client_indices
