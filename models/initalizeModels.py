from .registry import EMBEDDING_REGISTRY, GNN_REGISTRY
from torch.nn.parallel import DistributedDataParallel as DDP

def initalizeModels(config, channel_size, latens_size, num_class, device, is_ddp):

    # -------- EMBEDDING --------
    if config.embedding not in EMBEDDING_REGISTRY:
        raise ValueError(f"Unknown embedding: {config.embedding}")

    embedder_cls = EMBEDDING_REGISTRY[config.embedding]

    if config.embedding in ["bert", "qwen"]:
        embedder = embedder_cls().to(device)
    else:
        embedder = embedder_cls(output_dim=latens_size, channel_size=channel_size).to(device)

    # -------- GNN --------
    if config.gnn_model not in GNN_REGISTRY:
        raise ValueError(f"Unknown gnn_model: {config.gnn_model}")

    gnn_cls = GNN_REGISTRY[config.gnn_model]
    gnn = gnn_cls(num_features=latens_size, num_classes=num_class).to(device)

    # -------- DDP --------
    if is_ddp:
        if not config.use_minibatch:
            raise ValueError("DDP only supported with minibatch mode.")

        embedder = DDP(embedder, device_ids=[device.index])
        gnn = DDP(gnn, device_ids=[device.index])

    return embedder, gnn