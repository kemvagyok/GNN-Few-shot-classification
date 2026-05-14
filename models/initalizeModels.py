from .registry import EMBEDDING_REGISTRY, GNN_REGISTRY
from torch.nn.parallel import DistributedDataParallel as DDP

def initalizeModels(config, channel_size, latens_size, num_class, device, is_ddp, input_size = -1):

    # -------- EMBEDDING --------
    if config.embedding not in EMBEDDING_REGISTRY:
        raise ValueError(f"Unknown embedding: {config.embedding}")

    embedder_cls = EMBEDDING_REGISTRY[config.embedding]

    if config.embedding in ["bert", "qwen"]:
        embedder = embedder_cls(
            isFreeze = config.isFreeze,
            isClassificator = latens_size == num_class
        ).to(device)
    else:
        if input_size == -1: #Azaz nem tabulátoros
            embedder = embedder_cls(output_dim=latens_size, 
                                    channel_size=channel_size,
                                    isFreeze = config.isFreeze,
                                    isClassificator = latens_size == num_class
                                    ).to(device)
        else:
            embedder = embedder_cls(input_dim=input_size,
                        output_dim=latens_size, 
                        channel_size=channel_size,
                        isFreeze = config.isFreeze,
                        isClassificator = latens_size == num_class
                        ).to(device)   

    # -------- GNN --------
    if config.gnn_model not in GNN_REGISTRY:
        raise ValueError(f"Unknown gnn_model: {config.gnn_model}")

    gnn_cls = GNN_REGISTRY[config.gnn_model]
    gnn = gnn_cls(num_features=latens_size, 
                  num_classes=num_class
                  ).to(device)

    return embedder, gnn