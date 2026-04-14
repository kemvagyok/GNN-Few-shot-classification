EMBEDDING_REGISTRY = {}
GNN_REGISTRY = {}

def register_embedding(name):
    def decorator(cls):
        EMBEDDING_REGISTRY[name] = cls
        return cls
    return decorator

def register_gnn(name):
    def decorator(cls):
        GNN_REGISTRY[name] = cls
        return cls
    return decorator