from metrics import macro_f1

def build_metrics(config):
    metrics = {}

    if "macro_f1" in config.metrics:
        metrics = macro_f1
    elif "accuracy" in config.metrics:
         metrics = None
    else:
        raise ValueError(f"Unsupported metric: {config.metrics}")
    return metrics