from metrics import macro_f1
from sklearn.metrics import classification_report

def build_metrics(config):
    metrics = {}

    if "macro_f1" in config.metrics:
        metrics = macro_f1
    elif "accuracy" in config.metrics:
         metrics = None
    elif "classification_report" in config.metrics:
        metrics = classification_report
    else:
        raise ValueError(f"Unsupported metric: {config.metrics}")
    return metrics