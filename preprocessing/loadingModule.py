import os
import torch
from .preprocessing import get_dataset_class


# =========================================================
# GENERIC LOADER
# =========================================================
def load_dataset_cached(
    dataset_name,
    data_pth,
    files_size=4000,
    img_size=28,
    force_reload=False,
    **kwargs
):
    cache_path = os.path.join(
        data_pth,
        f"preprocessed/{dataset_name}_data_{img_size}.pt"
    )

    if os.path.exists(cache_path) and not force_reload:
        print(f"Loading cached {dataset_name}...")
        data = torch.load(cache_path)
        return (
            data["x"], 
            data["y"],
            data["n_classes"], 
            data["n_channels"]
        )

    print(f"Processing {dataset_name} from source...")

    dataset_cls = get_dataset_class(dataset_name)
    

    dataset = dataset_cls(
        path_raw=os.path.join(data_pth, "raw", dataset_name),
        img_size=img_size,
        **kwargs
    )

    data = dataset.load()

    print(f"Ended processing {dataset_name} from source...")

    torch.save({
        "x": data[0],
        "y": data[1],
        "n_classes": data[2],
        "n_channels": data[3],
    }, cache_path)

    return data