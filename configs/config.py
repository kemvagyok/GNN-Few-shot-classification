import yaml
import os
class Config:
    def __init__(self, config_path="config.yaml"):
        # YAML fájl betöltése
        with open(config_path, "r", encoding="utf-8") as file:
            config_data = yaml.safe_load(file)

        # Végigmegyünk a fő kategóriákon (pl. 'paths', 'dataset') és a bennük lévő értékeken
        for section, parameters in config_data.items():
            if isinstance(parameters, dict):
                for key, value in parameters.items():
                    # A kulcsból változónevet csinálunk (pl. self.lr_cnn = 0.01)
                    setattr(self, key, value)

# --- Tesztelés ---
if __name__ == "__main__":
    # Feltételezve, hogy a config.yaml ugyanabban a mappában van
    cfg = Config("configs/mnist_without_dpp.yaml")
    
    print(f"Dataset path: {cfg.dataset_path}")
    print(f"Tanítási ráták: CNN={cfg.lr_cnn}, GCN={cfg.lr_gcn}")
    print(f"K_hop_list: {cfg.K_hop_list}")
    print(f"Minibatch használata: {cfg.use_minibatch}")