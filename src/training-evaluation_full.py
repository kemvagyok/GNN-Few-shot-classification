from random import random
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import faiss
from tqdm import tqdm
import numpy as np
import pandas as pd

# Saját modulok
from graph_tools import create_edge_index, graph_creating
from models import CNNModel, GCNModel
from preprocessing import dataLoading_MNIST, traindatasetMasking
import config_summary
from config import Config

def main():
    config = Config()
    
    # Adott epoch-ban megnézi, hogy mennyire javult a modell pontossága
    epochs = np.arange(config.epochs_max+1, step=10)[1:] 
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_x, train_y, test_x, test_y, num_class, channel_size = dataLoading_MNIST()

    # Tegyük fel a teszt adatokat a GPU-ra egyszer
    test_x_filtered = test_x[:config.test_size].to(device)
    test_y_filtered = test_y[:config.test_size].to(device)
    best_results = []
    for max_label in config.train_images_per_class:
        # Tanító halmaz maszkolása 
        train_x_filtered, train_y_filtered = traindatasetMasking(train_x, train_y, num_class, max_label)
        
        # GPU-ra mozgatás és összefűzés a ciklus előtt
        train_test_x = torch.cat((train_x_filtered.to(device), test_x_filtered)).to(device)
        train_test_y = torch.cat((train_y_filtered.to(device), test_y_filtered)).to(device)

        # Maszkok létrehozása és GPU-n tartása
        train_mask = torch.zeros(len(train_test_y), dtype=torch.bool, device=device)
        train_mask[:len(train_x_filtered)] = True 
        test_mask = ~train_mask

        cnn = CNNModel(channel_size=channel_size).to(device)
        gcn = GCNModel(num_features=config.latens_size, num_classes=num_class).to(device)
        opt_cnn = torch.optim.Adam(cnn.parameters(), lr=config.lr_cnn)
        opt_gcn = torch.optim.Adam(gcn.parameters(), lr=config.lr_gcn)

        best_acc = -1
        best_epoch = -1
        
        print(f"\nTrain size: {len(train_x_filtered)}, Test size: {len(test_y_filtered)}")

        # Gráf élek inicializálása
        edge_index = None

        for epoch in tqdm(range(config.epochs_max+1), ascii=True, desc=f"Max labeled: {max_label}"):
            cnn.train()
            gcn.train()
            opt_cnn.zero_grad()
            opt_gcn.zero_grad()

            # Nincs detach! Így a CNN is kap gradienst a GCN-ből visszaterjedve.
            latens = cnn(train_test_x)
            
            # Csak akkor frissítjük a gráfot, ha kell (vagy a legelső epochban)
            if epoch == 0 or epoch % config.graph_refresh == 0:
                # A Faiss CPU-n fut, ezért ide kell a detach() és a numpy átalakítás
                latens_cpu = latens.detach().cpu().numpy()
                databaseVector = faiss.IndexFlatL2(latens_cpu.shape[1])
                databaseVector.add(latens_cpu)
                
                # top_neighbours[:, 1:] levágja az önmagába mutató éleket (self-loops)
                top_neighbours = databaseVector.search(latens_cpu, config.K_neigh+1)[1][:, 1:]
                
                # Közvetlenül GPU-ra tesszük a tenzort
                top_neighbours_tensor = torch.tensor(top_neighbours, device=device)
                edge_index = create_edge_index(top_neighbours_tensor).to(device)

            # Data objektum létrehozása az adott epochhoz
            data = Data(x=latens, edge_index=edge_index)
            
            # GCN Forward
            preds = gcn(data)
            
            # Loss számítás csak a tanító adatokon
            loss = F.cross_entropy(preds[train_mask], train_test_y[train_mask])

            loss.backward()
            opt_cnn.step()
            opt_gcn.step()
            
            # Kiértékelés
            if epoch in epochs:
                cnn.eval()
                gcn.eval()
                with torch.no_grad():
                    # Transzduktív kiértékelés: Nem építünk új gráfot, 
                    # használjuk a meglévő hálózatot a teszt elemek előrejelzésére.
                    latens_val = cnn(train_test_x)
                    data_val = Data(x=latens_val, edge_index=edge_index)
                    
                    out = gcn(data_val) 
                    
                    # Predikció és pontosság számítás csak a teszt adatokon
                    pred_classes = out[test_mask].argmax(dim=1)
                    acc = (pred_classes == train_test_y[test_mask]).float().mean().item()
                    
                    if acc > best_acc:
                        best_acc = acc
                        best_epoch = epoch

        # INDENTATION FIXED HERE: Now runs once per max_label loop
        best_results.append((max_label, best_acc, best_epoch))

    # Save logic remains outside the loops
    results_df = pd.DataFrame(best_results, columns=["max_label", "best_acc", "best_epoch"])
    file_name = f"../results/{config.CHOSE_DATASET}_results_{config.K_neigh}h_{config.epochs_max}e.csv"
    results_df.to_csv(file_name, index=False)

if __name__ == "__main__":
    config = Config()
    config_summary.print_config(config)
    print("\nTraining/Testing starting.")
    main()
    print("\nTraining/Testing completed.")