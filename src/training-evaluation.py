from random import random
from networkx import subgraph
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import faiss
from tqdm import tqdm
# Metódusok kidolgozása
from graph_tools import create_edge_index
from models import CNNModel, GCNModel
from preprocessing import dataLoading_MNIST, traindatasetMasking
import config_summary
#---
import numpy as np
import pandas as pd
#---
from config import Config
import random


def main():

	config = Config()

	epochs = np.arange(config.epochs_max+1, step=10)[1:] # Adott epoch-ban megnézi, hogy mennyire javult a modellt pontosság metrikával
		
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	train_x, train_y, test_x, test_y, num_class, channel_size = dataLoading_MNIST()

	test_x_filtered = test_x[:config.test_size].to(device)
	test_y_filtered = test_y[:config.test_size].to(device)


	for K_hop in config.K_hop_list:
		best_results = []
		print("" + "="*50)
		print(f"Training with K_hop: {K_hop} and K_neigh: {config.K_neigh}")
		print("" + "="*50)

		for max_label in config.train_images_per_class:
			#Tanító halmaz maszkolása 
			train_x_filtered, train_y_filtered = traindatasetMasking(train_x, train_y, num_class, max_label)
			
			train_test_x = torch.cat((train_x_filtered.to(device), test_x_filtered.to(device)))
			train_test_y = torch.cat((train_y_filtered.to(device), test_y_filtered.to(device)))


			train_mask = torch.zeros(len(train_test_y), dtype=torch.bool)
			#TRAIN+TEST SIZE MASK, TRUE for train, FALSE for test
			train_mask[:len(train_x_filtered)] = True 
			test_mask = ~train_mask

			cnn = CNNModel(channel_size = channel_size).to(device)
			gcn = GCNModel(num_features = config.latens_size, num_classes = num_class).to(device)
			opt_cnn = torch.optim.Adam(cnn.parameters(), lr=config.lr_cnn)
			opt_gcn = torch.optim.Adam(gcn.parameters(), lr=config.lr_gcn)

			latens = cnn(train_test_x).detach()
			latens_cpu = latens.detach().cpu().numpy() # FAISS CPU-ra várja a bemenetet
			
			databaseVector = faiss.IndexFlatL2(latens_cpu.shape[1]) # Hány dimenzió a latens tér
			databaseVector.add(latens_cpu)

			top_neighbours = databaseVector.search(latens_cpu, config.K_neigh+1)[1][:,1:] # K_neigh+1, mert a legközelebbi szomszéd maga a pont, amit keresünk, azt kihagyjuk

			top_neighbours = torch.tensor(top_neighbours, device = device)
			edge_index = create_edge_index(top_neighbours).to(device)
			
			data = Data(x = latens, edge_index = edge_index).to(device)
			data.num_classes = num_class
			data.y = train_test_y
			data.train_mask = train_mask.to(device)
			data.test_mask = ~train_mask.to(device)


			#root_node_idx = random.choice(range(len(train_x_filtered)))#!!!!!!

			loader = NeighborLoader(
				data,
				num_neighbors = [config.K_neigh] * K_hop,
				#input_nodes = torch.tensor([root_node_idx], device=device),
				input_nodes = train_mask,
				shuffle = True
			)

			#subgraph = next(iter(loader))

			best_acc = -1
			best_epoch = -1
			
			print(f"\nTrain size: {len(train_x_filtered)}, Test size: {len(test_y_filtered)}")

			for epoch in tqdm(range(config.epochs_max+1), ascii=True, desc=f"Max labeled train image size: {max_label}", disable=False):
				if epoch % config.graph_refresh == 0:
					with torch.no_grad():
						cnn.eval()

						latens = cnn(train_test_x.to(device))
						latens_cpu = latens.detach().cpu().numpy()

						databaseVector.reset()
						databaseVector.add(latens_cpu)
						D, I = databaseVector.search(latens_cpu, config.K_neigh+1)
						neighbors = torch.tensor(I[:,1:], device=device)
						
						edge_index = create_edge_index(neighbors).to(device)
						data.edge_index = edge_index
						data.x = latens
						loader = NeighborLoader(
							data,
							num_neighbors = [config.K_neigh] * K_hop,
							input_nodes = train_mask,
							shuffle = True
						)

				
				cnn.train()
				gcn.train()

				for subgraph in loader:
					opt_cnn.zero_grad() 
					opt_gcn.zero_grad()

					subimages = train_test_x[subgraph.n_id]
					sublatens = cnn(subimages)
					subgraph.x = sublatens
					preds = gcn(subgraph) # Belső, 4.lépés
					train_mask_sub = subgraph.train_mask
					#print("Train nodes in batch:", train_mask_sub.sum().item())
					#loss = F.cross_entropy(preds[data.train_mask[subgraph.n_id]], subgraph.y[data.train_mask[subgraph.n_id]]) # Only calculate loss on labeled data
					loss = F.cross_entropy(preds[train_mask_sub], subgraph.y[train_mask_sub]) # Only calculate loss on labeled data

					loss.backward()
					opt_cnn.step()
					opt_gcn.step()



				if epoch in epochs:
					with torch.no_grad():
						cnn.eval()
						gcn.eval()

						# 1. CNN embedding
						#latens_test = cnn(train_test_x.to(device))
						latens_test = cnn(test_x_filtered.to(device))
						latens_test_cpu = latens_test.detach().cpu().numpy()
						
						index = faiss.IndexFlatL2(latens_test_cpu.shape[1])
						index.add(latens_test_cpu)

						D, I = index.search(latens_test_cpu, config.K_neigh+1)

						neighbors_test = torch.tensor(I[:,1:], device=device)
						edge_index_test = create_edge_index(neighbors_test).to(device)

						data_test = Data(
							x = latens_test,
							edge_index = edge_index_test
						).to(device)
						# 3. GNN forward
						out = gcn(data_test) # Only test nodes

						# 4. Accuracy
						#pred = out[test_mask].argmax(dim=1)
						pred = out.argmax(dim=1)
						#acc = (pred == train_test_y[test_mask].to(device)).float().mean().item()
						acc = (pred == test_y_filtered.to(device)).float().mean().item()
						if acc > best_acc:
							best_acc = acc
							best_epoch = epoch
							print(f"New best accuracy: {best_acc:.4f} at epoch {best_epoch} with max_label {max_label}")
			best_results.append((max_label, best_acc, best_epoch))
	
			
		results_df = pd.DataFrame(best_results, columns=["max_label", "best_acc", "best_epoch"])
		file_name = f"../results/{config.CHOSE_DATASET}_results_{config.K_neigh}k_{K_hop}h_{config.epochs_max}e.csv"
		results_df.to_csv(file_name, index=False)

if __name__ == "__main__":
	config = Config()
	config_summary.print_config(config)
	print("\nTraining/Testing starting.")
	main()
	print("\nTraining/Testing completed.")
