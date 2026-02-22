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
#---
import numpy as np
import pandas as pd
#---
import config
import random


def main():
	epochs = np.arange(config.epochs_max+1, step=10)[1:] # Adott epoch-ban megnézi, hogy mennyire javult a modellt pontosság metrikával
		
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	train_x, train_y, test_x, test_y, num_class, channel_size = dataLoading_MNIST()

	test_x_filtered = test_x[:config.test_size].to(device)
	test_y_filtered = test_y[:config.test_size].to(device)

	best_results = []

	for max_label in config.train_images_per_class:
		#Tanító halmaz maszkolása
		train_x_filtered, train_y_filtered = traindatasetMasking(train_x, train_y, num_class, max_label)
		
		train_test_x = torch.cat((train_x_filtered.to(device), test_x_filtered.to(device)))
		train_test_y = torch.cat((train_y_filtered.to(device), test_y_filtered.to(device)))

		cnn = CNNModel(channel_size = channel_size).to(device)
		gcn = GCNModel(num_features = config.latens_size, num_classes = num_class).to(device)
		opt_cnn = torch.optim.Adam(cnn.parameters(), lr=config.lr)
		opt_gcn = torch.optim.Adam(gcn.parameters(), lr=config.lr)

		latens = cnn(train_test_x)

		databaseVector = faiss.IndexFlatL2(latens.shape[1])
		latens_cpu = latens.detach().cpu().numpy()
		databaseVector.add(latens_cpu)
		top_neighbours = databaseVector.search(latens_cpu, config.K_neigh+1)[1]
		top_neighbours = top_neighbours[:,1:]


		top_neighbours = torch.asarray(top_neighbours).to(device)
		edge_index = create_edge_index(top_neighbours).to(device)
		data = Data(x = latens, edge_index = edge_index).to(device)
		data.num_classes = num_class
		data.y = train_test_y

		best_acc = -1
		best_epoch = -1
		
		print(f"\nTrain size: {len(train_x_filtered)}, Test size: {len(test_y_filtered)}")

		for epoch in tqdm(range(config.epochs_max+1), ascii=True, desc=f"max_label: {max_label}"):
			cnn.train()
			gcn.train()

			opt_cnn.zero_grad()
			opt_gcn.zero_grad()

			root_node_idx = random.choice(range(len(train_x_filtered)))

			loader = NeighborLoader(
				data,
				num_neighbors = [config.K_neigh] * config.K_hop,
				input_nodes = torch.tensor([root_node_idx], device=device),
				shuffle = False,
			)
    		
			subgraph = next(iter(loader))
			subimages = train_test_x[subgraph.n_id]
			sublatens = cnn(subimages)
			subgraph.x = sublatens

			preds = gcn(subgraph) # Belső, 4.lépés
			loss = F.cross_entropy(preds, subgraph.y) # Only calculate loss on labeled data

			loss.backward()
			opt_cnn.step()
			opt_gcn.step()

			if epoch % config.graph_refresh == 0:
				with torch.no_grad():
					latens = cnn(train_test_x.to(device))
					databaseVector.reset()
					databaseVector.add(latens.cpu().numpy())
					D, I = databaseVector.search(latens.cpu().numpy(), config.K_neigh+1)
					neighbors = torch.tensor(I[:,1:], device=device)
					edge_index = create_edge_index(neighbors).to(device)
					data.edge_index = edge_index
			if epoch in epochs:
				with torch.no_grad():
					cnn.eval()
					gcn.eval()

					# 1. CNN embedding
					latens_test = cnn(test_x_filtered.to(device))
					index = faiss.IndexFlatL2(latens_test.shape[1])
					index.add(latens_test.cpu().numpy())

					D, I = index.search(latens_test.cpu().numpy(), config.K_neigh+1)

					neighbors_test = torch.tensor(I[:,1:], device=device)
					edge_index_test = create_edge_index(neighbors_test).to(device)


					data_test = Data(
						x=latens_test,
						edge_index=edge_index_test
					).to(device)
					# 3. GNN forward
					out = gcn(data_test)

					# 4. Accuracy
					pred = out.argmax(dim=1)
					acc = (pred == test_y_filtered.to(device)).float().mean().item()
					if acc > best_acc:
						best_acc = acc
						best_epoch = epoch

	best_results.append((max_label, best_acc, best_epoch))
	results_df = pd.DataFrame(best_results, columns=["max_label", "best_acc", "best_epoch"])
	file_name = f"{config.CHOSE_DATASET}_results_{config.K_neigh}k_{config.K_hop}h_{config.epochs_max}e.csv"
	results_df.to_csv(file_name, index=False)

if "__main__"==__name__:
	print("\nTraining/Testing starting.")
	main()
	print("\nTraining/Testing completed.")
