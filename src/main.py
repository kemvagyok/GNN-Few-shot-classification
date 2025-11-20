import torch
import torch.nn.functional as F
# Metódusok kidolgozása
import numpy as np
from CNNAutocoder import CNNAutoCoder
from GCN_model import GCN
from preprocessing import MNIST_data_loading
from graph_tools import graph_creating



def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	max_image_labeled_class = [1,2,5,10,20,50,100] # How many labeled images are in each classes.
	K = 3 #How many edges does have each nodes? (KNN)
	method_similar = 'p_norm'
	p = 2
	epochs_max = 1000
	#epochs = np.arange(epochs_max+1, step=10)[1:]
	#patience = 150
	#accuses = []
	
	train_data, test_data = MNIST_data_loading()
	
	test_images = test_data.data.unsqueeze(1).float() / 255.0
	test_images = test_images.to(device)
	test_targets = test_data.targets
	test_targets = test_targets.to(device)

	for max_label in max_image_labeled_class:
		print(max_label)
		#Tanító halmaz maszkolása
		indexs = np.arange(len(train_data))
		train_targets_index_bool = [train_data.targets==target for target in range(10)]# Choosing X images from each classes as labeled
		train_targets_indexs = [np.asarray(indexs[index_bool]) for index_bool in train_targets_index_bool]
		train_mask_index = np.hstack(np.array([train_targets_indexs[target][:max_label] for target in range(10)])) #Labeling

		train_images = train_data.data[train_mask_index].unsqueeze(1).float() / 255.0
		train_images = train_images.to(device)
		train_targets = train_data.targets[train_mask_index]
		train_targets = train_targets.to(device)


		modelAutocoder = CNNAutoCoder(1).to(device) # 64 jellemzővel tér vissza
		modelGNN = GCN(64, 10).to(device)

		optimizerGNN = torch.optim.Adam(modelGNN.parameters(), lr=0.01)
		optimizerAutoEncoder = torch.optim.Adam(modelAutocoder.parameters(), lr=0.01)
		for epoch in range(epochs_max+1):
			modelAutocoder.train()
			modelGNN.train()

			optimizerGNN.zero_grad()
			optimizerAutoEncoder.zero_grad()
			train_test_images =  torch.cat((train_images, test_images), dim=0).to(device)
			train_test_latens_vectors = modelAutocoder(train_test_images).to(device)
			train_mask_index = torch.arange(len(train_images)).to(device)
			#train_latens_vectors = modelAutocoder(train_images).to(device)
			#test_latens_vectors = modelAutocoder(test_images).to(device)
			#tr_te_latens_vectors = torch.cat((train_latens_vectors, test_latens_vectors), dim=0).to(device)
			full_graph = graph_creating(train_test_latens_vectors, method_similar, p=p).to(device)
			out = modelGNN(full_graph).to(device)

			loss = F.cross_entropy(out[train_mask_index], train_targets)
			loss.backward()

			optimizerGNN.step()
			optimizerAutoEncoder.step()

			#print(f"Epoch {epoch+1}")
			#print(f"Training loss: {loss.item():.4f}")
		modelAutocoder.eval()
		modelGNN.eval()
		test_latens_vectors = modelAutocoder(test_images)
		test_graph = graph_creating(test_latens_vectors, method_similar, p = p)
		out = modelGNN(test_graph)
		pred = out.argmax(dim=1)
		correct = pred.eq(test_targets).sum().item()
		acc = correct / len(test_targets)
		print(f"Accuracy: {acc:.4f}")
	
if "__main__"==__name__:
	main()