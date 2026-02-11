import torch
import torch.nn.functional as F
# Metódusok kidolgozása
import numpy as np
from FewShotModel import FewShotModel
from preprocessing import dataLoading_MNIST, traindatasetMasking
from graph_tools import graph_creating



def main():
	max_image_labeled_class = [1,2,5,10,20,50,100] # How many labeled images are in each classes.
	#validation_size = 1200
	test_size = 100
	K = 3 #How many edges does have each nodes? (KNN)
	lr = 0.01
	method_similar = 'p_norm'
	#p_n_vectors = [1,2,3]

	epochs_max = 2500
	epochs = np.arange(epochs_max+1, step=10)[1:] # Adott epoch-ban megnézi, hogy mennyire javult a modellt pontosség metrikával
		
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	train_x, train_y, test_x, test_y, class_num, channel_size = dataLoading_MNIST()
	train_x_filtered, train_y_filtered = traindatasetMasking(train_x, train_y, class_num, 1)

	test_x_filtered = test_x[:test_size]
	test_y_filtered = test_y[:test_size]

	best_results = []

	for max_label in max_image_labeled_class:
		print(max_label)
		#Tanító halmaz maszkolása
		train_x_filtered, train_y_filtered = traindatasetMasking(train_x, train_y, class_num, max_label)
		train_test_x = torch.cat((train_x_filtered, test_x_filtered))
		train_test_y = torch.cat((train_y_filtered, test_y_filtered))

		model = FewShotModel(input_size=28, output_size=class_num, latens_size = 64, channel_size = channel_size, device = device).to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=lr)
		best_acc = -1
		best_epoch = -1
		
		for epoch in range(epochs_max+1):
			model.train()
			optimizer.zero_grad()
			output = model(train_test_x.to(device), method_similar, 2)
			loss = F.cross_entropy(output[:len(train_y_filtered)], train_y_filtered.to(device))
			loss.backward()
			optimizer.step()

			if epoch in epochs:
				with torch.no_grad():
					model.eval()
					out = model(test_x_filtered, method_similar, 2)
					pred = out.argmax(dim=1)
					correct = pred.eq(test_y_filtered).sum().item()
					acc = correct / len(test_y_filtered)
					if acc > best_acc:
						best_acc = acc
						best_epoch = epoch

		best_results.append((max_label, best_acc, best_epoch))


if "__main__"==__name__:
	main()