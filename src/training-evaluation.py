import torch
import torch.nn.functional as F
from tqdm import tqdm
# Metódusok kidolgozása
from models import FewShotModel
from preprocessing import dataLoading_MNIST, traindatasetMasking
#---
import numpy as np
import pandas as pd
#---
import config


def main():
	epochs = np.arange(config.epochs_max+1, step=10)[1:] # Adott epoch-ban megnézi, hogy mennyire javult a modellt pontosság metrikával
		
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	train_x, train_y, test_x, test_y, class_num, channel_size = dataLoading_MNIST()
	train_x_filtered, train_y_filtered = traindatasetMasking(train_x, train_y, class_num, 1)

	train_x_filtered = train_x_filtered.to(device)
	train_y_filtered = train_y_filtered.to(device)
	test_x_filtered = test_x[:config.test_size].to(device)
	test_y_filtered = test_y[:config.test_size].to(device)

	best_results = []

	for max_label in config.train_images_per_class:
		#Tanító halmaz maszkolása
		train_x_filtered, train_y_filtered = traindatasetMasking(train_x, train_y, class_num, max_label)
		
		train_test_x = torch.cat((train_x_filtered.to(device), test_x_filtered.to(device)))
		train_test_y = torch.cat((train_y_filtered.to(device), test_y_filtered.to(device)))

		model = FewShotModel(input_size=28, output_size=class_num, latens_size = 64, channel_size = channel_size, device = device).to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
		best_acc = -1
		best_epoch = -1
		
		for epoch in tqdm(range(config.epochs_max+1), ascii=True, desc=f"max_label: {max_label}"):
			model.train()
			optimizer.zero_grad()
			output = model(train_test_x.to(device), config.method_similar, 2).to(device)
			loss = F.cross_entropy(output[:len(train_y_filtered)].to(device), train_y_filtered.to(device))
			loss.backward()
			optimizer.step()

			if epoch in epochs:
				with torch.no_grad():
					model.eval()
					out = model(test_x_filtered, config.method_similar, 2).to(device)
					pred = out.argmax(dim=1)
					correct = pred.eq(test_y_filtered.to(device)).sum().item()
					acc = correct / len(test_y_filtered)
					if acc > best_acc:
						best_acc = acc
						best_epoch = epoch

		best_results.append((max_label, best_acc, best_epoch))
	pd.DataFrame(best_results, columns=['Max Labeled Images per Class', 'Best Accuracy', 'Best Epoch']).to_csv('results.csv', index=False)
		

if "__main__"==__name__:
	main()