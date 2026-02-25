# config.py

class Config:
    def __init__(self):

        # Dataset
        self.CHOSE_DATASET = "MNIST"
        self.train_images_per_class = [1,2,5,10,20,50,100]
        self.validation_size = 1200
        self.test_size = 1000

        # Graph
        self.K_neigh = 3
        self.K_hop_list = [1,2,3,4,5,6]
        self.graph_refresh = 10

        # Training
        self.lr_cnn = 0.01
        self.lr_gcn = 0.01
        self.epochs_max = 2500

        # Similarity
        self.method_similar = "p_norm"
        self.p_n_vectors = [1,2,3]

        # Model
        self.latens_size = 64

        # Reproducibility
        self.seed = 42