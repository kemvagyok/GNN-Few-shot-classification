class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0  # Reset, ha javult a modell
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1 # Növeljük a számlálót, ha romlik
            if self.counter >= self.patience:
                return True
        return False