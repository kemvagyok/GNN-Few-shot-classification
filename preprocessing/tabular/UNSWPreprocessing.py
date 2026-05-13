import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

class UNSWPreprocessor:
    def __init__(self):
        self.cat_cols = ['proto', 'service', 'state']
        self.num_cols = None

        self.scaler = StandardScaler()
        self.label_encoders = {col: LabelEncoder() for col in self.cat_cols}

        self.fitted = False

    def fit(self, df):
        df = df.copy()

        # target eltávolítás
        df = df.drop(columns=['label', 'attack_cat'], errors='ignore')

        # numerikus oszlopok meghatározása
        self.num_cols = [c for c in df.columns if c not in self.cat_cols]

        # --- kategóriák ---
        for col in self.cat_cols:
            df[col] = df[col].astype(str)
            self.label_encoders[col].fit(df[col])

        # --- numerikus ---
        self.scaler.fit(df[self.num_cols])

        self.fitted = True

    def transform(self, df):
        assert self.fitted, "Call fit first"

        df = df.copy()

        y = None
        if 'label' in df.columns:
            y = df['label'].values

        # --- kategóriák ---
        cat_features = []
        for col in self.cat_cols:
            vals = df[col].astype(str)

            # unknown kezelése
            le = self.label_encoders[col]
            vals = vals.map(lambda x: x if x in le.classes_ else le.classes_[0])

            encoded = le.transform(vals)
            cat_features.append(encoded.reshape(-1, 1))

        cat_features = np.concatenate(cat_features, axis=1)

        # --- numerikus ---
        num_features = df[self.num_cols].values
        num_features = self.scaler.transform(num_features)

        # --- concat ---
        X = np.concatenate([num_features, cat_features], axis=1)

        return X.astype(np.float32), y
        

        