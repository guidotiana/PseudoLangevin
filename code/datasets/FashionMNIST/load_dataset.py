# data/load_dataset.py

import torch
from torch.utils.data import TensorDataset, Dataset
import os

torch.set_default_dtype(torch.float64)

# Percorsi dei file nella cartella attuale
SAVE_DIR = os.path.expanduser("~/workspace/PseudoLangevin/code/datasets/FashionMNIST")
OUT_DIM = 100
TRAIN_F = f"{SAVE_DIR}/fashion_train_{OUT_DIM}.pt"
TEST_F  = f"{SAVE_DIR}/fashion_test_{OUT_DIM}.pt"
PROJ_F  = f"{SAVE_DIR}/R_{OUT_DIM}.pt"

# ======== 1. Carica TRAIN e TEST proiettati =========
X_train, Y_train = torch.load(TRAIN_F, weights_only=True)
X_test,  Y_test  = torch.load(TEST_F,  weights_only=True)

# Converti in float64 per coerenza con la rete
X_train = X_train.double()
X_test  = X_test.double()

print("✔️ Dataset proiettati:")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ======== 2. Funzione per sottoinsieme del training set =========
def make_fashion_subset(P, seed=42):
    """Crea un sottoinsieme casuale del training set"""
    g   = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(X_train), generator=g)[:P]
    return TensorDataset(X_train[idx], Y_train[idx])

# ======== 3. Classe per gestione sottoinseme del training set ===
class FashionDataset(Dataset):

	def __init__(self,
		P:int,
		seed:int = 0,
		device:str|torch.device|None = None
	):
		super().__init__()
		self.P = P
		self.seed = seed
		self.device = device

		g = torch.Generator().manual_seed(seed)
		idx = torch.randperm(len(X_train), generator=g)[:P]
		self.x = X_train[idx].to(device)
		self.y = Y_train[idx].to(device)

		if len(self.x) < self.P:
			self.P = len(self.x)

	def __getitem__(self, idx):
		return self.x[idx], self.y[idx], idx

	def __len__(self):
		return self.P
