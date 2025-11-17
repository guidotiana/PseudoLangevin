# data/make_projection.py

import os, torch, numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.utils.data import TensorDataset

torch.set_default_dtype(torch.float64)

OUT_DIM  = 100
SAVE_DIR = os.path.expanduser("~/data")

PROJ_F   = f"{SAVE_DIR}/R_{OUT_DIM}.pt"
TRAIN_F  = f"{SAVE_DIR}/fashion_train_{OUT_DIM}.pt"
TEST_F   = f"{SAVE_DIR}/fashion_test_{OUT_DIM}.pt"

os.makedirs(SAVE_DIR, exist_ok=True)

def make_projection(out_dim=100, seed=42):
    rng = np.random.default_rng(seed)
    R = rng.choice([-1,1], size=(28*28, out_dim)).astype(np.float64)
    return torch.tensor(R)

# ---------- Matrice di proiezione ----------
if os.path.exists(PROJ_F):
    R = torch.load(PROJ_F)
    print("✔️ proiezione caricata")
else:
    R = make_projection(out_dim=OUT_DIM)
    torch.save(R, PROJ_F)
    print("✅ creata e salvata la matrice di proiezione")

# ---------- Dati ----------
if os.path.exists(TRAIN_F) and os.path.exists(TEST_F):
    print("✔️ dataset già proiettato, salto la generazione")
else:
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1).double())
    ])
    train_ds = datasets.FashionMNIST(SAVE_DIR, train=True,  download=True, transform=tfm)
    test_ds  = datasets.FashionMNIST(SAVE_DIR, train=False, download=True, transform=tfm)

    def proj(ds):
        X, Y = [], []
        for x,y in tqdm(ds, total=len(ds)):
            v = x @ R
            v = torch.sign(v)
            X.append(v)
            Y.append(torch.tensor(y))
        return torch.stack(X), torch.stack(Y)

    print("Proiezione train …")
    Xtr, Ytr = proj(train_ds)
    torch.save((Xtr, Ytr), TRAIN_F)
    print("✔️ salvato train")

    print("Proiezione test …")
    Xte, Yte = proj(test_ds)
    torch.save((Xte, Yte), TEST_F)
    print("✔️ salvato test")

