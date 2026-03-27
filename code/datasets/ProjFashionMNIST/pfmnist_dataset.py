import torch
from torch.utils.data import Dataset

P_TRAIN = 60000
P_TEST  = 10000


# Load projected FashionMNIST data
def load_data(which):
	assert which in ["train", "test"], f"load(): unexpected value of the inputted 'which' variable ({which}). Allowed values: 'train', 'test'."
	x, y = torch.load(f"fashion_{which}_100.pt", weights_only=True)
	return x, y


# Generate three datasets, i.e. training, validation and test
def load_datasets(P_train, P_val, P_test, seed_tvs=0, seed_test=1, device="cpu"):
	assert (P_train>0) and (P_val>=0) and (P_train+P_val<=P_TRAIN), \
		f"generate_datasets(): unexpected values for 'P_train' ({P_train}) and 'P_val' ({P_val}) variables. Both must be non-negative (with P_train>0) and the sum should not exceed {P_TRAIN}."
	assert (P_test>=0) and (P_test<=P_TEST), \
		f"generate_datasets(): unexpected values for 'P_test' ({P_test}) variable. It must be non-negative and should not exceed {P_TEST}."
	x, y = load_data("train")
	g = torch.Generator().manual_seed(seed_tvs)
	idx = torch.randperm(P_TRAIN, generator=g)
	datasets = {"train": QuickDataset(x=x[idx[:P_train]], y=y[idx[:P_train]], device=device)}
	if P_val>0:
		datasets["val"] = QuickDataset(x=x[idx[P_train:P_train+P_val]], y=y[idx[P_train:P_train+P_val]], device=device)
	if P_test>0:
		x, y = load_data("test")
		g = torch.Generator().manual_seed(seed_test)
		idx = torch.randperm(P_TEST, generator=g)
		datasets["test"] = QuickDataset(x=x[idx[:P_test]], y=y[idx[:P_test]], device=device)
	return datasets


# Dataset class
class QuickDataset(Dataset):
	def __init__(
		self,
		x,
		y,
		device,
	):
		assert len(x)==len(y), f"QuickDataset.__init__(): mismatch between the lengths of x ({len(x)}) and y ({len(y)}). Lengths must coincide."
		self.x = x.float().to(device)
		self.y = y.to(device)

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]

	def to(self, device):
		device = device if isinstance(device, torch.device) else torch.device(device)
		if ("cuda" in device.type) and torch.cuda.is_available():
			self.x = self.x.to(device)
			self.y = self.y.to(device)
