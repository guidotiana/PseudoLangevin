import numpy as np
import torch
from torch.utils.data import Dataset
import ast

P_MAX = 15000
VOCAB_SIZE = 21
MASK_ID = 20


def load_seq(filename="SequencesMasked.txt", n=None):
	masked_sequences_ohe = []
	sequences_ohe = []
	masks = []

	with open(filename, "r") as file:
		next(file)  
		for i, line in enumerate(file):
			if n is not None and i >= n:
				break
			smn, sn, mp = line.strip().split("\t")
			masked_sequences_ohe.append(ast.literal_eval(smn))  
			sequences_ohe.append(ast.literal_eval(sn))  
			masks.append(np.array(ast.literal_eval(mp)))  

	return masked_sequences_ohe, sequences_ohe, masks


def load_datasets(P_train, P_val=0, P_test=0, seed=0, device="cpu", filename="SequencesMasked.txt", n=None):
	assert P_train>0, f"load_datasets(): invalid value for 'P_train' ({P_train}). It must be positive, P_train>0."
    assert P_val>0, f"load_datasets(): invalid value for 'P_val' ({P_val}). It must be non-negative, P_val>=0."
	assert P_test>0, f"load_datasets(): invalid value for 'P_test' ({P_test}). It must be non-negative, P_test>=0."
	assert P_train+P_val+P_test<=P_max, f"load_datasets(): invalid values for 'P_train' ({P_train}), 'P_val' ({P_val}), and P_test ({P_test}). The sum must not exceed {P_max}."

	masked_sequences_ohe, sequences_ohe, masks = load_seq(filename, n)
	masked_sequences_ohe = torch.tensor(masked_sequences_ohe)
	sequences_ohe = torch.tensor(sequences_ohe)

	torch.manual_seed(seed)
	perm_idxs = torch.randperm(len(masked_sequences_ohe))
	
	P_current = 0
	datasets = {}
	for key, P_key in zip(["train", "val", "test"], [P_train, P_val, P_test]):
		if P_key==0: continue
		datasets[key] = MaskedDataset(
			x=masked_sequences_ohe[P_current:P_current+P_key],
			y=masks[P_current:P_current+P_key],
			original_sequences=sequences_ohe[P_current:P_current+P_key],
			device=device,
		)
		P_current += P_key
	
	return datasets

	

class MaskedDataset(Dataset):

	def __init__(self,
			x,                   #Long torch.tensor of one-hot-encoded masked sequences
			y,                   #List[List[masked site index, correct amino acid token]]
			original_sequences,  #Long torch.tensor of one-hot-encoded unmasked sequences (ground-truth)
			device='cpu',
	):
		super(MaskedDataset, self).__init__()

		check_device = any([device == 'cpu', 'cuda:' in device])
		assert check_device, f'MaskedDataset.__init__(): invalid value for device, {device}. Allowed values: "cpu" (default), "cuda:<int>".'

		self.x = x
		self.y = y
		self.original_sequences = original_sequences
		self.P = len(self.x)

		if ('cuda' in device) and torch.cuda.is_available():
			self.x = self.x.to(device)
			#self.y = self.y.to(device)
			self.device = device
		else:
			self.device = 'cpu'


	def __len__(self):
		return self.P

	def __getitem__(self, idx):
		return self.x[idx], [array for array, flag in zip(self.y, idx) if flag], idx

	def to(self, device):
		device = device if isinstance(device, torch.device) else torch.device(device)
		if ("cuda" in device.type) and torch.cuda.is_available():
			self.x = self.x.to(device)
			self.y = [torch.tensor(a, device=device) for a in self.y]
