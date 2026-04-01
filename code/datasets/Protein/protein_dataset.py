import numpy as np
import torch
from torch.utils.data import Dataset
import ast


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
		self.original_sequences = torch.tensor(original_sequences)
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
