import numpy as np
import torch
from torch.utils.data import Dataset
import ast

torch.set_default_dtype(torch.float64)


# load: sequence masked in number, seqeunces non masked in number and mask (position of mask and masked number)
def load_seq(filename="SequencesMasked.txt", n=None):
	sequences_masked_number = []
	sequences_number = []
	masked_positions = []

	with open(filename, "r") as file:
		next(file)  
		for i, line in enumerate(file):
			if n is not None and i >= n:
				break
			smn, sn, mp = line.strip().split("\t")
			sequences_masked_number.append(ast.literal_eval(smn))  
			sequences_number.append(ast.literal_eval(sn))  
			masked_positions.append(np.array(ast.literal_eval(mp)))  

	return sequences_masked_number, sequences_number, masked_positions


class MaskedDataset(Dataset):

	def __init__(self, x, y, original_sequences, P, device='cpu', only_index=True):
		super(MaskedDataset, self).__init__()

		# check inputs
		check_device = any([device == 'cpu', 'cuda:' in device])
		assert check_device, f'Invalid value for device: {device}. Allowed values: "cpu" (default), "cuda:<int>".'
        
		self.x = torch.tensor(x) # x = original_sequences, dtype=torch.long
		self.y = y # y = mask
		self.original_sequences = torch.tensor(original_sequences)
		self.P = P


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
