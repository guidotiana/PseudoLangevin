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




## FEED FORWARD


def to_one_hot(sequences, vocab_size):
	"""
	sequences: lista di liste (numeri interi)
	vocab_size: numero totale dei token
	return: tensore shape (batch, seq_len, vocab_size)
	"""
	batch_size = len(sequences)
	seq_len = len(sequences[0])
	one_hot = torch.zeros((batch_size, seq_len, vocab_size))
	
	for i, seq in enumerate(sequences):
		for j, token in enumerate(seq):
			if 0 <= token < vocab_size:
				one_hot[i, j, token] = 1.0
	return one_hot




class OneHotSequenceDataset(Dataset):
	def __init__(self, input_ids, labels, vocab_size):
		self.input_onehot = to_one_hot(input_ids, vocab_size)
		self.labels = labels #torch.tensor()

	def __len__(self):
		return len(self.input_onehot)

	def __getitem__(self, idx):
		return {
			'input': self.input_onehot[idx],     # (seq_len, vocab_size)
			'label': self.labels[idx]            # (seq_len,)
		}

