import torch
import torch.nn as nn
import torch.nn.functional as F



class Embedding(nn.Module):
	def __init__(self, vocab_size, d):
		super(Embedding, self).__init__()
		self.embedding = nn.Embedding(vocab_size, d)

	def forward(self, x):
		return self.embedding(x)



class PositionalEncoding(nn.Module):
	def __init__(self, d, n, device):
		super(PositionalEncoding, self).__init__()
		self.encoding = torch.zeros(n, d, device=device)
		position = torch.arange(0, n, dtype=torch.float, device=device).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d, 2, dtype=torch.float, device=device) * (-torch.log(torch.tensor(10000.0, device=device)) / d))
		self.encoding[:, 0::2] = torch.sin(position * div_term)
		self.encoding[:, 1::2] = torch.cos(position * div_term)
		self.encoding = self.encoding.unsqueeze(0)

	def forward(self, x):
		seq_len = x.size(1)
		return self.encoding[:, :seq_len, :]



class MultiHeadAttention(nn.Module):
	def __init__(self, d, H):
		super(MultiHeadAttention, self).__init__()
		self.H = H
		self.d = d
		self.k = d // H

		assert self.k * H == d, "d must be divisible by H"

		# Wqk ≈ Wq @ Wk.T
		self.Wvc = nn.Linear(d, d, bias = False)
		# Wvc ≈ Wv @ Wc
		self.Wqk = nn.Linear(d, d, bias = False)

	# forward con due matrici
	def forward(self, x, mask=None):

		# N numero di frasi, n lughezza massima numero di parole, d embedding
		# x: (N, n, d)
		N, n, d = x.size()

		#  X · Wqk · X^T
		scores = torch.matmul(self.Wqk(x), x.transpose(1,2)) # (N, n, n)
		scores = scores / (self.d ** 0.5)

		if mask is not None:
			scores = scores.masked_fill(mask == 0, float('-1e20'))

		# Softmax (N, n, n)
		attention = F.softmax(scores, dim=-1)

		# (N, n, n) @ (N, n, d) --> (N, n, d)
		attended = torch.bmm(attention, x)

		return self.Wvc(attended) # (N, n, d) @ (N, d, d) --> (N, n, d)



class FeedForward(nn.Module):
	def __init__(self, d, m, n, device):
		super(FeedForward, self).__init__()
		self.W1 = nn.Linear(d, m)
		self.W2 = nn.Linear(m, d)
		self.PE_FF = PositionalEncoding(m, n, device)

	def forward(self, x):
		return self.W2(F.relu(self.W1(x)) + self.PE_FF(self.W1(x))) # aggiungo PE in modo da togliere la simmetria



class TransformerBlock(nn.Module):
	def __init__(self, d, H, m, n, device, dropout=0.1):
		super(TransformerBlock, self).__init__()
		self.attention = MultiHeadAttention(d, H)
		self.norm1 = nn.LayerNorm(d)
		self.norm2 = nn.LayerNorm(d)
		self.ff = FeedForward(d, m, n, device)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, mask):
		attn_out = self.dropout(self.attention(x, mask))
		x = self.norm1(x + attn_out)
		ff_out = self.dropout(self.ff(x))
		x = self.norm2(x + ff_out)
		return x
