import torch.nn as nn

from .layers import Embedding, PositionalEncoding, MultiHeadAttention, FeedForward, TransformerBlock


class Transformer(nn.Module):
	def __init__(self, vocab_size, d, H, m, L, n, device, dropout=0.1):
		super(Transformer, self).__init__()
		self.token_embedding = Embedding(vocab_size, d)
		#self.register_buffer("positional_encoding", PositionalEncoding(d, n, device))
		self.positional_encoding = PositionalEncoding(d, n, device)
		self.layers = nn.ModuleList([
			TransformerBlock(d, H, m, n, device, dropout) for _ in range(L)
		])
		self.dropout = nn.Dropout(dropout)
		self.device = device

	def forward(self, x, mask=None):
		x = self.token_embedding(x) + self.positional_encoding(x)
		x = self.dropout(x)
		for layer in self.layers:
			x = layer(x, mask)
		return x



class TModel(nn.Module):
	def __init__(self, transformer, d, vocab_size, device):
		super().__init__()
		self.transformer = transformer
		self.linear = nn.Linear(d, vocab_size)
		self.device = device

	def forward(self, x):
		x = self.transformer(x)
		return self.linear(x)
