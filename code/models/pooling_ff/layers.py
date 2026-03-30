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
		self.encoding[:, 0::2] = torch.sin(position * div_term)#.to(float())
		self.encoding[:, 1::2] = torch.cos(position * div_term)#.to(float())
		self.encoding = self.encoding.unsqueeze(0)
		self.encoding = self.encoding.float()

	def forward(self, x):
		seq_len = x.size(1)
		return self.encoding[:, :seq_len, :]



class FeedForward(nn.Module):
	def __init__(self, d, m, n, device):
		super(FeedForward, self).__init__()
		self.W1 = nn.Linear(2*d, m)         
		self.W2 = nn.Linear(m, m)
		self.PE_FF = PositionalEncoding(m, n, device)
	
	def forward(self, x):
        # x: (B, n, 2d)	
		h = self.W1(x)                      
		return F.relu(self.W2(F.relu(h) + self.PE_FF(h)))

class FF(nn.Module):
	def __init__(self, vocab_size, d, m, n, device, dropout=0.1):
		super(FF, self).__init__()
		self.token_embedding = Embedding(vocab_size, d)
		self.positional_encoding = PositionalEncoding(d, n, device)
		self.ff = FeedForward(d, m, n, device)
		self.norm1 = nn.LayerNorm(d)
		self.norm2 = nn.LayerNorm(m)
		self.device = device
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)
		self.linear = nn.Linear(d, m)

	def forward(self, x, mask=None):
		
		x = x.to(self.device)

        # embedding + PE
		x = self.token_embedding(x) + self.positional_encoding(x)  # (B, n, d)
		x = self.dropout1(x)
		x = self.norm1(x)                                          # (B, n, d)

        # -------- GLOBAL CONTEXT --------
		if mask is None:
			visible = torch.ones(x.size(0), x.size(1), device=x.device, dtype=torch.bool)
		else:
			visible = (~mask).to(x.device) 

		visible_f = visible.float().unsqueeze(-1)                  # (B, n, 1)

		sum_x = (x * visible_f).sum(dim=1)                         # (B, d)
		denom = visible_f.sum(dim=1).clamp(min=1.0)                # (B, 1)
		g = sum_x / denom                                          # (B, d)

		g_rep = g.unsqueeze(1).expand(-1, x.size(1), -1)           # (B, n, d)
		x_in = torch.cat([x, g_rep], dim=-1)                       # (B, n, 2d)
        # --------------------------------

		x_ff = self.dropout2(self.ff(x_in))                        # (B, n, m)
		x = self.norm2(x_ff + self.linear(x))                      # (B, n, m)
		return x



class Model(nn.Module):
	def __init__(self, ff, m, vocab_size, device):
		super().__init__()
		self.ff = ff
		self.linear = nn.Linear(m, vocab_size)
		self.device = device
	
	def forward(self, x, mask=None):
		x = self.ff(x, mask)
		return self.linear(x)

