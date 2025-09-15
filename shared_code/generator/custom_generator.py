import torch
import numpy as np


class CustomGenerator:
	def __init__(
		self,
		seed: int | None = 0,
		device: str | torch.device | None = "cpu",
	):
		self.device = device
		self.seed = seed if isinstance(seed, int) else 0
		self.generator = torch.Generator(device)
		self.generator.manual_seed(self.seed)

	def save(self, fn):
		state = self.generator.get_state().detach().cpu().numpy()
		with open(fn, 'wb') as f:
			np.save(f, state)

	def load(self, fn):
		with open(fn, 'rb') as f:
			state = np.load(f, allow_pickle=True)
		state = torch.tensor(state, dtype=torch.uint8)
		self.generator.set_state(state)

	def get(self):
		return self.generator
