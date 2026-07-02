import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalNet(nn.Module):
	def __init__(self, h, w, K, seed=0):
		super(ConvolutionalNet, self).__init__()
		torch.manual_seed(seed)
        
		# Convolutional layers
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
		
		# Pooling
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

		# Compute flattened size dynamically
		self._to_linear = None
		self._compute_linear_size(h, w)

		# Fully connected layers
		self.fc1 = nn.Linear(self._to_linear, 200)
		self.fc2 = nn.Linear(200, 200)
		self.fc3 = nn.Linear(200, K)

	def _compute_linear_size(self, h, w):
		"""Pass a dummy tensor to infer the flattened size."""
		x = torch.zeros(1, h, w)
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		self._to_linear = x.view(1, -1).shape[1]

	def forward(self, x):
		x = x.unsqueeze(1)

		# Conv block 1
		x = self.pool(F.relu(self.conv1(x)))
		
		# Conv block 2
		x = self.pool(F.relu(self.conv2(x)))
		
		# Flatten
		x = x.view(x.size(0), -1)
		
		# Fully connected layers
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		
		# Output layer (no activation, use with CrossEntropyLoss)
		x = self.fc3(x)
		
		return x
