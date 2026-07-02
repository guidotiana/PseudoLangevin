import torch
from torchvision import datasets, transforms


download_d = "./download"
data_d = "."

tfm = transforms.Compose([
    transforms.ToTensor(),
])
train_ds = datasets.MNIST(download_d, train=True,  download=True, transform=tfm)
test_ds  = datasets.MNIST(download_d, train=False, download=True, transform=tfm)

for ds, label in zip([train_ds, test_ds], ["train", "test"]):
	X = ds.data.detach().clone()
	y = torch.LongTensor(ds.targets)
	print(f"Label <{label}>:")
	print(f" X -> {X.shape}")
	print(f" y -> {y.shape}")
	torch.save((X,y), f"{data_d}/mnist_{label}.pt")
