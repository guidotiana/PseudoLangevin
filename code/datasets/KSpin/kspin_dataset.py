import torch
from torch.utils.data import Dataset

from .functions import generate_kspin_data


def generate_kspin_datasets(pars, device="cpu"):
	assert pars["P_train"]>0, "generate_kspin_datasets(): P_train is a necessary key of the pars dictionary. Its value must be non-negative."
	datasets = {
		"train": KSpinDataset(
			P=pars['P_train'],
			K=pars['K'],
			d=pars['d'],
			pflip=pars['pflip'],
			ref_vectors=None,
			seed=pars['seed_train'],
			device=device,
		)
	}

	for key in ["val", "test"]:
		
		P_key, seed_key = f"P_{key}", f"seed_{key}"
		if P_key not in pars.keys():
			continue
		assert pars[P_key]>0, "generate_kspin_datasets(): P_<key> must be non-negative for each key, 'train', 'val' and 'test'."
		if seed_key not in pars.keys():
            pars[seed_key] = 0
		
		datasets[key] = KSpinDataset(
			P=pars[P_key],
			K=pars['K'],
			d=pars['d'],
			pflip=pars['pflip'],
			ref_vectors=datasets['train'].ref_x,
			seed=pars[seed_key],
			device=device,
		)
	
	return datasets


class KSpinDataset(Dataset):

    def __init__(
        self,
        P, K, d,
        pflip,
		ref_vectors = None,
        one_hot_encode_labels = False,
        seed = 0,
        device = "cpu",
        load_from = None,
    ):
        self.settings = {
            "P": P,
            "K": K,
            "d": d,
            "pflip": pflip,
            "one_hot_encode_labels": one_hot_encode_labels,
            "seed": seed,
        }
        
        # Legend:
        # ref_x -> reference kspin vectors
        # x     -> flipped kspin vectors
        # y     -> labels
        if load_from is None:
            self.ref_x, self.x, self.y = generate_kspin_data(ref_vectors=ref_vectors, **self.settings)
            self.to(device)
        else:
            self.load(load_from)

    def __len__(self):
        return self.settings["P"]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
        
    def save(self, fn):
        log = {
            "ref_vectors": self.ref_x,
            "vectors": self.x.to("cpu"),
            "labels": self.y.to("cpu"),
            "settings": self.settings,
        }
        torch.save(log, fn)

    def load(self, fn, device="cpu", same_settings=True):
        log = torch.load(fn, map_location=torch.device("cpu"))

        if same_settings:
            check = all([v==self.settings[k] for k,v in log["settings"].items()])
            assert check, (
                f"SpinDataset.load(): ",
                f"the settings saved in '{fn}' do not coincide with the current ones. ",
                f"To avoid this error, set same_settings=False. ",
                f"Exit!",
            )
        else:
            self.settings = log["settings"]
            
        self.ref_x = log["ref_vectors"]
        self.x = log["vectors"]
        self.y = log["labels"]

        self.to(device)

    def to(self, device):
        device = device if isinstance(device, torch.device) else torch.device(device)
        if ("cuda" in device.type) and torch.cuda.is_available():
            self.x = self.x.to(device)
            self.y = self.y.to(device)
