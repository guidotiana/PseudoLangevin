import os, sys, argparse
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
torch.cuda.empty_cache()

sys.path.append('../code')
from datasets.KSpin.kspin_dataset import KSpinDataset
from samplers.hmc_sampler import HMCSampler
from models.ffn.ffn import FeedforwardNet
from models.nnmodel import NNModel
from utils.general import load_inputs, create_path, find_path



# Load the input files, complete them and
# and make the results directory
def prepare_directory(args):
	pars = {key: load_inputs(args.pars_file, start=f"## {key}", end="##") for key in ["model", "data", "sampler"]}
	settings = load_inputs(args.settings_file)

	create_path(settings['results_dir'])
	settings['results_dir'] = find_path(raw_path=settings['results_dir'], dname='sim', pfile=args.pars_file, pname='pars.txt', lpfunc=load_inputs)
	settings['weights_dir'] = f"{settings['results_dir']}/weights"
	create_path(settings['weights_dir'])

    dims = [pars["model"]["input_dim"]] + pars["model"]["hidden_dims"] + [pars["model"]["output_dim"]]
    pars["model"]["N"] = sum([ (dims[i]+1)*dims[i+1] for i in range(len(dims)-1) ])
    pars["data"]["P"] = int(pars["data"]["alpha"]*pars["model"]["N"])
    if "Lamda" in pars["sampler"]:
        pars["sampler"]["lamda"] = pars["sampler"]["Lamda"] / pars["model"]["N"]
        pars["sampler"].pop("Lamda")

    return pars, settings


# Main
def main(args):
	print(f'PID: {os.getpid()}\n')

	print('Loading inputs...')
	pars, settings = prepare_directory(args)

	print('Loading model...')
	net = FeedforwardNet(
			input_dim=pars['model']['input_dim'],
			hidden_dims=pars['model']['hidden_dims'],
			output_dim=pars['model']['output_dim'],
            seed=pars['model']['model_seed'],
	)
	model = NNModel(net)

	print('Initializing dataset...')
	dataset = KSpinDataset(
			P=pars['data']['P'], 
			K=pars['data']['K'], 
			d=pars['data']['d'], 
			pflip=pars['data']['pflip'], 
			seed=pars['data']['data_seed'],
	)

	print('Defining cost and metric functions...')
	Cost = lambda logits, target: F.cross_entropy(logits, target)
	Metric = lambda logits, target: (logits.argmax(dim=1) == target).float().mean()

	print('Initializing sampler...')
	sampler = HMCSampler(
			model=model,
			dataset=dataset,
			Cost=Cost,
			Metric=Metric,
	)

	print(f'Starting the simulation!')
	sampler.sample(
			pars=pars["sampler"],
			settings=settings,
			start_fn=pars["model"]["from"],
	)
	print(f'\nSimulation completed!')
	

def create_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--pars-file",
		type = str,
		default = "inputs/pars.txt",
		help = "str variable, path to the parameters file used in the simulation. Default: 'inputs/pars.txt'."
	)
	parser.add_argument(
		"--settings-file",
		type = str,
		default = "inputs/settings.txt",
		help = "str variable, path to a secondary input file for specifics which do not alter the simulation. Default: 'inputs/settings.txt'."
	)
	return parser
    
if __name__ == '__main__':
	parser = create_parser()
	args = parser.parse_args()
	main(args)
