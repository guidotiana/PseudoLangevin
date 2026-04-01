import os, sys, argparse
import torch
import torch.nn.functional as F

torch.cuda.empty_cache()

sys.path.append('../../code')
from datasets.ProjFashionMNIST.pfmnist_dataset import load_datasets
from samplers.pl_sampler import PLSampler
from models.plain_ffn.ffn import PlainFFNet
from models.nnmodel import NNModel
from utils.general import load_inputs, create_path, find_path

# Load the input files and make the results directory
def prepare_directory(args):
	pars = {key: load_inputs(args.pars_file, start=f"## {key}", end="##") for key in ["model", "data", "sampler"]}
	settings = load_inputs(args.settings_file)

	create_path(settings['results_dir'])
	settings['results_dir'] = find_path(raw_path=settings['results_dir'], dname='sim', pfile=args.pars_file, pname='pars.txt', lpfunc=load_inputs)
	if 'weights_dir' not in settings.keys():
		settings['weights_dir'] = f"{settings['results_dir']}/weights"
		create_path(settings['weights_dir'])

	return pars, settings


# Main
def main(args):
	print(f'PID: {os.getpid()}\n')

	print('Loading inputs...')
	pars, settings = prepare_directory(args)

	print('Loading nn-model...')
	net = PlainFFNet(
			input_dim=pars['model']['input_dim'],
			hidden_dims=pars['model']['hidden_dims'],
			output_dim=pars['model']['output_dim'],
			seed=pars['model']['model_seed'],
	)
	model = NNModel(net)

	print('Initializing datasets...')
	datasets = load_datasets(**pars['data'])

	print('Defining cost and metric functions...')
	Cost = lambda logits, target: F.cross_entropy(logits, target)
	Metric = lambda logits, target: 1. - (logits.argmax(dim=1) == target).float().mean()

	print('Initializing sampler...')
	sampler = PLSampler(
			model=model,
			datasets=datasets,
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
		default = "pars.txt",
		help = "str variable, path to the parameters file used in the simulation. Default: 'pars.txt'."
	)
	parser.add_argument(
		"--settings-file",
		type = str,
		default = "settings.txt",
		help = "str variable, path to a secondary input file for specifics which do not alter the simulation. Default: 'settings.txt'."
	)
	return parser
    
if __name__ == '__main__':
	parser = create_parser()
	args = parser.parse_args()
	main(args)
