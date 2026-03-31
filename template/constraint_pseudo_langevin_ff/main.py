import os, sys, argparse
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
torch.cuda.empty_cache()

sys.path.append('../code')
from datasets.sequences_dataset import load_seq, MaskedDataset
from generator.custom_generator import CustomGenerator
from utils.general import load_stuff, load_inputs, create_path, find_path
from utils.energies import choose_cost, choose_metric
from models.pooling_ff.layers import Model, FF
from samplers.cpl_sampler import ConstraintPLSampler
from models.nnmodel import NNModel

# Load the input files, check them and
# and make the results directory
def prepare_directory(args):
	pars = {key: load_inputs(args.pars_file, start=f"## {key}", end="##") for key in ["model", "sampler", "dataset","cost","metric"]}
	settings = load_inputs(args.settings_file)

	create_path(settings['results_dir'])
	settings['results_dir'] = find_path(raw_path=settings['results_dir'], dname='sim', pfile=args.pars_file, pname='pars.txt', lpfunc=load_inputs)
	settings['weights_dir'] = f"{settings['results_dir']}/weights"
	create_path(settings['weights_dir'])

	return pars, settings


# Main
def main(args):
	print(f'PID: {os.getpid()}\n')

	print('Loading inputs...')
	pars, settings = prepare_directory(args)

	print('Loading model...')
	VOCAB_SIZE = 21
	d = pars['model']['d']
	m = pars['model']['m']
	n = pars['model']['n']
	FFModel = FF(VOCAB_SIZE, d, m, n,'cpu', mask_id=20, dropout=0.)
	model_FF = Model(FFModel, m, VOCAB_SIZE, 'cpu')
	model = NNModel(model_FF, 'cpu', f=pars['model']['from'])
	
    print('Initializing dataset...')
	masked_sequences, sequences, mask = load_seq(pars['dataset']['filename'], pars['dataset']['n_data'])
	index_val = int(len(sequences) - len(sequences) * pars['dataset']['validation_rate'])
	dataset = MaskedDataset(
			x = masked_sequences[:index_val],
			y = mask[:index_val],
			original_sequences = sequences[:index_val],
			P = index_val,
	)
	dataset_val = MaskedDataset(
			x = masked_sequences[index_val:],
			y = mask[index_val:],
			original_sequences = sequences[index_val:],
			P = index_val,
	)


	print('Defining cost and metric functions...')
	Cost = choose_cost(
			name=pars['cost']['cost'],
			**pars['cost']
	)
	Metric = choose_metric(
			name=pars['metric']['metric'],
			**pars['metric']
	)

	print('Initializing sampler...')
	sampler = ConstraintPLSampler(
			model=model,
			dataset=dataset,
			dataset_val=dataset_val,
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
	print(f'Total time: {format(sampler.log["data"]["time"]/3600., ".2f")} h')


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
