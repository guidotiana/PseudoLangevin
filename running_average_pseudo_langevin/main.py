import os, argparse
import torch
import numpy as np

torch.set_default_dtype(torch.float64)
torch.cuda.empty_cache()

import sys
sys.path.append('../shared_code')

from datasets.sequences_dataset import load_seq, MaskedDataset
from samplers.efficient_rapl_sampler import RAPLSampler
from generator.custom_generator import CustomGenerator
from utils.general import load_stuff, load_inputs, create_path, find_path
from utils.energies import choose_cost, choose_metric
from models.transformer import Transformer, TModel
from models.nnmodel import NNModel



# Check simulation inputs
def check_inputs(pars, extra, args):

	## Set directories
	create_path(extra['results_dir'])
	extra['results_dir'] = find_path(raw_path=extra['results_dir'], dname='sim', pfile=args.pars_file, pname='pars.txt', lpfunc=load_inputs)
	for name in ("pars", "weights"):
		extra[f"{name}_dir"] = f"{extra['results_dir']}/{name}"
		create_path(extra[f"{name}_dir"])

	## Device and threads
	if 'cuda' in extra['device']:
		extra['device'] = extra['device'] if torch.cuda.is_available() else 'cpu'
	assert extra['num_threads'] <= 5, f'main.check(): invalid value for "num_threads" variable: {extra["num_threads"]}. Allowed values: num_threads <= 5.'
	torch.set_num_threads(extra['num_threads'])

	return pars, extra


# Check if previous simulation data exists
def get_exists(extra):
	results_files = ['data.dat', 'pars.txt', 'generator.npy', 'log.pt', 'lists.pt']
	weights_files = [f"{name}_0.pt" for name in ('weights', 'momenta')]
	exists = all([rf in os.listdir(extra['results_dir']) for rf in results_files] + ["pars_0.pt" in os.listdir(extra['pars_dir'])] + [wf in os.listdir(extra['weights_dir']) for wf in weights_files])
	return exists


# Restart previous simulation
def restart(pars, extra):
	
	## Get last simulated move
	last_move = max([int(fn.strip(f'weights_.pt')) for fn in os.listdir(extra['weights_dir']) if os.path.isfile(f'{extra["weights_dir"]}/{fn}') and ("weights_" in fn)])

	## Clean files after the last log
	data = load_stuff(f'{extra["results_dir"]}/data.dat')
	data = data.loc[np.array(data['move']) <= last_move, :].reset_index(drop=True)
	with open(f'{extra["results_dir"]}/data.dat', 'w') as f:
		data.to_csv(f, sep='\t', index=False)

	for pf in os.listdir(extra['pars_dir']):
		move = int(pf.split('_')[-1].rstrip('.pt'))
		if move > last_move:
			os.remove(f'{extra["pars_dir"]}/{pf}')

	## Define list of spars to feed (sequentially) to the sampler
	tot_moves = 0
	spars_list = []
	for idx, (stime, dt) in enumerate(zip(pars['sampler']['stime_list'], pars['sampler']['dt_list'])):
		moves = int(stime/dt)
		tot_moves += moves
		if last_move > tot_moves:
			continue

		spars = {'moves': moves, 'tot_moves': tot_moves}
		for key, value in pars['sampler'].items():
			if '_list' in key:
				spars[key.split('_list')[0]] = value[idx]
			else:
				spars[key] = value
		spars['T_mb_0'] = spars['T']*spars['Tratio']
		spars_list.append(spars)
    
	## Find logfile to restart the simulation
	logfile = f"{extra['results_dir']}/log.pt"

	return logfile, spars_list


# Start new simulation from scratch
def reset(pars, extra):

	## Remove old simulation files (if they exist)
	for d in [extra['results_dir'], extra['pars_dir'], extra['weights_dir']]:
		for fn in os.listdir(d):
			if fn == 'pars.txt': continue
			if os.path.isfile(f'{d}/{fn}'):
				os.remove(f'{d}/{fn}')

	## Define list of spars to feed (sequentially) to the sampler
	tot_moves = 0
	spars_list = []
	for idx, (stime, dt) in enumerate(zip(pars['sampler']['stime_list'], pars['sampler']['dt_list'])):
		moves = int(stime/dt)
		tot_moves += moves
        
		spars = {'moves': moves, 'tot_moves': tot_moves}
		for key, value in pars['sampler'].items():
			if '_list' in key:
				spars[key.split('_list')[0]] = value[idx]
			else:
				spars[key] = value
		spars['T_mb_0'] = spars['T']*spars['Tratio']
		spars_list.append(spars)
	
	return None, spars_list


# Sampler parameters summary
def summary(spars, extra, model, title):
	fixed = ''
	for name, param in model.NN.named_parameters():
		if not param.requires_grad:
			fixed = f'{fixed}, {name}'
	fixed = f'({fixed[2:]})'

	lines = []
	lines.append(f'# {title} parameters summary:')
	lines.append(f'# ')
	lines.append(f'# moves:                  {spars["moves"]:.1e}')
	lines.append(f'# temperature:            {spars["T"]:.1e}')
	lines.append(f'# mini-batch temperature: {spars["T_mb_0"]:.1e}')
	lines.append(f'# mobility:        {spars["m1"]:.2f}')
	lines.append(f'# time step:       {spars["dt"]:.1e}')
	lines.append(f'# mini-batch size: {spars["mbs"]:.0f}')
	lines.append(f'# lambda:          {spars["lamda"]:.1e}')
	lines.append(f'# ')
	lines.append(f'# fixed layers: {fixed}')
	lines.append(f'# ')
	lines.append(f'# results directory: {extra["results_dir"]}')
	lines.append(f'# pars directory:    {extra["pars_dir"]}')
	lines.append(f'# weights directory: {extra["weights_dir"]}')
	lines.append(f'# restart:           {bool(extra["restart"])}')
	lines.append(f'# ')

	max_length = max([len(line) for line in lines])
	print('\n')
	print(''.join(['#'] * (max_length+2)))
	for line in lines:
		line = line + ''.join([' '] * (max_length-len(line)+1)) + '#'
		print(line)
	print(''.join(['#'] * (max_length+2)))
	print()


# Main
def main(args):
	print(f'PID: {os.getpid()}\n')

	print('Loading inputs...')
	pars = {key: load_inputs(args.pars_file, start=f"## {key}", end="##") for key in ["model", "dataset", "cost", "metric", "generator", "sampler"]}
	extra = load_inputs(args.extra_file)
	pars, extra = check_inputs(pars, extra, args)

	print('Loading model...')
	VOCAB_SIZE = 21
	d = pars['model']['d']
	H = pars['model']['H']
	m = pars['model']['m']
	L = pars['model']['L']
	n = pars['model']['n']
	transformer = Transformer(VOCAB_SIZE, d, H, m, L, n, extra['device'], dropout=0.)
	tmodel = TModel(transformer, d, VOCAB_SIZE, extra['device'])
	model = NNModel(tmodel, extra['device'], f=pars['model']['from'])

	print('Defining cost and metric functions...')
	Cost = choose_cost(
			name=pars['cost']['cost'],
			device = extra['device'],
			**pars['cost']
	)
	Metric = choose_metric(
			name=pars['metric']['metric'],
			device = extra['device'],
			**pars['metric']
	)

	print('Initializing dataset...')
	masked_sequences, sequences, mask = load_seq(pars['dataset']['filename'], pars['dataset']['n_data'])
	index_val = int(len(sequences) - len(sequences) * pars['dataset']['validation_rate'])
	dataset = MaskedDataset(
			x = masked_sequences[:index_val],
			y = mask[:index_val],
			original_sequences = sequences[:index_val],
			P = index_val,
			device=extra['device'],
	)


	print('Initializing generator...')
	generator = CustomGenerator(
			seed=pars['generator']['generator_seed'],
			device=extra['device']
	)

	print('Initializing sampler...')
	sampler = RAPLSampler(
			model=model, 
			Cost=Cost,
			Metric=Metric,
			dataset=dataset, 
			generator=generator,
	)

	exists = get_exists(extra)
	if extra['restart'] and exists:
		print('Recovering previous simulation...')
		logfile, spars_list = restart(pars, extra)
	else:
		print('Resetting directories...')
		logfile, spars_list = reset(pars, extra)

	print(f'Starting the simulation!')
	for i, spars in enumerate(spars_list):
		logfile = logfile if i==0 else None
		keep_going = i>0
		
		summary(spars, extra, model, sampler.name)
		sampler.sample(
				pars=spars,
				settings=extra,
				logfile=logfile,
				keep_going=keep_going,
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
		"--extra-file",
		type = str,
		default = "inputs/extra.txt",
		help = "str variable, path to a secondary input file for specifics which do not alter the simulation. Default: 'inputs/extra.txt'."
	)
	return parser
    
if __name__ == '__main__':
	parser = create_parser()
	args = parser.parse_args()
	main(args)
