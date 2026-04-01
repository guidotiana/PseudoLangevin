import os, sys, argparse
import torch
import torch.nn.functional as F

torch.cuda.empty_cache()

sys.path.append('../../code')
from models.nnmodel import NNModel
from models.transformer.transformer import Transformer, TFNet
from datasets.Protein.protein_dataset import load_datasets, VOCAB_SIZE, MASK_ID
from samplers.cpl_sampler import ConstrainedPLSampler
from utils.general import load_inputs, create_path, find_path

# Load the input files and make the results directory
def prepare_directory(args):
	pars = {key: load_inputs(args.pars_file, start=f"## {key}", end="##") for key in ["model", "sampler", "data"]}
	pars['data']['filename'] = '../../code/datasets/Protein/SequencesMasked.txt'
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
	d = pars['model']['d']
	H = pars['model']['H']
	m = pars['model']['m']
	L = pars['model']['L']
	n = pars['model']['n']
	transformer = Transformer(VOCAB_SIZE, d, H, m, L, n, dropout=0.)
	net = TFNet(transformer, d, VOCAB_SIZE)
	model = NNModel(net)

	print('Initializing datasets...')
	datasets = load_datasets(**pars['data'])

	print('Defining cost and metric functions...')
	def Cost(fx, masks):
		log_probs = F.log_softmax(fx, dim=-1)
		costs = [
			log_probs[iseq, mask[:,0], mask[:,1]].mean() if len(mask)>0 else torch.tensor(0)
			for iseq, mask in enumerate(masks)
		]
		return sum(costs) / len(masks)

	def Metric(fx, masks):
		softmax_fx = F.softmax(fx, dim=-1)
		total_mean = 0.0
		for iseq, mask in enumerate(masks):
			count = 0.
			predictions = torch.argmax(softmax_fx[iseq], dim=1)[mask[:,0]]
			for i, x in enumerate(mask[:,1]):
				count += (predictions[i] == x)
			total_mean += count/len(predictions)
		return 1. - total_mean / len(masks)	

	print('Initializing sampler...')
	sampler = ConstrainedPLSampler(
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
	print(f'Total time: {format(sampler.log["data"]["time"]/3600., ".2f")} h')


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
