from io import StringIO

import math
import numpy as np
import torch
from torcheval.metrics.functional import mean as weighted_mean
from time import process_time as ptime

from utils.operations import merge_dict, merge_dicts, compute_q, compute_d, compute_mod2

# Default precision
torch.set_default_dtype(torch.float64)



### ---------------------------------------------------- ###
### Running-Average Double-Noise Pseudo-Langevin Sampler ###
### ---------------------------------------------------- ###
class RAPLSampler():

	def __init__(self, model, Cost, Metric, dataset, generator, name:str='RAPLSampler'):
		self.model = model
		self.Cost = Cost
		self.Metric = Metric
		self.dataset = dataset
		self.generator = generator
		self.name = name

		self._init_attributes()


	def _setup(self, pars, settings, logfile, keep_going):

		# Check settings
		if settings is None:
			settings = self.settings.copy()
		elif isinstance(settings, dict):
			for key, value in self.settings.items():
				if (key not in settings) or (settings[key] is None):
					settings[key] = value
		else:
			raise TypeError("{self.name}._setup(): settings expected types are either Dict or None, but found {type(settings)}.")

		assert settings["data_step"] > 0 and settings["pars_step"] > 0 and settings["log_step"] > 0, (
			f"{self.name}._setup(): data_step, pars_step and log_step in settings must be positive, "
			f"but found {settings['data_step']}, {settings['pars_step']} and {settings['log_step']} respectively."
		)

		if settings["data_step"] <= settings["log_step"]:
			settings["log_step"] = round(settings["log_step"]/settings["data_step"]) * settings["data_step"]
		else:
			settings["data_step"] = round(settings["data_step"]/settings["log_step"]) * settings["log_step"]

		# Start a new sampling...
		if logfile is None:
			default_data = {
				'q': 1.,
				'step_d': 0.,
				'step_dt': 0.,
				'eff_v': 0.,
				'dK': 0.,
			}

			# ... from the last checkpoint (stored in log) with new parameters
			if keep_going and len(self.log)>0:
				data = self.log['data'].copy()
				t0 = ptime() - data['time']
				data = merge_dict(from_dict=default_data, into_dict=data, overwrite=True)

			# ... from scratch
			else:
				t0 = ptime()
				data = {'move': 0, 'time': 0.}
				obs = self._compute_observables(x=self.dataset.x, y=self.dataset.y, lamda=pars['lamda'], backward=False)
				data = merge_dicts(from_dicts=[default_data, obs], into_dict=data)

			pars, lists = self._init_pars_and_lists(pars=pars, verbose=settings['verbose'])
			momenta = {
				layer: torch.randn(values.shape, device=self.model.device, generator=self.generator.get()) * math.sqrt(pars['T']*pars['M'][layer])
				for layer, values in self.model.weights.items()
			}
			weights_ref = self.model.copy(grad=False)

			self._extend_buffer(data, header=data['move']==0)
			self._save_log(data, pars, lists, momenta, settings, is_ref=True)
			if settings["print_step"] > 0: self._print_status(data)

		# Restart an interrupted simulation from the last saved logfile
		else:
			self.log = torch.load(logfile)
            
			data = self.log['data'].copy()
			t0 = ptime()-data['time']

			pars = torch.load(self.log['files']['pars'])
			lists = torch.load(self.log['files']['lists'])
			momenta = torch.load(self.log['files']['momenta'], map_location=self.model.device, weights_only=True)
			self.model.load(self.log['files']['weights'])
			self.generator.load(self.log['files']['generator'])
			weights_ref = torch.load(self.log['files']['weights_ref'], map_location=self.model.device, weights_only=True)

			if settings["print_step"] > 0:
				print(f'// {self.name} status register:')
				print(f'{self.separator}\n{self.header}\n{self.separator}')
				self._print_status(data)

		# Correct variable types
		pars = self._correct_types(pars, 'pars')
		lists = self._correct_types(lists, 'lists')
		data = self._correct_types(data, 'data')

		return data, pars, lists, momenta, weights_ref, t0, settings


	def sample(
			self,
			pars: dict,
			settings: dict | None = None,
			logfile: str| None = None,
			keep_going: bool = False,
	):
		data, pars, lists, momenta, weights_ref, t0, settings = self._setup(pars, settings, logfile, keep_going)

		for move in range(data['move'], pars['tot_moves']):            
			
			# Perform a step and sample data
			if (move+1)%settings['data_step'] == 0:
				K_i = self._compute_K(momenta, pars)
				weights_i = self.model.copy(grad=False)
				step_dt = ptime()
				momenta, pars, lists = self._step_and_update(momenta, pars, lists)
				step_dt = ptime()-step_dt
				step_d = compute_d(self.model.weights, weights_i)

				data = {
					'move': move+1,
					'time': ptime()-t0,
					'q': compute_q(self.model.weights, weights_ref),
					'step_d': step_d,
					'step_dt': step_dt,
					'eff_v': step_d/step_dt,
					'dK': self._compute_K(momenta, pars) - K_i
				}
				obs = self._compute_observables(x=self.dataset.x, y=self.dataset.y, lamda=pars['lamda'], backward=False)
				data = merge_dict(from_dict=obs, into_dict=data)
				self._extend_buffer(data)

				del weights_i, K_i, step_dt, step_d

			# Just perform a step
			else:
				momenta, pars, lists = self._step_and_update(momenta, pars, lists)

			if (move+1)%settings["log_step"] == 0: self._save_log(data, pars, lists, momenta, settings)  #Full logfile saving
			elif (move+1)%settings["pars_step"] == 0: self._save_pars(pars, settings, move+1)            #Limited to pars, only if log has not already been saved
			if settings["print_step"] > 0 and (move+1)%settings["print_step"] == 0: self._print_status(data)


	def _step_and_update(self, momenta, pars, lists):
		#Integrate the Langevin equations using the Bussi-Parrinello velocity-Verlet-like integrator
		old_grad = self._compute_grad(pars)
		old_noise = self._generate_noise()

		with torch.no_grad():
			for layer in self.model.weights:
				self.model.weights[layer] += momenta[layer]*pars['c1']*pars['dt']/pars['M'][layer] - old_grad[layer]*pars['dt']**2./(2.*pars['M'][layer]) + old_noise[layer]*pars['k_wn'][layer]*pars['dt']/pars['M'][layer]

		new_grad = self._compute_grad(pars)
		new_noise = self._generate_noise()

		for layer in self.model.weights:
			momenta[layer] = momenta[layer]*pars['c1']**2. - (new_grad[layer]+old_grad[layer])*pars['c1']*pars['dt']/2. + old_noise[layer]*pars['c1']*pars['k_wn'][layer] + new_noise[layer]*math.sqrt((1.-pars['c1']**2.)*pars['k_mb'][layer]**2. + pars['k_wn'][layer]**2.)

		# Update grads and grad2s dictionaries using the recently computed gradients
		# Then, if allowed, estimate the current variance and add it to the list of variances
		if lists['len_grads'] == 0:
			for layer in lists['grads']:
				lists['grads'][layer] = [old_grad[layer].float(), new_grad[layer].float()]
				lists['grad2s'][layer] = [(old_grad[layer]**2.).float(), (new_grad[layer]**2.).float()]
			lists['len_grads'] += 2

		elif lists['len_grads'] < pars['max_len_grads']:
			for layer in lists['grads']:
				lists['grads'][layer] += [old_grad[layer].float(), new_grad[layer].float()]
				lists['grad2s'][layer] += [(old_grad[layer]**2.).float(), (new_grad[layer]**2.).float()]
			lists['len_grads'] += 2

		else:
			if lists['len_vars'] < pars['max_len_vars']:
				var_shift = 0
				lists['len_vars'] += 1
			else:
				var_shift = 1

			for layer in lists['grads']:
				lists['grads'][layer] = lists['grads'][layer][2:]
				lists['grads'][layer] += [old_grad[layer].float(), new_grad[layer].float()]
				lists['grad2s'][layer] = lists['grad2s'][layer][2:]
				lists['grad2s'][layer] += [(old_grad[layer]**2.).float(), (new_grad[layer]**2.).float()]
				
				lists['vars'][layer].append(
						( torch.stack(lists['grad2s'][layer], dim=0).mean(axis=0) - torch.stack(lists['grads'][layer], dim=0).mean(axis=0)**2. ).mean().item()
				)
				lists['vars'][layer] = lists['vars'][layer][var_shift:]


		# If the list of variances is sufficiently populated, start updating the value of vars used for the integration
		if lists['len_vars'] >= pars['min_len_vars']:
			# TODO:
			# if stationary:
			#	 update also other variables (T_mb, k_mb, k_wn)
			# else:
			#	 see below, update just var
			for layer in lists['vars']:
				pars['var'][layer] = weighted_mean(
						torch.tensor(lists['vars'][layer]),
						pars['var_weights'][pars['max_len_vars']-lists['len_vars']:],
				).item()

		return momenta, pars, lists


	def _init_pars_and_lists(self, pars, verbose=True):
		# Start by computing a starting value for the mini-batch extraction noise variance
		# This is done by extracting up to <max_extractions> mini-batches and computing the variance element-wise
		print("\n!!! Initialization of the mini-batch extraction variance and related parameters !!!\n")
		sum_grad, sum2_grad = {}, {}
		for iext in range(pars['extractions']):
			grad = self._compute_grad(pars)
			if iext == 0:
				for layer in grad:
					sum_grad[layer] = grad[layer].detach()
					sum2_grad[layer] = (grad[layer]**2.).detach()
			else:
				for layer in grad:
					sum_grad[layer] += grad[layer].detach()
					sum2_grad[layer] += (grad[layer]**2.).detach()
		tot_extractions = pars['extractions']
		"""
		In this version, we are adopting a different standard deviation for each layer of the network.
		An alternative could be to employ different values for each neuron.
		"""
		prev_var = {
			layer: (sum2_grad[layer]/(tot_extractions) - (sum_grad[layer]/(tot_extractions))**2.).mean().item()
			for layer in sum_grad
		}

		if verbose:
			print(f"Extractions {tot_extractions} (first estimate)")
			for layer in prev_var:
				print(f"- {layer}:\t{prev_var[layer]:.3e}")
			print()

		while tot_extractions < pars['max_extractions']:
			for _ in range(pars['extractions']):
				grad = self._compute_grad(pars)
				for name in grad:
					sum_grad[name] += grad[name].detach()
					sum2_grad[name] += (grad[name]**2.).detach()
			tot_extractions += pars['extractions']
			curr_var = {
				layer: (sum2_grad[layer]/(tot_extractions) - (sum_grad[layer]/(tot_extractions))**2.).mean().item()
				for layer in sum_grad
			}

			is_converged = [abs(curr_var[layer]-prev_var[layer])/prev_var[layer] < pars['threshold_est'] for layer in prev_var]

			if verbose:
				print(f"Extractions: {tot_extractions}")
				for ilayer, layer in enumerate(prev_var):
					print(f"- {layer}:\t{prev_var[layer]:.3e}\t{curr_var[layer]:.3e}\t{is_converged[ilayer]}")
				print()

			prev_var = curr_var.copy()
			if all(is_converged):
				break

		if verbose:
			print(f"Reached convergence in {tot_extractions} extractions.")

		# Once convergence is reached, the other parameters necessary for integration can be computed too
		pars['var'] = prev_var.copy()
		pars['c1'] = math.sqrt(1.-pars['m1']**2.)
		pars['M'], pars['T_mb'], pars['k_mb'], pars['k_wn'] = {}, {}, {}, {}
		for layer in self.model.weights:
			pars['M'][layer] = (pars['var'][layer]*pars['dt']**2.)/(4.*pars['T_mb_0']*pars['m1']**2.)
			pars['T_mb'][layer] = pars['T_mb_0']
			pars['k_mb'][layer] = pars['dt']*math.sqrt(pars['var'][layer])/2.
			pars['k_wn'][layer] = pars['k_mb'][layer] * math.sqrt(pars['T']/pars['T_mb_0'] - 1.)

		pars['corr_stime'] = -(pars['max_len_vars']-1)/math.log(pars['min_var_weight'])
		pars['var_weights'] = torch.exp( (torch.arange(pars['max_len_vars']) - (pars['max_len_vars']-1)) / pars['corr_stime'] )

		# Finally the lists of gradients, squared gradients and variances are also initialized
		# TODO:
		# add a list for data storing, to check
		# when to begin the full-update
		lists = {
			'grads': {k: None for k in pars['var']},
			'grad2s': {k: None for k in pars['var']},
			'len_grads': 0,
			'vars': {k: [v] for k, v in pars['var'].items()},
			'len_vars': 1,
		}
        
		print(f'Back to the simulation.\n')
		print(f'// {self.name} status register:')
		print(f'{self.separator}\n{self.header}\n{self.separator}')
		return pars, lists


	def _generate_noise(self):
		noise = {
			layer: torch.randn(values.shape, device=self.model.device, generator=self.generator.get())
			for layer, values in self.model.weights.items()
		}
		return noise


	def _compute_grad(self, pars):
		mb_mask = torch.zeros((len(self.dataset),), dtype=torch.bool)
		#while mb_mask.sum() < pars['mbs']:
		#	idx = torch.randint(low=0, high=len(self.dataset), size=(1,), device=self.model.device, generator=self.generator.get())
		#	mb_mask[idx] = True
		mb_idxs = torch.randperm(len(self.dataset), device=self.model.device, generator=self.generator.get())[:pars['mbs']]
		mb_mask[mb_idxs] = True

		x, y, _ = self.dataset[mb_mask]
		_ = self._compute_observables(x, y, pars['lamda'], backward=True)
		grad = self.model.copy(grad=True)
		self.model.zero_grad()
		return grad
    
    
	def _compute_observables(self, x, y, lamda, backward=True):
		fx = self.model(x)
		cost = self.Cost(fx, y)
		mod2 = compute_mod2(self.model.weights)
		loss = cost + (lamda/2.)*mod2

		if backward:
			loss.backward()
		else: # observables computed only on the full batch, outside of _step_and_update()
			metric = self.Metric(fx, self.dataset.y)

		return {
			'loss':loss.detach().item(), 
			'cost':cost.detach().item(), 
			'mod2':mod2.detach().item(), 
			'metric':metric.detach().item() if not backward else None,
		}


	def _compute_K(self, momenta, pars):
		K = 0.
		for layer in momenta:
			K += (0.5*(momenta[layer]**2.)/pars["M"][layer]).sum()
		return K.item()


	def _extend_buffer(self, dikt, header=False):
		if header:
			header, line = '', ''
			for key in dikt:
				header = header + f'{key}\t'
				line = line + f'{dikt[key]}\t'
			self.buffer.write(f"{header[:-1]}\n{line[:-1]}\n")
		else:
			line = ''
			for key in dikt: line = line + f'{dikt[key]}\t'
			self.buffer.write(f"{line[:-1]}\n")


	def _flush_buffer(self, settings):
		with open(f'{settings["results_dir"]}/data.dat', 'a') as f:
			print(self.buffer.getvalue(), file=f, end="")
		self.buffer.seek(0)
		self.buffer.truncate(0)


	def _save_pars(self, pars, settings, move):
		torch.save(pars, f'{settings["pars_dir"]}/pars_{move}.pt')


	def _save_log(self, data, pars, lists, momenta, settings, is_ref=False):
		self._flush_buffer(settings)
		self._save_pars(pars, settings, data['move'])
		torch.save(lists, f'{settings["results_dir"]}/lists.pt')
		torch.save(momenta, f'{settings["weights_dir"]}/momenta_{data["move"]}.pt')
		self.model.save(f'{settings["weights_dir"]}/weights_{data["move"]}.pt')
		self.generator.save(f'{settings["results_dir"]}/generator.npy')

		self.log['data'] = data.copy()
		self.log['files']['pars'] = f'{settings["pars_dir"]}/pars_{data["move"]}.pt'
		self.log['files']['lists'] = f'{settings["results_dir"]}/lists.pt'
		self.log['files']['momenta'] = f'{settings["weights_dir"]}/momenta_{data["move"]}.pt'
		self.log['files']['weights'] = f'{settings["weights_dir"]}/weights_{data["move"]}.pt'
		self.log['files']['generator'] = f'{settings["results_dir"]}/generator.npy'
		if is_ref:
			self.log['files']['weights_ref'] = f'{settings["weights_dir"]}/weights_{data["move"]}.pt'
		
		torch.save(self.log, f'{settings["results_dir"]}/log.pt')


	def _print_status(self, data):
		data['time_h'] = data["time"] / 3600.

		line = ''
		for key, _, fp in self.formatter['sampling']: line = f'{line}|{format(data[f"{key}"], f".{fp}f"):^12}'
		line = f'{line}|' + ''.join([' ']*5)
		for key, _, fp in self.formatter['efficiency']: line = f'{line}|{format(data[f"{key}"], f".{fp}f"):^12}'
		line = f'{line}|'

		data.pop('time_h')
		print(f'{line}\n{self.separator}')


	def _correct_types(self, d, dname):
		if dname == 'pars':
			types_and_keys = [(int, ['moves', 'tot_moves', 'mbs', 'max_len_grads', 'max_len_vars', 'min_len_vars', 'max_extractions', 'extractions'])]
		else:
			types_and_keys = [(int, ['move'])]
        
		for key in d:
			for _type, keys in types_and_keys:
				if key in keys:
					d[key] = _type(d[key])

		return d


	def _init_attributes(self):
		self.buffer = StringIO()

		self.log = {
            'data': None,
            'files': {
                'pars': None,
				'lists': None,
                'momenta': None,
                'weights': None,
                'generator': None,
				'weights_ref': None,
            },
        }

		self.settings = {
			"results_dir": ".",
			"pars_dir": "./pars",
			"weights_dir": "./weights",
			"data_step": 10,
			"pars_step": 100,
			"log_step": 1000,
			"print_step": 1000,
			"verbose": True,
		}

		self.formatter = {
			'sampling':[
				['move', 'move', 0],
				['loss', 'U', 5],
				['cost', 'loss', 5],
				['metric', 'metric', 5],
				['mod2', 'mod2', 1],
				['time_h', 'time', 2],
            ],
			'efficiency':[
				['move', 'move', 0],
				['q', 'q', 5],
				['eff_v', 'eff_v', 5],
			],
		}

		self.separator = ''.join(['-']*(13*len(self.formatter['sampling'])+1)) + ''.join([' ']*5) + ''.join(['-']*(13*len(self.formatter['efficiency'])+1))
		self.header = ''
		for _, symbol, _ in self.formatter['sampling']: self.header = f'{self.header}|{symbol:^12}'
		self.header = f'{self.header}|' + ''.join([' ']*5)
		for _, symbol, _ in self.formatter['efficiency']: self.header = f'{self.header}|{symbol:^12}'
		self.header = f'{self.header}|'
