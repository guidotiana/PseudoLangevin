import math
import numpy as np
import torch
from time import process_time as ptime

from utils.operations import merge_dict, merge_dicts, compute_q, compute_d, compute_mod2

# Default precision
torch.set_default_dtype(torch.float64)



### ------------------------------------ ###
### Double-Noise Pseudo-Langevin Sampler ###
### ------------------------------------ ###
class PLSampler():

	def __init__(self, model, Cost, Metric, dataset, generator, name:str='PLSampler'):
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

		assert settings["data_step"] > 0 and settings["log_step"] > 0, (
			f"{self.name}._setup(): both data_step and log_step in settings must be positive, "
			f"but found {settings['data_step']} and {settings['log_step']} respectively."
		)

		if settings["data_step"]<=settings["log_step"]:
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
				obs = self._compute_observables(x=self.dataset.x, y=self.dataset.y, lamda=pars['lamda'], backward=False, extra=True)
				data = merge_dicts(from_dicts=[default_data, obs], into_dict=data)

			pars = self._update_pars(pars=pars, verbose=settings['verbose'])
			momenta = {
				layer: torch.randn(values.shape, device=self.model.device, generator=self.generator.get()) * math.sqrt(pars['T']*pars['M'][layer])
				for layer, values in self.model.weights.items()
			}
			weights_ref = self.model.copy(grad=False)

			self._save_data(data, f'{settings["results_dir"]}/data.dat', header=data['move']==0)
			self._save_log(data, pars, momenta, settings, is_ref=True)
			if settings["print_step"] > 0: self._print_status(data)

		# Restart an interrupted simulation from the last saved logfile
		else:
			self.log = torch.load(logfile)
            
			data = self.log['data'].copy()
			t0 = ptime()-data['time']

			pars = torch.load(self.log['files']['pars'])
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
		data = self._correct_types(data, 'data')

		return data, pars, momenta, weights_ref, t0, settings


	def sample(
			self,
			pars: dict,
			settings: dict | None = None,
			logfile: str| None = None,
			keep_going: bool = False,
			is_last: bool = True
	):
		data, pars, momenta, weights_ref, t0, settings = self._setup(pars, settings, logfile, keep_going)

		for move in range(data['move'], pars['tot_moves']):            
			# Perform a step and sample data
			if (move+1)%settings['data_step'] == 0:
				K_i = self._compute_K(momenta, pars)
				weights_i = self.model.copy(grad=False)
				step_dt = ptime()
				momenta = self._step(momenta, pars)
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
				obs = self._compute_observables(x=self.dataset.x, y=self.dataset.y, lamda=pars['lamda'], backward=False, extra=True)
				data = merge_dict(from_dict=obs, into_dict=data)
				self._save_data(data, f'{settings["results_dir"]}/data.dat')

				del weights_i, K_i, step_dt, step_d

			# Just perform a step
			else:
				momenta = self._step(momenta, pars)

			if settings["print_step"] > 0 and (move+1)%settings["print_step"] == 0: self._print_status(data)

			# Update mini-batch noise and related parameters, and then perform a step
			if (move+1)%pars['adj_step'] == 0:
				if (move+1)==pars['tot_moves'] and (not is_last): continue
				pars = self._update_pars(pars=pars, verbose=settings['verbose'])

			if (move+1)%settings["log_step"] == 0: self._save_log(data, pars, momenta, settings)


	def _step(self, momenta, pars):
		old_grad = self._compute_grad(pars)
		old_noise = self._generate_noise()

		with torch.no_grad():
			for layer in self.model.weights:
				self.model.weights[layer] += momenta[layer]*pars['c1']*pars['dt']/pars['M'][layer] - old_grad[layer]*pars['dt']**2./(2.*pars['M'][layer]) + old_noise[layer]*pars['k_wn'][layer]*pars['dt']/pars['M'][layer]

		new_grad = self._compute_grad(pars)
		new_noise = self._generate_noise()

		for layer in self.model.weights:
			momenta[layer] = momenta[layer]*pars['c1']**2. - (new_grad[layer]+old_grad[layer])*pars['c1']*pars['dt']/2. + old_noise[layer]*pars['c1']*pars['k_wn'][layer] + new_noise[layer]*math.sqrt((1.-pars['c1']**2.)*pars['k_mb'][layer]**2. + pars['k_wn'][layer]**2.)

		return momenta


	def _estimate_std(self, pars, verbose):
		sum_grad, sum2_grad = {}, {}
		for iext in range(pars['extractions']):
			grad = self._compute_grad(pars)
			if iext == 0:
				for layer in grad:
					sum_grad[layer] = grad[layer].detach().clone()
					sum2_grad[layer] = (grad[layer]**2.).detach().clone()
			else:
				for layer in grad:
					sum_grad[layer] += grad[layer].detach().clone()
					sum2_grad[layer] += (grad[layer]**2.).detach().clone()
		tot_extractions = pars['extractions']
		"""
		In this version, we are adopting a different standard deviation for each layer of the network.
		An alternative could be to employ different values for each neuron.
		"""
		prev_std = {
			layer: torch.sqrt( (sum2_grad[layer]/(tot_extractions) - (sum_grad[layer]/(tot_extractions))**2.).mean() ).item()
			for layer in sum_grad
		}

		if verbose:
			print(f"Extractions {tot_extractions} (first estimate)")
			for layer in prev_std:
				print(f"- {layer}:\t{prev_std[layer]:.3e}")
			print()

		while tot_extractions < pars['max_extractions']:
			for _ in range(pars['extractions']):
				grad = self._compute_grad(pars)
				for name in grad:
					sum_grad[name] += grad[name].detach().clone()
					sum2_grad[name] += (grad[name]**2.).detach().clone()
			tot_extractions += pars['extractions']
			curr_std = {
				layer: torch.sqrt( (sum2_grad[layer]/(tot_extractions) - (sum_grad[layer]/(tot_extractions))**2.).mean() ).item()
				for layer in sum_grad
			}
			convergence = self._is_close(prev_std, curr_std, pars['threshold_est'])
            
			if verbose:
				print(f"Extractions: {tot_extractions}")
				for layer in prev_std:
					is_close = abs(curr_std[layer]-prev_std[layer])/prev_std[layer] < pars['threshold_est']
					print(f"- {layer}:\t{prev_std[layer]:.3e}\t{curr_std[layer]:.3e}\t{is_close}")
				print()
            
			prev_std = curr_std.copy()
			if convergence:
				break

		if verbose:
			print(f"Reached convergence in {tot_extractions} extractions.")

		return prev_std


	def _update_pars(self, pars, verbose=True):
		print("\n!!! Update of the current mini-batch standard deviations and related parameters !!!\n")
		curr_std = self._estimate_std(pars, verbose)

		if 'std' not in pars.keys():
			pars['streak'] = 1
			print(f"New starting parameters: the streak is {pars['streak']}.")

			pars['c1'] = math.sqrt(1.-pars['m1']**2.)
			pars['M'], pars['T_mb'], pars['k_mb'], pars['k_wn'] = {}, {}, {}, {}
			for layer in self.model.weights:
				pars['M'][layer] = (pars['dt']*curr_std[layer])**2./(4.*pars['T_mb_0']*pars['m1']**2.)
				pars['T_mb'][layer] = pars['T_mb_0']
				pars['k_mb'][layer] = pars['dt']*curr_std[layer]/2.
				pars['k_wn'][layer] = pars['k_mb'][layer] * math.sqrt(pars['T']/pars['T_mb_0'] - 1.)

		else:
			keep_streak = self._is_close(pars['std'], curr_std, pars['threshold_adj'])
			if keep_streak:
				pars['streak'] += 1
				print(f"Compatible current and previous stds: the streak is currently {pars['streak']}.")
			else:
				pars['streak'] = 1
				print(f"Incompatible current and previous stds: the streak is set back to {pars['streak']}.")

			for layer in self.model.weights:
				pars['T_mb'][layer] *= (curr_std[layer]/pars['std'][layer])**2.
				pars['k_mb'][layer] = pars['dt']*curr_std[layer]/2.
				pars['k_wn'][layer] = pars['k_mb'][layer] * math.sqrt(pars['T']/pars['T_mb'][layer] - 1.)

		print(f'Back to the simulation.\n')
		print(f'// {self.name} status register:')
		print(f'{self.separator}\n{self.header}\n{self.separator}')

		pars['adj_step'] = round( math.tanh(pars['streak']/pars['opt_streak']) * pars['max_adj_step']/pars['min_adj_step'] ) * pars['min_adj_step']
		pars['std'] = curr_std.copy()
		return pars


	def _is_close(self, prev_std, curr_std, threshold):
		return all([ abs(curr_std[layer]-prev_std[layer])/prev_std[layer] < threshold for layer in prev_std ])


	def _generate_noise(self):
		noise = {
			layer: torch.randn(values.shape, device=self.model.device, generator=self.generator.get())
			for layer, values in self.model.weights.items()
		}
		return noise


	def _compute_grad(self, pars):
		mb_idxs = torch.zeros((len(self.dataset),), dtype=torch.bool)
		while mb_idxs.sum() < pars['mbs']:
			idx = torch.randint(low=0, high=len(self.dataset), size=(1,), device=self.model.device, generator=self.generator.get())
			mb_idxs[idx] = True
		x, y, _ = self.dataset[mb_idxs]
		_ = self._compute_observables(x, y, pars['lamda'], backward=True, extra=False)
		grad = self.model.copy(grad=True)
		self.model.zero_grad()
		return grad
    
    
	def _compute_observables(self, x, y, lamda, backward=True, extra=False):
		fx = self.model(x)
		cost = self.Cost(fx, y)
		mod2 = compute_mod2(self.model.weights)
		loss = cost + (lamda/2.)*mod2

		if backward:
			loss.backward()

		if extra: ### accuracy computed only on the full batch
			metric = self.Metric(fx, self.dataset.y)
			return {'loss':loss, 'cost':cost, 'mod2':mod2, 'metric':metric}
		else:
			return {'loss':loss, 'cost':cost, 'mod2':mod2}


	def _compute_K(self, momenta, pars):
		K = 0.
		for layer in momenta:
			K += (0.5*(momenta[layer]**2.)/pars["M"][layer]).sum()
		return K.item()


	def _save_data(self, dikt, filename, header=False):
		if header:
			with open(filename, 'w') as f:
				header, line = '', ''
				for key in dikt:
					header = header + f'{key}\t'
					line = line + f'{dikt[key]}\t'
				print(header[:-1], file=f)
				print(line[:-1], file=f)
		else:
			with open(filename, 'a') as f:
				line = ''
				for key in dikt: line = line + f'{dikt[key]}\t'
				print(line[:-1], file=f)


	def _save_log(self, data, pars, momenta, settings, is_ref=False):
		torch.save(pars, f'{settings["weights_dir"]}/pars_{data["move"]}.pt')
		torch.save(momenta, f'{settings["weights_dir"]}/momenta_{data["move"]}.pt')
		self.model.save(f'{settings["weights_dir"]}/weights_{data["move"]}.pt')
		self.generator.save(f'{settings["results_dir"]}/generator.npy')

		self.log['data'] = data.copy()
		self.log['files']['pars'] = f'{settings["weights_dir"]}/pars_{data["move"]}.pt'
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
			types_and_keys = [(int, ['moves', 'tot_moves', 'mbs', 'max_extractions', 'extractions', 'max_adj_step', 'min_adj_step', 'opt_streak'])]
		else:
			types_and_keys = [(int, ['move'])]
        
		for key in d:
			for _type, keys in types_and_keys:
				if key in keys:
					d[key] = _type(d[key])

		return d


	def _init_attributes(self):
		self.log = {
            'data': None,
            'files': {
                'pars': None,
                'momenta': None,
                'weights': None,
                'generator': None,
				'weights_ref': None,
            },
        }

		self.settings = {
			"results_dir": ".",
			"weights_dir": "./weights",
			"data_step": 10,
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
