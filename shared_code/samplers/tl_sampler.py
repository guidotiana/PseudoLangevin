import os
import torch
import numpy as np
from time import process_time as ptime

from utils.operations import merge_dict, merge_dicts, compute_q, compute_d, compute_mod2, wsum, kprod

# Default precision
torch.set_default_dtype(torch.float64)



### --------------------- ###
### True-Langevin Sampler ###
### --------------------- ###
class TLSampler():

	def __init__(self, model, Cost, Metric, dataset, generator, name:str='TLSampler'):
		self.model = model
		self.Cost = Cost
		self.Metric = Metric
		self.dataset = dataset
		self.generator = generator
		self.name = name

		self._init_attributes()


	def _setup(self, pars, settings, logfile, keep_going):

		# Complete parameters
		pars['c1'] = np.sqrt(1.-pars['m1']**2.)
		pars['c2'] = np.sqrt((pars['m1']**2.)*pars['m']*pars['T'])

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

			momenta = {
				layer: torch.randn(values.shape, device=self.model.device, generator=self.generator.get()) * np.sqrt(pars['T']*pars['m'])
				for layer, values in self.model.weights.items()
			}
			weights_ref = self.model.copy(grad=False)

			self._save_data(data, f'{settings["results_dir"]}/data.dat', header=True)
			self._save_log(data, pars, momenta, settings, is_ref=True)

		# Restart an interrupted simulation from the last saved logfile
		else:
			self.log = torch.load(logfile)
            
			data = self.log['data'].copy()
			t0 = ptime()-data['time']

			pars = torch.load(self.log['files']['pars'], weights_only=True)
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
			if (move+1)%settings["log_step"] == 0: self._save_log(data, pars, momenta, settings)


	def _step(self, momenta, pars):
		old_grad = self._compute_grad(pars)
		old_noise = self._generate_noise()

		with torch.no_grad():
			for layer in self.model.weights:
				self.model.weights[layer] += momenta[layer]*pars['c1']*pars['dt']/pars['m'] - old_grad[layer]*pars['dt']**2./(2.*pars['m']) + old_noise[layer]*pars['c2']*pars['dt']/pars['m']

		new_grad = self._compute_grad(pars)
		new_noise = self._generate_noise()

		for layer in self.model.weights:
			momenta[layer] = momenta[layer]*pars['c1']**2. - (new_grad[layer]+old_grad[layer])*pars['c1']*pars['dt']/2. + old_noise[layer]*pars['c1']*pars['c2'] + new_noise[layer]*pars['c2']

		return momenta


	def _generate_noise(self):
		noise = {
			layer: torch.randn(values.shape, device=self.model.device, generator=self.generator.get())
			for layer, values in self.model.weights.items()
		}
		return noise


	def _compute_grad(self, pars):
		P = len(self.dataset)
		N_mb = round(P/pars['mbs']) if P%pars['mbs']==0 else round(P/pars['mbs'])+1
		for i in range(N_mb):
			x, y = self.dataset.x[i*pars['mbs']:(i+1)*pars['mbs']], self.dataset.y[i*pars['mbs']:(i+1)*pars['mbs']]
			_ = self._compute_observables(
				x=x, y=y, 
				lamda=pars['lamda'], 
				backward=True, extra=False
			)
			if i==0:
				grad = kprod(self.model.copy(grad=True), len(y)/P)
			else:
				grad = wsum(
					grad,
					kprod(self.model.copy(grad=True), len(y)/P),
				)
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
			K += (0.5*(momenta[layer]**2.)/pars["m"]).sum()
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

		self.log = {
			'data': data.copy(),
			'files': {
				'pars': f'{settings["weights_dir"]}/pars_{data["move"]}.pt',
				'momenta': f'{settings["weights_dir"]}/momenta_{data["move"]}.pt',
				'weights': f'{settings["weights_dir"]}/weights_{data["move"]}.pt',
				'generator': f'{settings["results_dir"]}/generator.npy',
			},
		}
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
			types_and_keys = [(int, ['moves', 'tot_moves', 'mbs'])]
		else:
			types_and_keys = [(int, ['move'])]
        
		for key in d:
			for _type, keys in types_and_keys:
				if key in keys:
					d[key] = _type(d[key])

		return d


	def _init_attributes(self):
		self.log = {}

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
