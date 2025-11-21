import os

from io import StringIO
from time import process_time as ptime
from collections.abc import Callable

import math
import torch

from models.nnmodel import NNModel
from generator.custom_generator import CustomGenerator
from utils.general import create_path
from utils.operations import wcopy, compute_q, compute_d, compute_d2, compute_mod2, is_subset, merge_dict, roundup, estimate_variance



### --------------------------------------------------- ###
### Double-Ratchet Double-Noise Pseudo-Langevin Sampler ###
### --------------------------------------------------- ###
class DRPLSampler():

	def __init__(
			self,
			models: dict[str, NNModel],
			dataset: torch.utils.data.Dataset,
			dataset_val: torch.utils.data.Dataset,
			Cost: Callable,
			Metric: Callable,
			name: str = 'PLSampler'
	):
		assert tuple(models.keys()) == ('fe', 'se'), f'{self.name}.__init__(): invalid "models" dictionary keys. Expected keys: "fe", "se".'
		assert tuple(models['fe'].weights.keys()) == tuple(models['se'].weights.keys()), f'{self.name}.__init__(): mismatch between models["fe"] and models["se"] weights keys.'
		self.models = models
		self.dataset = dataset
		self.dataset_val = dataset_val
		self.Cost = Cost
		self.Metric = Metric
		self.name = name
		self._init_attributes()
		self.mod2 = {}
		with torch.no_grad():
			for which_end in ('fe', 'se'):
				self.mod2[which_end] = compute_mod2(self.models[which_end].weights)

	def sample(
			self,
			pars: dict,
			settings: dict,
			start_fn_fe: str | None = None,
			start_fn_se: str | None = None,
	):
		pars_list, settings, data, varpars, momenta = self._setup(pars, settings, start_fn_fe, start_fn_se)
		del pars
		
		for idx, pars_idx in enumerate(pars_list):
			if idx > 0:
				data, varpars, momenta = self._start(pars_idx, settings, idx)
			del pars_idx

			data = self._correct_types(data, "data")
			varpars = self._correct_types(varpars, "varpars")
			for move in range(data["move"]+1, varpars["tot_moves"]):
				
#				for which_end in ('fe', 'se'):
#					print(f"\n!!! Model {which_end} weights before step !!!\n")
#					for layer in self.models[which_end].weights:
#						print(f'{layer} : {self.models[which_end].weights[layer]}')

				if move%settings["data_step"] == 0:
					data, momenta = self._step_and_sample(momenta, varpars, move, data)
				else:
					for which_end in ('fe', 'se'):
						data, momenta = self._step(momenta, which_end, varpars, data)
					data['move'] += 1
				if data['fe_cost'] > varpars['loss_eq'] and data['se_cost'] > varpars['loss_eq'] and data['eq']==False:
					print(data['fe_cost'])
					print(data['se_cost'])
					data['eq'] = True
					data['ratchet'] = torch.tensor([0.], device=self.models['fe'].device).item()
					data['dmin'] =  compute_d(self.models['fe'].weights, self.models['se'].weights).item()

				if move%settings["print_step"] == 0:

					self._print_status(data)
				if move%varpars['fe_adj_step'] == 0 or move%varpars['se_adj_step'] == 0:
					varpars, momenta = self._update_varpars(varpars=varpars, ratchet=0., eq=data['eq'], momenta=momenta, verbose=settings['verbose'])
				if move%settings["log_step"]==0:
					self._save_log(data, varpars, momenta, settings)
				
				
#				for which_end in ('fe', 'se'):
#					print(f"\n!!! Model {which_end} weights after step !!!\n")
#					for layer in self.models[which_end].weights:
#						print(f'{layer} : {self.models[which_end].weights[layer]}')


			if idx+1 < len(pars_list):
				for which_end in ('fe', 'se'):
					data, momenta = self._step(momenta, which_end, varpars, data)
			else:
				data, momenta = self._step_and_sample(momenta, varpars, varpars["tot_moves"], data)
				varpars, momenta = self._update_varpars(varpars=varpars, ratchet=0., eq=data['eq'], momenta=momenta, verbose=settings['verbose'])
				self._save_log(data, varpars, momenta, settings)
				self._print_status(data)

		del self.log, self.generator, self.weights_ref, self.t0



	def _setup(
			self,
			pars: dict,
			settings: dict,
			start_fn_fe: str | None = None,
			start_fn_se: str | None = None,
	):
		# 1. PARS
		# First the inputted pars dictionary is checked, verifying the all the necessary keys are present.
		# Then, the pars dictionary is completed, adding missing keys and checking the type for the inputted ones.
		# Then, the pars dictionary is splitted up in a list of dictionaries, where each instance of parameters must be executed 
		# when the previous one has been completed. Finally, the range of the values are checked for each parameters instance.
		assert is_subset(pars.keys(), self.defpars.keys()), f"{self.name}._setup(): unexpected key in inputted pars dictionary. Expected keys are: {list(self.defpars.keys())}."
		for key, (value, typ) in self.defpars.items():
			if value is None:
				assert key in pars.keys(), f"{self.name}._setup(): necessary key '{key}' missing from the inputted pars dictionary."
			else:
				if key not in pars:
					pars[key] = value
				else:
					if isinstance(pars[key], list):
						try:
							pars[key] = [typ(el) for el in pars[key]]
						except ValueError:
							raise ValueError(f"{self.name}.setup(): pars '{key}' type should be {typ}, but found {type(pars[key][0])}.")
					else:
						try:
							pars[key] = typ(pars[key])
						except ValueError:
							raise ValueError(f"{self.name}.setup(): pars '{key}' type should be {typ}, but found {type(pars[key])}.")

		list_lengths = [len(v) for k,v in pars.items() if isinstance(v, list)]
		if len(list_lengths) > 0:
			assert len(set(list_lengths)) == 1, f"{self.name}.setup(): list keys in the inputted pars dictionary with different lengths ({list_lengths}). All list keys must have the same length."
			for key, value in pars.items():
				if not isinstance(value, list):
					pars[key] = [value]*list_lengths[0]

			tot_moves = 0
			pars_list = []
			for idx, (stime, dt) in enumerate(zip(pars['stime'], pars['dt'])):
				moves = int(stime/dt)
				tot_moves += moves
				pars_idx = {'moves': moves, 'tot_moves': tot_moves}
				for key, value in pars.items():
					pars_idx[key] = value[idx]
				pars_list.append(pars_idx)

		else:
			pars["moves"] = int(pars["stime"]/pars["dt"])
			pars['tot_moves'] = pars['moves']
			pars_list = [pars]

		for idx in range(len(pars_list)):
			assert all([0.<=v<=1. for k,v in pars_list[idx].items() if k in ["p_reset"]]), (
				f'{self.name}._setup(): invalid value for one of the following keys ("p_reset") at index {idx}. Allowed values: 0<=v<=1.'
			)
			assert all([0.<v<1. for k,v in pars_list[idx].items() if k in ["T_ratio_i", "T_ratio_f", "T_ratio_max", "m1"]]), (
			    f'{self.name}._setup(): invalid value for one of the following keys ("T_ratio_i", "T_ratio_f", "T_ratio_max", "m1") at index {idx}. Allowed values: 0<v<1.'
			)
			assert all([v>=0. for k,v in pars_list[idx].items() if k in ["max_resets", "gamma", "lamda", "threshold_est", "threshold_adj"]]), (
			    f'{self.name}._setup(): invalid value for one of the following keys ("max_resets", "gamma", "lamda", "threshold_est", "threshold_adj") at index {idx}. Allowed values: v>=0.'
			)
			assert all([v>0. for k,v in pars_list[idx].items() if k in ["stime", "moves", "tot_moves", "T", "dt", "max_extractions", "min_extractions", "max_adj_step", "min_adj_step"]]), (
			    f'{self.name}._setup(): invalid value for one of the following keys ',
			    f'("stime", "moves", "tot_moves", "T", "dt", "max_extractions", "min_extractions", "max_adj_step", "min_adj_step") at index {idx}. Allowed values: v>0.'
			)
			assert all([v in [0,1] for k,v in pars_list[idx].items() if k in ["adj_ref", "mean"]]), (
			    f'{self.name}._setup(): invalid value for one of the following keys ("adj_ref", "mean") at index {idx}. Allowed values: v==0 or v==1.'
			)
			pars_list[idx]["max_extractions"] = roundup(pars_list[idx]["max_extractions"], pars_list[idx]["min_extractions"])
			pars_list[idx]["min_adj_step"] = roundup(pars_list[idx]["min_adj_step"], self.stepscale)
			pars_list[idx]["max_adj_step"] = roundup(pars_list[idx]["max_adj_step"], pars_list[idx]["min_adj_step"])
			if idx == 0:
				pars_list[idx]["adj_ref"] = True

		# 2. SETTINGS
		# First inputted settings are controlled and completed. Then, data, log and print steps are rounded to be divisible.
		# Finally threads and devices are set. Once the device is defined, the model, the data and the generator are loaded on the correct device.
		assert is_subset(settings.keys(), self.defsettings.keys()), f"{self.name}._setup(): unexpected key in inputted settings dictionary. Expected keys are: {list(self.defsettings.keys())}."
		for key, (value, typ) in self.defsettings.items():
			if key not in settings:
				settings[key] = value
				if key in ["results_dir", "weights_dir"]:
					create_path(value)
			else:
				try:
					settings[key] = typ(settings[key])
				except ValueError:
					raise ValueError(f"{self.name}.setup(): settings '{key}' type should be {typ}, but found {type(settings[key])}.")
		assert settings["data_step"] > 0 and settings["log_step"] > 0 and settings["print_step"] > 0, (
			f"{self.name}._setup(): 'steps' keys in settings must all be positive, "
			f"but found {settings['data_step']} ('data'), {settings['log_step']} ('log') and {settings['print_step']} ('print')."
		)
	#	settings["data_step"] = roundup(settings["data_step"], self.stepscale)
		settings["log_step"] = roundup(settings["log_step"], self.stepscale)

		assert settings["num_threads"] <= 5, f"{self.name}.setup(): invalid value for 'num_threads' variable {settings['num_threads']}. Allowed values: num_threads <= 5."
		torch.set_num_threads(settings["num_threads"])

		settings["device"] = torch.device(settings["device"]) if isinstance(settings["device"], str) else settings["device"]
		if ("cuda" in settings["device"].type) and (not torch.cuda.is_available()):
			settings["device"] = torch.device("cpu")

		self.models['fe'].to(settings["device"])
		self.models['se'].to(settings["device"])
		self._weights_old = {}
		with torch.no_grad():
			for which_end in ('fe', 'se'):
				self._weights_old[which_end] = {k: torch.empty_like(v) for k, v in self.models[which_end].weights.items()}
				for k, v in self.models[which_end].weights.items():
					self._weights_old[which_end][k].copy_(v)
		self.dataset.to(settings["device"])
		self.dataset_val.to(settings["device"])
		self.generator = CustomGenerator(
				seed=pars_list[0]["seed"],
				device=settings["device"],
		)
		# 3. CLEAN
		# If restart is True (and everything required to restart a previous simulation exists), load the log information.
		# Otherwise, reset everything and proceed to clean every file (except for pars.txt) in results and weights directories.
		# The last three temporary attributes are here initiated, (log, weights_ref and t0).
		if settings["restart"]:
			nec_rfiles = ['data.dat', 'pars.txt', 'generator.npy', 'log.pt']
			nec_wfiles = [f"{name}_0.pt" for name in ('weights_fe', 'weights_se', 'momenta_fe', 'momenta_se', 'varpars')]
			settings["restart"] *= all([rf in os.listdir(settings['results_dir']) for rf in nec_rfiles] + [wf in os.listdir(settings['weights_dir']) for wf in nec_wfiles])

		if settings["restart"]:
			self.log = torch.load(f'{settings["results_dir"]}/log.pt')
			print(self.log)
			data = self.log["data"].copy()
			self.t0 = ptime()-data["time"]
			pars_list = [
					pars_idx for pars_idx in pars_list if pars_idx["tot_moves"]>data["move"]
			]
			
			momenta = {}
			self.weights_ref = {}
			self._weights_old = {}
			for which_end in self.log['files']['weights'].keys():
				self._weights_old[which_end] = {k: torch.empty_like(v) for k, v in self.models[which_end].weights.items()}
				self.models[which_end].load(self.log['files']['weights'][which_end] )
				self.weights_ref[which_end] = torch.load(self.log['files']['weights_ref'][which_end], map_location=self.models[which_end].device, weights_only=True)
				momenta[which_end] = torch.load(self.log['files']['momenta'][which_end], map_location =settings['device'], weights_only=True)
				with torch.no_grad():
					for k, v in self.models[which_end].weights.items():
						self._weights_old[which_end][k].copy_(v)

			self.generator.load(self.log['files']['generator'])
			varpars = torch.load(self.log["files"]["varpars"])

			self._print_pars(pars_list[0], settings, 0)
			self._print_status(data, header=True)

		else:

			for d in [settings['results_dir'], settings['weights_dir']]:
				for fn in os.listdir(d):
					if fn == 'pars.txt': continue
					if os.path.isfile(f'{d}/{fn}'):
						os.remove(f'{d}/{fn}')

			self.t0 = ptime()
			if start_fn_fe is not None and start_fn_se is not None:
				start_fn = {
					"fe": start_fn_fe,
					"se": start_fn_se,
				}
			#	self.models = {}
				for which_end in ('fe', 'se'):
					self.models[which_end].load(start_fn[which_end])
			data, varpars, momenta = self._start(pars_list[0], settings, 0)


		return (
			pars_list,
			settings,
			data,
			varpars,
			momenta,
		)



	def _start(self, pars_idx, settings, idx):
		self._print_pars(pars_idx, settings, idx)
		
		d2 = {}
		q ={}
		if pars_idx["adj_ref"]:
			self.weights_ref  ={}
			q_models = compute_q(self.models['fe'].weights, self.models['se'].weights).item()
			d_models = compute_d(self.models['fe'].weights, self.models['se'].weights).item()
			for which_end in ('fe', 'se'):
				self.weights_ref[which_end] = self.models[which_end].copy(grad=False)
				q[which_end] = 1. 
				d2[which_end] =  0.

		else:
			q_models = compute_q(self.models['fe'].weights, self.models['se'].weights).item()
			d_models = compute_d(self.models['fe'].weights, self.models['se'].weights).item()
			for which_end in ('fe', 'se'):
				q[which_end] = compute_q(self.models[which_end].weights, self.weights_ref[which_end]).item()
				d2[which_end] = compute_d2(self.models[which_end].weights, self.weights_ref[which_end]).item()
		varpars, _ = self._update_varpars(varpars=pars_idx, ratchet=0., eq=False, momenta=None, verbose=settings["verbose"])
		momenta = {}
		for which_end in ('fe', 'se'):
			momenta[which_end] = {
					layer: torch.randn(values.shape, device=settings["device"], generator=self.generator.get()) * torch.sqrt(varpars['T']*varpars[f'{which_end}_M'][layer])
					for layer, values in self.models[which_end].weights.items()
			}
			# vincolo 
			with torch.no_grad():
				for layer in self.models[which_end].weights:
					momenta[which_end][layer] -=  (momenta[which_end][layer]*self.models[which_end].weights[layer]/varpars[f'{which_end}_M'][layer]).sum()*self.models[which_end].weights[layer]*varpars[f'{which_end}_M'][layer]/ compute_mod2(self.models[which_end].weights)
		#
		data = {
			'move': varpars['tot_moves']-varpars['moves'],
			'time': ptime()-self.t0,
			'q': q_models,
			'fe_q': q['fe'],
			'se_q': q['se'],
			'd': d_models,
			'fe_d2': d2['fe'],
			'se_d2': d2['se'],
			'fe_step_d': 0.,
			'se_step_d': 0.,
			'fe_step_dt': 0.,
			'se_step_dt': 0.,
			'fe_eff_v': 0.,
			'se_eff_v': 0.,
			'fe_K': self._compute_K(momenta['fe'], 'fe', varpars),
			'se_K': self._compute_K(momenta['se'], 'se', varpars),
			'fe_dK': 0.,
			'se_dK': 0.,
			'ratchet': torch.tensor([0.], device=self.models['fe'].device).item(),
			'dmin': d_models,
			'eq': False,
		}


		for which_end in ('fe', 'se'):
			obs = self._compute_observables(which_end=which_end, x=self.dataset.x, y=self.dataset.y, lamda=varpars['lamda'], k=varpars['k'], gamma=varpars['gamma'], ratchet=data['ratchet'], eq=data['eq'], backward=False)
			for key in obs:
				data[f'{which_end}_{key}'] = obs[key]#.item()
		#data = merge_dict(from_dict=obs, into_dict=data)
		self._extend_buffer(data, header=data['move']==0)
		self._save_log(data, varpars, momenta, settings, is_ref=varpars["adj_ref"])
		self._print_status(data)
		return data, varpars, momenta



	def _step(self, momenta, which_end, varpars, data):
	#	print(data)
	#	print(data['ratchet'])

	#	ratchet = torch.tensor(data['ratchet'], device=self.models[which_end].device, dtype=torch.float32)
		old_grad = self._compute_grad(varpars, which_end, data['ratchet'], data['eq'])
		old_noise = self._generate_noise()
		
		with torch.no_grad():

			for k, v in self.models[which_end].weights.items():
				self._weights_old[which_end][k].copy_(v)
			for layer in self.models[which_end].weights:
				vvd = momenta[which_end][layer]*varpars[f'{which_end}_c1']*varpars['dt']/varpars[f'{which_end}_M'][layer] - old_grad[layer]*varpars['dt']**2./(2.*varpars[f'{which_end}_M'][layer])
				bpn = old_noise[layer]*varpars[f'{which_end}_k_wn'][layer]*varpars['dt']/varpars[f'{which_end}_M'][layer]
				self.models[which_end].weights[layer] += vvd + bpn
		
		S_1 = 0
		S_2 = 0
		S_3 = 0
		with torch.no_grad():
			for layer in self.models[which_end].weights:
				S_1 += (self.models[which_end].weights[layer]*self._weights_old[which_end][layer]/varpars[f'{which_end}_M'][layer]).sum()
				S_2 += (self.models[which_end].weights[layer]**2).sum()
				S_3 += ((self._weights_old[which_end][layer]/varpars[f'{which_end}_M'][layer])**2).sum()
			delta = S_1**2 + self.mod2[which_end]*S_3 - S_3*S_2
			lamda_1 = (-S_1 +math.sqrt(delta)) / (varpars['dt']**2 * S_3)
			lamda_2 = (-S_1 - math.sqrt(delta)) / (varpars['dt']**2 * S_3)
			lamda = 0
			if abs(lamda_1) < abs(lamda_2):
				lamda = lamda_1
			else:
				lamda = lamda_2

			for layer in self.models[which_end].weights:
				self.models[which_end].weights[layer] += varpars['dt']**2/varpars[f'{which_end}_M'][layer]*lamda*self._weights_old[which_end][layer]
		
		d = compute_d(self.models['fe'].weights, self.models['se'].weights)
		if d.item() < data['dmin']: data['dmin'] = d.item()
		data['d'] = d
#		print( 'd ', d)
#		print('dmin ' , data['dmin'])
		data['ratchet'] = torch.max(data['d']-data['dmin'], torch.tensor([0.], device=self.models['fe'].device))**2.
		data[f'{which_end}_d2'] = compute_d2(self.models[which_end].weights, self.weights_ref[which_end]).item()
		
		new_grad = self._compute_grad(varpars, which_end, data['ratchet'], data['eq'])
		new_noise = self._generate_noise()
		with torch.no_grad():
			for layer in self.models[which_end].weights:
				vvd = momenta[which_end][layer]*(varpars[f'{which_end}_c1']**2.-1.) - (new_grad[layer]+old_grad[layer] - 2*lamda*self._weights_old[which_end][layer])*varpars[f'{which_end}_c1']*varpars['dt']/2.
				bpn = old_noise[layer]*varpars[f'{which_end}_c1']*varpars[f'{which_end}_k_wn'][layer] + new_noise[layer]*torch.sqrt(varpars[f'{which_end}_var'][layer]*(varpars['dt']*varpars['m1']/2. )**2. + varpars[f'{which_end}_k_wn'][layer]**2.)
				momenta[which_end][layer] += vvd + bpn
			
		S_4 = 0
		S_5 = 0
		with torch.no_grad():
			for layer in self.models[which_end].weights:
				S_4 += (self.models[which_end].weights[layer]*momenta[which_end][layer]/varpars[f'{which_end}_M'][layer]).sum()
				S_5 += (self.models[which_end].weights[layer]**2/varpars[f'{which_end}_M'][layer]).sum()
			Lamda = - S_4 / (S_5*varpars['dt']*varpars[f'{which_end}_c1'])

			for layer in self.models[which_end].weights:
				momenta[which_end][layer] -= varpars[f'{which_end}_c1']*varpars['dt']*self.models[which_end].weights[layer]*Lamda
		return data, momenta



	def _step_and_sample(self, momenta, varpars, move, data):
		for which_end in ('fe', 'se'):
			K_i = self._compute_K(momenta[which_end], which_end, varpars)
			weights_i = self.models[which_end].copy(grad=False)
			step_dt = ptime()
			data, momenta = self._step(momenta, which_end, varpars, data)
			step_dt = ptime()-step_dt
			step_d = compute_d(self.models[which_end].weights, weights_i).item()
			data[f'{which_end}_step_d'] =  step_d
			data[f'{which_end}_step_dt'] = step_dt
			data[f'{which_end}_eff_v'] = step_d/step_dt
			data[f'{which_end}_dK'] = self._compute_K(momenta[which_end], which_end, varpars) - K_i
			data[f'{which_end}_q'] = compute_q(self.models[which_end].weights, self.weights_ref[which_end])
			obs = self._compute_observables(which_end=which_end, x=self.dataset.x, y=self.dataset.y, lamda=varpars['lamda'], gamma=varpars['gamma'], k=varpars['k'], ratchet=data['ratchet'], eq=data['eq'], backward=False)
			for key in obs:
				data[f'{which_end}_{key}'] = obs[key]#.item()
		data['move'] += 1 
		data['time'] = ptime()-self.t0
		data['q'] = compute_q(self.models['fe'].weights, self.models['se'].weights)
		self._extend_buffer(data)

		return data, momenta



	def _compute_observables(self, which_end, x, y, lamda, gamma, k, ratchet, eq, backward=True):

		fx = self.models[which_end](x)
		cost = self.Cost(fx, y)
		mod2 = compute_mod2(self.models[which_end].weights)
		d2 = compute_d2(self.models[which_end].weights, self.weights_ref[which_end])
		loss = cost + (lamda/2.)*mod2 + (gamma/2.)*d2
		if eq == False:
			k = 0.
		if backward:
			U = loss + (k/2.)*ratchet
			U.backward(retain_graph=True)

		else: # observables computed only on the full batch (mainly during _step_and_sample())
			metric = self.Metric(fx, self.dataset.y)
			fx_val = self.models[which_end](self.dataset_val.x)
			metric_val = self.Metric(fx_val, self.dataset_val.y)
		return {
			'loss':loss.detach().item(),
			'cost':cost.detach().item(),
			'mod2':mod2.detach().item(),
			'd': math.sqrt(d2.detach().item()),
			'metric':metric.detach().item() if not backward else None,
			'metric_val':metric_val.detach().item() if not backward else None,
		}

	def _compute_K(self, momenta, which_end, varpars):
		K = 0.
		for layer in momenta:
			K += (0.5*(momenta[layer]**2.)/varpars[f'{which_end}_M'][layer]).sum()
		return K.item()

	def _compute_grad(self, varpars, which_end, ratchet, eq):
		mb_mask = torch.zeros((len(self.dataset),), dtype=torch.bool)
		mb_idxs = torch.randperm(len(self.dataset), device=self.models['fe'].device, generator=self.generator.get())[:varpars['mbs']]
		mb_mask[mb_idxs] = True
		x, y, _ = self.dataset[mb_mask]
		
		_ = self._compute_observables(which_end=which_end, x=x, y=y, lamda=varpars['lamda'], gamma=varpars['gamma'], k=varpars['k'], ratchet=ratchet, eq=eq, backward=True)
		grad = self.models[which_end].copy(grad=True)
		self.models[which_end].zero_grad()
		return grad

	def _generate_noise(self):
		return {
			layer: torch.randn(values.shape, device=self.models['fe'].device, generator=self.generator.get())
			for layer, values in self.models['fe'].weights.items()
		}



	def _estimate_var(self, varpars, which_end, ratchet, eq, verbose):
		# 1. Start by first computing the mini-batch-induced variances for each weight of the network
		sum_grad, sum2_grad = {}, {}
		for iext in range(varpars['min_extractions']):
			grad = self._compute_grad(varpars, which_end, ratchet, eq)
			if iext == 0:
				for layer in grad:
					sum_grad[layer] = grad[layer].detach()
					sum2_grad[layer] = (grad[layer]**2.).detach()
			else:
				for layer in grad:
					sum_grad[layer] += grad[layer].detach()
					sum2_grad[layer] += (grad[layer]**2.).detach()
		tot_extractions = varpars['min_extractions']
		curr_var = {layer: estimate_variance(sum2_grad[layer], sum_grad[layer], tot_extractions, mean=varpars["mean"], axis=varpars["axis"]) for layer in sum_grad}
		if verbose:
			print(f"First estimate at {tot_extractions} extractions terminated.")
			for layer in curr_var:
				print(f"- {layer}:\t{curr_var[layer]}")
			print()

		# 2. Refine the current estimate up until the variance converges for each weight (or the maximum number of mini-batch extractions is reached)
		while tot_extractions < varpars['max_extractions']:
			for _ in range(varpars['min_extractions']):
				grad = self._compute_grad(varpars, which_end, ratchet, eq )
				for name in grad:
					sum_grad[name] += grad[name].detach()
					sum2_grad[name] += (grad[name]**2.).detach()
			tot_extractions += varpars['min_extractions']

			converged = []
			next_var = {}
			for layer, curr_var_l in curr_var.items():
				next_var[layer] = estimate_variance(sum2_grad[layer], sum_grad[layer], tot_extractions, mean=varpars["mean"], axis=varpars["axis"])
				converged.append( 
						torch.all( abs(next_var[layer]-curr_var_l)/curr_var_l < varpars['threshold_est'] ).item()
				)
			if verbose:
				print(f"Further estimate at {tot_extractions} extractions. Converged: {all(converged)} ({sum(converged)}/{len(converged)}).")
				for layer in next_var:
					print(f"- {layer}:\t{next_var[layer]}")
				print()
			curr_var = wcopy(next_var)
			del next_var
			if all(converged):
				break

		return curr_var, tot_extractions

	def _update_varpars(self, varpars, ratchet, eq, momenta=None, verbose=True):
		print("\n!!! Update of the current mini-batch standard deviations and related parameters !!!\n")
		
#		print("\n! Print varpars value !\n")
#		for par, value in varpars.items():
#			print(par, value)
		
		for which_end in ('fe', 'se'):

#			print(f"\n!!! Model {which_end} weights !!!\n")
#			for layer in self.models[which_end].weights:
#				print(f'{layer} : {self.models[which_end].weights[layer]}')
			
			curr_var, tot_extractions = self._estimate_var(varpars, which_end, ratchet, eq, verbose)
			# 0. Initiate mini-batch temperatures
			if f'{which_end}_var' not in varpars.keys():
				varpars[f'{which_end}_c1'] = math.sqrt(1.-varpars['m1']**2.)
				varpars[f'{which_end}_M'], varpars[f'{which_end}_T_ratio'], varpars[f'{which_end}_k_wn'] = {}, {}, {}
				for layer, curr_var_l in curr_var.items():
					varpars[f'{which_end}_T_ratio'][layer] = torch.full_like(curr_var_l, varpars['T_ratio_i'])
					varpars[f'{which_end}_M'][layer] = curr_var_l*varpars['dt']**2./(4.*varpars['T_ratio_i']*varpars['T']*varpars['m1']**2.)
					varpars[f'{which_end}_k_wn'][layer] = torch.sqrt( varpars[f'{which_end}_M'][layer]*varpars['T']*varpars['m1']**2. - curr_var_l*(varpars['dt']/2.)**2. )

				varpars[f'{which_end}_streak'] = 1
				print(f"\nNew set of starting parameters: the streak is {varpars[f'{which_end}_streak']}.")
			else:
				idxs = torch.randperm(len(curr_var), device=self.models[which_end].device, generator=self.generator.get())
				perm_layers = [list(curr_var.keys())[idx.item()] for idx in idxs]
				resets = torch.rand((len(curr_var),), device=self.models[which_end].device, generator=self.generator.get()) < varpars["p_reset"]

				keep_streak = True
				counter = 0
				for perm_layer, reset in zip(perm_layers, resets):
					curr_var_pl = curr_var[perm_layer]
					T_ratio_updated = varpars[f'{which_end}_T_ratio'][perm_layer] * curr_var_pl/varpars[f'{which_end}_var'][perm_layer]

					# 1.1 Reset mini-batch temperature to final value
					if reset and (counter < varpars['max_resets']):
						varpars[f'{which_end}_T_ratio'][perm_layer] = torch.full_like(curr_var_pl, varpars[f'{which_end}_T_ratio_f'])
						varpars[f'{which_end}_M'][perm_layer] *= T_ratio_updated/varpars['T_ratio_f']
						momenta[perm_layer] *= torch.sqrt(T_ratio_updated/varpars['T_ratio_f'])
						counter += 1
						print(f"RESET: T_ratios on layer {perm_layer} have been reset to goal (final) value {varpars['T_ratio_f']} (counter={counter}).")

					# 1.2 Standard (controlled) update of the mini-batch temperature
					else:
						mask = T_ratio_updated > varpars['T_ratio_max']
						varpars[f'{which_end}_T_ratio'][perm_layer][~mask] = T_ratio_updated
						if mask.sum().item() > 0:
							varpars[f'{which_end}_T_ratio'][perm_layer][mask] = varpars['T_ratio_max']
							varpars[f'{which_end}_M'][perm_layer][mask] *= T_ratio_updated[mask]/varpars['T_ratio_max']
							repated_shape = tuple( [1]*varpars["axis"] + list(momenta[which_end][perm_layer].shape[varpars["axis"]:]) )
							momenta[which_end][perm_layer][ mask.repeat(repated_shape) ] *= torch.sqrt(T_ratio_updated[mask]/varpars['T_ratio_max'])
							print(f"ALERT: T_ratios on layer {perm_layer} have reached the threshold value {varpars['T_ratio_max']}, which is highly unstable. Starting the update of the mass matrix M!")

					varpars[f'{which_end}_k_wn'][perm_layer] = torch.sqrt( varpars[f'{which_end}_M'][perm_layer]*varpars['T']*varpars['m1']**2. - curr_var_pl*(varpars['dt']/2.)**2. )
					# 2. Check streak
					keep_streak *= torch.all( abs( torch.sqrt(curr_var_pl/varpars[f'{which_end}_var'][perm_layer]) - 1. ) < varpars['threshold_adj'] ).item()
				if keep_streak:
					varpars[f'{which_end}_streak'] += 1
					print(f"\nCompatible current and previous variances: the streak is increased to {varpars[f'{which_end}_streak']}.")
				else:
					varpars[f'{which_end}_streak'] = 1
					print(f"\nIncompatible current and previous variances: the streak is set back to {varpars[f'{which_end}_streak']}.")

			varpars[f'{which_end}_adj_step'] = min([varpars[f'{which_end}_streak']*varpars['min_adj_step'], varpars['max_adj_step']])
			print(f"The current adjournment step is {varpars[f'{which_end}_adj_step']}.")

			varpars[f'{which_end}_var'], varpars[f'{which_end}_tot_extractions'] = curr_var, tot_extractions
			
			for par, value in varpars.items():
				print(par, value)

		print(f'Back to the simulation.\n')
		print(f'// {self.name} status register:')
		print(f'{self.separator}\n{self.header}\n{self.separator}')
		return varpars, momenta


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

	def _save_log(self, data, varpars, momenta, settings, is_ref=False):
		self._flush_buffer(settings)
		self.models['fe'].save(f'{settings["weights_dir"]}/weights_fe_{data["move"]}.pt')
		self.models['se'].save(f'{settings["weights_dir"]}/weights_se_{data["move"]}.pt')
		self.generator.save(f'{settings["results_dir"]}/generator.npy')
		torch.save(momenta['fe'], f'{settings["weights_dir"]}/momenta_fe_{data["move"]}.pt')
		torch.save(momenta['se'], f'{settings["weights_dir"]}/momenta_se_{data["move"]}.pt')
		torch.save(varpars, f'{settings["weights_dir"]}/varpars_{data["move"]}.pt')

		self.log = {
			"data": data.copy(),
			"files": {
				"weights": {'fe':f'{settings["weights_dir"]}/weights_fe_{data["move"]}.pt', 'se':f'{settings["weights_dir"]}/weights_se_{data["move"]}.pt' },
				"generator": f'{settings["results_dir"]}/generator.npy',
				"varpars": f'{settings["weights_dir"]}/varpars_{data["move"]}.pt',
				"momenta": {'fe':f'{settings["weights_dir"]}/momenta_fe_{data["move"]}.pt', 'se':f'{settings["weights_dir"]}/momenta_se_{data["move"]}.pt'},
				"weights_ref": { 'fe':f'{settings["weights_dir"]}/weights_fe_{data["move"]}.pt', 'se':f'{settings["weights_dir"]}/weights_se_{data["move"]}.pt' } if is_ref else self.log["files"]["weights_ref"],
			},
		}

		torch.save(self.log, f'{settings["results_dir"]}/log.pt')

	def _print_status(self, data, header=False):
		if header:
			print(f'// {self.name} status register:')
			print(f'{self.separator}\n{self.header}\n{self.separator}')

		data['time_h'] = data["time"] / 3600.

		line = ''
		for key, _, fp in self.formatter['sampling']: line = f'{line}|{format(data[f"{key}"], f".{fp}f"):^12}'
		line = f'{line}|' + ''.join([' ']*5)
		for key, _, fp in self.formatter['efficiency']: line = f'{line}|{format(data[f"{key}"], f".{fp}f"):^12}'
		line = f'{line}|'

		data.pop('time_h')
		print(f'{line}\n{self.separator}')
	
	def _print_pars(self, pars_idx, settings, idx):
		fixed = ''
		for name, param in self.models['fe'].NN.named_parameters():
			if not param.requires_grad:
				fixed = f'{fixed}, {name}'
		fixed = f'({fixed[2:]})'

		lines = []
		lines.append(f'# {self.name} parameters summary:')
		lines.append(f'# ')
		lines.append(f'# moves:                      {pars_idx["moves"]:.1e}')
		lines.append(f'# temperature:                {pars_idx["T"]:.1e}')
		lines.append(f'# initial temperatures ratio: {pars_idx["T_ratio_i"]:.1e}')
		lines.append(f'# final temperatures ratio:   {pars_idx["T_ratio_f"]:.1e}')
		lines.append(f'# mobility:                   {pars_idx["m1"]:.2f}')
		lines.append(f'# layer reset probability:    {pars_idx["p_reset"]:.2f}')
		lines.append(f'# maximum number of resets:   {pars_idx["max_resets"]:.0f}')
		lines.append(f'# mini-batch size:            {pars_idx["mbs"]:.0f}')
		lines.append(f'# ratchet constant:           {pars_idx["k"]:.1f}')
		if pars_idx["mean"]:
			lines.append(f'# mean variances:             {pars_idx["mean"]} (axis={pars_idx["axis"]})')
		else:
			lines.append(f'# mean variances:             {pars_idx["mean"]}')
		lines.append(f'# ')
		lines.append(f'# fixed layers: {fixed}')
		lines.append(f'# ')
		lines.append(f'# results directory: {settings["results_dir"]}')
		lines.append(f'# weights directory: {settings["weights_dir"]}')
		if idx == 0:
			lines.append(f'# restart:           {bool(settings["restart"])}')
		lines.append(f'# ')

		max_length = max([len(line) for line in lines])
		print('\n')
		print(''.join(['#'] * (max_length+2)))
		for line in lines:
			line = line + ''.join([' '] * (max_length-len(line)+1)) + '#'
			print(line)
		print(''.join(['#'] * (max_length+2)))
		print()



	def _correct_types(self, d, dname):
		# pars dictionary
		if dname == "varpars":
			types_and_keys = [
					(int,  ['moves', 'tot_moves', 'max_resets', 'axis', 'mbs', 'max_extractions', 'min_extractions', 'max_adj_step', 'min_adj_step', 'streak']),
					(bool, ['adj_ref', 'mean']),
			]
		# data dictionary
		else:
			types_and_keys = [(int, ['move'])]

		for key in d:
			for _type, keys in types_and_keys:
				if key in keys:
					d[key] = _type(d[key])

		return d

	def _init_attributes(self):
		self.buffer = StringIO()

		self.defpars = {
			"stime":(None, float),
			"T": (None, float),
			"T_ratio_i": (3.0e-2, float),
			"T_ratio_f": (1.0e-2, float),
			"m1": (0.3, float),
			"lamda": (0., float),
			"gamma": (0., float),
			"k": (1e4, float),
			"adj_ref": (1, bool),
			"p_reset": (0., float),
			"max_resets": (0, int),
			"dt": (1.0, float),
			"T_ratio_max": (1.0e-1, float),
			"mean": (True, bool),
			"axis": (0, int),
			"mbs": (512, int),
			"max_extractions": (5000, int),
			"min_extractions": (200, int),
			"threshold_est": (0.1, float),
			"max_adj_step": (50000, int),
			"min_adj_step": (5000, int),
			"threshold_adj": (0.1, float),
			"seed": (0, int),
			"loss_eq": (1., float), 
		}

		self.defsettings = {
			"results_dir": ("./results", str),
			"weights_dir": ("./results/weights", str),
			"data_step": (100, int),
			"log_step": (1000, int),
			"print_step": (1000, int),
			"verbose": (True, bool),
			"restart": (False, bool),
			"device": ("cpu", str),
			"num_threads": (1, int),
		}
		self.stepscale = 100
        
		self.formatter = {
			'sampling':[
				['move', 'move', 0],
				['fe_loss', 'U_fe', 5],
				['fe_cost', 'loss_fe', 5],
				['fe_metric', 'metric_fe', 5],
				['fe_metric_val', 'metric_val_fe', 5],
				['fe_mod2', 'mod2_fe', 1],
				['se_loss', 'U_se', 5],
				['se_cost', 'loss_se', 5],
				['se_metric', 'metric_se', 5],
				['se_metric_val', 'metric_val_se', 5],
				['se_mod2', 'mod2_se', 1],
				['time_h', 'time', 2],
            ],
			'efficiency':[
				['move', 'move', 0],
				['q', 'q', 5],
				['fe_q', 'q_fe', 5],
				['se_q', 'q_se', 5],
				['fe_eff_v', 'eff_v_fe', 5],
				['se_eff_v', 'eff_v_se', 5],
				['d', 'd', 3],
				['fe_d2', 'd2_fe', 3],
				['se_d2', 'd2_se', 3],
			],
		}

		self.separator = ''.join(['-']*(13*len(self.formatter['sampling'])+1)) + ''.join([' ']*5) + ''.join(['-']*(13*len(self.formatter['efficiency'])+1))
		self.header = ''
		for _, symbol, _ in self.formatter['sampling']: self.header = f'{self.header}|{symbol:^12}'
		self.header = f'{self.header}|' + ''.join([' ']*5)
		for _, symbol, _ in self.formatter['efficiency']: self.header = f'{self.header}|{symbol:^12}'
		self.header = f'{self.header}|'
