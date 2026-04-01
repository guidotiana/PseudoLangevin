import os

from io import StringIO
from time import process_time as ptime
from collections.abc import Callable

import math
import torch
import numpy as np

from models.nnmodel import NNModel
from generator.custom_generator import CustomGenerator
from utils.general import create_path
from utils.operations import wcopy, compute_q, compute_d, compute_d2, compute_mod2, is_subset, merge_dict, roundup, estimate_variance



### --------------------------------------------------------- ###
### Double-Noise Pseudo-Langevin Sampler with Norm Constraint ###
### --------------------------------------------------------- ###
class ConstrainedPLSampler():

	def __init__(
			self,
			model: NNModel,
			dataset: torch.utils.data.Dataset,
			dataset_val: torch.utils.data.Dataset,
			Cost: Callable,
			Metric: Callable,
			name: str = 'PLSampler'
	):
		self.model = model
		self.dataset = dataset
		self.dataset_val = dataset_val
		self.Cost = Cost
		self.Metric = Metric
		self.name = name
		self._init_attributes()
		self.mod2 = compute_mod2(self.model.weights)
		
	def sample(
			self,
			pars: dict,
			settings: dict,
			start_fn: str | None = None,
	):
		pars_list, settings, data, varpars, momenta = self._setup(pars, settings, start_fn)
		del pars
		
		for idx, pars_idx in enumerate(pars_list):
			if idx > 0:
				data, varpars, momenta = self._start(pars_idx, settings, idx)
			del pars_idx

			data = self._correct_types(data, "data")
			varpars = self._correct_types(varpars, "varpars")
			for move in range(data["move"]+1, varpars["tot_moves"]):
				if move%settings["data_step"] == 0:
					data, momenta, grad = self._step_and_sample(momenta, varpars, move)
				else:
					momenta, grad = self._step(momenta, varpars)

				if move%settings["print_step"] == 0:
					self._print_status(data)
				if move%varpars["adj_step"] == 0:
					varpars, momenta = self._update_varpars(varpars=varpars, momenta=momenta, verbose=settings['verbose'])
				if move%settings["log_step"]==0:
					self._save_log(data, varpars, momenta, settings)
			if idx+1 < len(pars_list):
				momenta = self._step(momenta, varpars)
			else:
				data, momenta = self._step_and_sample(momenta, varpars, varpars["tot_moves"])
				varpars, momenta = self._update_varpars(varpars=varpars, momenta=momenta, verbose=settings['verbose'])
				self._save_log(data, varpars, momenta, settings)
				self._print_status(data)

		del self.log, self.generator, self.weights_ref, self.t0



	def _setup(
			self,
			pars: dict,
			settings: dict,
			start_fn: str | None = None,
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

		self.model.to(settings["device"])
		self._weights_old = {k: torch.empty_like(v) for k, v in self.model.weights.items()}
		with torch.no_grad():
			for k, v in self.model.weights.items():
				self._weights_old[k].copy_(v)
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
			nec_wfiles = [f"{name}_0.pt" for name in ('weights', 'momenta', 'varpars')]
			settings["restart"] *= all([rf in os.listdir(settings['results_dir']) for rf in nec_rfiles] + [wf in os.listdir(settings['weights_dir']) for wf in nec_wfiles])

		if settings["restart"]:
			self.log = torch.load(f'{settings["results_dir"]}/log.pt')

			data = self.log["data"].copy()
			self.t0 = ptime()-data["time"]
			pars_list = [
					pars_idx for pars_idx in pars_list if pars_idx["tot_moves"]>data["move"]
			]

			self.model.load(self.log['files']['weights'])
			self._weights_old = {k: torch.empty_like(v) for k, v in self.model.weights.items()}
			with torch.no_grad():
				for k, v in self.model.weights.items():
					self._weights_old[k].copy_(v)
			self.generator.load(self.log['files']['generator'])
			varpars = torch.load(self.log["files"]["varpars"])
			momenta = torch.load(self.log["files"]["momenta"])
			self.weights_ref = torch.load(self.log['files']['weights_ref'], map_location=settings["device"], weights_only=True)

			self._print_pars(pars_list[0], settings, 0)
			self._print_status(data, header=True)

		else:

			for d in [settings['results_dir'], settings['weights_dir']]:
				for fn in os.listdir(d):
					if fn == 'pars.txt': continue
					if os.path.isfile(f'{d}/{fn}'):
						os.remove(f'{d}/{fn}')

			self.t0 = ptime()
			if start_fn is not None:
				self.model.load(start_fn)
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

		if pars_idx["adj_ref"]:
			self.weights_ref = self.model.copy(grad=False)
			q, d2 = 1., 0.
		else:
			q = compute_q(self.model.weights, self.weights_ref).item()
			d2 = compute_d2(self.model.weights, self.weights_ref).item()

		varpars, _ = self._update_varpars(varpars=pars_idx, momenta=None, verbose=settings["verbose"])
		momenta = {
				layer: torch.randn(values.shape, device=settings["device"], generator=self.generator.get()) * torch.sqrt(varpars['T']*varpars['M'][layer])
				for layer, values in self.model.weights.items()
		}
		# vincolo 
		with torch.no_grad():
				for layer in self.model.weights:
					momenta[layer] -=  (momenta[layer]*self.model.weights[layer]/varpars['M'][layer]).sum()*self.model.weights[layer]*varpars['M'][layer]/ compute_mod2(self.model.weights)
		
		data = {
			'move': varpars['tot_moves']-varpars['moves'],
			'time': ptime()-self.t0,
			'q': q,
			'd2': d2,
			'step_d': 0.,
			'step_dt': 0.,
			'eff_v': 0.,
			'K': self._compute_K(momenta, varpars),
			'dK': 0.,
		}
		obs = self._compute_observables(x=self.dataset.x, y=self.dataset.y, lamda=varpars['lamda'], gamma=varpars['gamma'], backward=False)
		data = merge_dict(from_dict=obs, into_dict=data)
		self._extend_buffer(data, header=data['move']==0)
		self._save_log(data, varpars, momenta, settings, is_ref=varpars["adj_ref"])
		self._print_status(data)
		return data, varpars, momenta


	# Integration with constraint
	def _step(self, momenta, varpars):
		old_grad = self._compute_grad(varpars)
		old_noise = self._generate_noise()
		
		with torch.no_grad():

			for k, v in self.model.weights.items():
				self._weights_old[k].copy_(v)
			for layer in self.model.weights:
				vvd = momenta[layer]*varpars['c1']*varpars['dt']/varpars['M'][layer] - old_grad[layer]*varpars['dt']**2./(2.*varpars['M'][layer])
				bpn = old_noise[layer]*varpars['k_wn'][layer]*varpars['dt']/varpars['M'][layer]
				self.model.weights[layer] += vvd + bpn
		
		S_1 = 0
		S_2 = 0
		S_3 = 0
		with torch.no_grad():
			for layer in self.model.weights:
				S_1 += (self.model.weights[layer]*self._weights_old[layer]/varpars['M'][layer]).sum()
				S_2 += (self.model.weights[layer]**2).sum()
				S_3 += ((self._weights_old[layer]/varpars['M'][layer])**2).sum()
			delta = S_1**2 + self.mod2*S_3 - S_3*S_2
			lamda_1 = (-S_1 +math.sqrt(delta)) / (varpars['dt']**2 * S_3)
			lamda_2 = (-S_1 - math.sqrt(delta)) / (varpars['dt']**2 * S_3)
			lamda = 0
			if abs(lamda_1) < abs(lamda_2):
				lamda = lamda_1
			else:
				lamda = lamda_2

			for layer in self.model.weights:
				self.model.weights[layer] += varpars['dt']**2/varpars['M'][layer]*lamda*self._weights_old[layer]


		new_grad = self._compute_grad(varpars)
		new_noise = self._generate_noise()
		with torch.no_grad():
			for layer in self.model.weights:
				vvd = momenta[layer]*(varpars['c1']**2.-1.) - (new_grad[layer]+old_grad[layer] - 2*lamda*self._weights_old[layer])*varpars['c1']*varpars['dt']/2.
				bpn = old_noise[layer]*varpars['c1']*varpars['k_wn'][layer] + new_noise[layer]*torch.sqrt(varpars['var'][layer]*(varpars['dt']*varpars['m1']/2. )**2. + varpars['k_wn'][layer]**2.)
				momenta[layer] += vvd + bpn
			
		S_4 = 0
		S_5 = 0
		with torch.no_grad():
			for layer in self.model.weights:
				S_4 += (self.model.weights[layer]*momenta[layer]/varpars['M'][layer]).sum()
				S_5 += (self.model.weights[layer]**2/varpars['M'][layer]).sum()
			Lamda = - S_4 / (S_5*varpars['dt']*varpars['c1'])

			for layer in self.model.weights:
				momenta[layer] -= varpars['c1']*varpars['dt']*self.model.weights[layer]*Lamda

		return momenta, old_grad



	def _step_and_sample(self, momenta, varpars, move):
		K_i = self._compute_K(momenta, varpars)
		weights_i = self.model.copy(grad=False)
		step_dt = ptime()
		momenta, grad = self._step(momenta, varpars)
		step_dt = ptime()-step_dt
		step_d = compute_d(self.model.weights, weights_i).item()

		data = {
			'move': move,
			'time': ptime()-self.t0,
			'q': compute_q(self.model.weights, self.weights_ref).item(),
			'd2': compute_d2(self.model.weights, self.weights_ref).item(),
			'step_d': step_d,
			'step_dt': step_dt,
			'eff_v': step_d/step_dt,
			'K': K_i,
			'dK': self._compute_K(momenta, varpars) - K_i
		}
		obs = self._compute_observables(x=self.dataset.x, y=self.dataset.y, lamda=varpars['lamda'], gamma=varpars['gamma'], backward=False)
		data = merge_dict(from_dict=obs, into_dict=data)
		self._extend_buffer(data)

		return data, momenta, grad



	def _compute_observables(self, x, y, lamda, gamma, backward=True):

		fx = self.model(x)
		cost = self.Cost(fx, y)
		mod2 = compute_mod2(self.model.weights)
		d2 = compute_d2(self.model.weights, self.weights_ref)
		loss = cost + (lamda/2.)*mod2 + (gamma/2.)*d2
		if backward:
			loss.backward()
		else: # observables computed only on the full batch (mainly during _step_and_sample())
			metric = self.Metric(fx, self.dataset.y)
			fx_val = self.model(self.dataset_val.x)
			metric_val = self.Metric(fx_val, self.dataset_val.y)
		return {
			'loss':loss.detach().item(),
			'cost':cost.detach().item(),
			'mod2':mod2.detach().item(),
			'd2': d2.detach().item(),
			'metric':metric.detach().item() if not backward else None,
			'metric_val':metric_val.detach().item() if not backward else None,
		}

	def _compute_K(self, momenta, varpars):
		K = 0.
		for layer in momenta:
			K += (0.5*(momenta[layer]**2.)/varpars["M"][layer]).sum()
		return K.item()

	def _compute_grad(self, varpars):
		mb_mask = torch.zeros((len(self.dataset),), dtype=torch.bool)
		mb_idxs = torch.randint(low=0, high=len(self.dataset), size=(varpars['mbs'],), device=self.model.device, generator=self.generator.get())
		mb_mask[mb_idxs] = True
		x, y, _ = self.dataset[mb_mask]
		
		_ = self._compute_observables(x=x, y=y, lamda=varpars['lamda'], gamma=varpars['gamma'], backward=True)
		grad = self.model.copy(grad=True)
		self.model.zero_grad()
		return grad

	def _generate_noise(self):
		return {
			layer: torch.randn(values.shape, device=self.model.device, generator=self.generator.get())
			for layer, values in self.model.weights.items()
		}


	def _estimate_var(self, varpars, verbose):
		# 1. Start by first computing the mini-batch-induced variances for each weight of the network
		sum_grad, sum2_grad = {}, {}
		for iext in range(varpars['min_extractions']):
			grad = self._compute_grad(varpars)
			if iext == 0:
				for layer, value in grad.items():
					sum_grad[layer] = value.detach()
					sum2_grad[layer] = (value**2.).detach()
			else:
				for layer, value in grad.items():
					sum_grad[layer] += value.detach()
					sum2_grad[layer] += (value**2.).detach()
			del grad
		
		tot_extractions = varpars['min_extractions']
		curr_var = {}
		for layer in sum_grad:
			curr_var[layer] = estimate_variance(sum2_grad[layer].detach(), sum_grad[layer].detach(), tot_extractions, mean=varpars["mean"], axis=varpars["axis"])
			curr_var[layer].clamp_(min=10**varpars["log_zerovar"])
		if verbose:
			print(f"First estimate at {tot_extractions} extractions terminated.")

		# 2. Refine the current estimate up until the variance converges for each weight (or the maximum number of mini-batch extractions is reached)
		while tot_extractions < varpars['max_extractions']:
			for _ in range(varpars['min_extractions']):
				grad = self._compute_grad(varpars)
				for layer, value in grad.items():
					sum_grad[layer] += value.detach()
					sum2_grad[layer] += (value**2.).detach()
				del grad
			tot_extractions += varpars['min_extractions']

			converged = []
			next_var = {}
			for layer, curr_var_l in curr_var.items():
				next_var[layer] = estimate_variance(sum2_grad[layer].detach(), sum_grad[layer].detach(), tot_extractions, mean=varpars["mean"], axis=varpars["axis"])
				next_var[layer].clamp_(min=10**varpars["log_zerovar"])
				converged.append(
					torch.allclose(
						torch.sqrt(next_var[layer]/curr_var_l), torch.ones_like(curr_var_l),
						atol=varpars['threshold_est'],
					)
				)
			if verbose:
				print(f"Further estimate at {tot_extractions} extractions. Converged: {all(converged)} ({sum(converged)}/{len(converged)}).")

			curr_var = wcopy(next_var)
			del next_var
			if all(converged):
				break

		return curr_var, tot_extractions

	def _update_varpars(self, varpars, momenta=None, verbose=True):
		print("\n!!! Update of the mini-batch noise variances and all the related parameters !!!\n")
		curr_var, tot_extractions = self._estimate_var(varpars, verbose)

		# 0. Initiate temperatures ratio and weight masses
		if 'var' not in varpars.keys():
			print(f"\nInitialization of the temperatures ratios and other parameters.\nThe streak is set to 1.")
			varpars['streak'] = 1
			varpars['c1'] = np.sqrt(1.-varpars['m1']**2.).item()
			varpars['M'], varpars['T_ratio'], varpars['k_wn'] = {}, {}, {}
			for layer, curr_var_l in curr_var.items():
				varpars['T_ratio'][layer] = torch.full_like(curr_var_l, varpars['T_ratio_i'])
				varpars['M'][layer] = curr_var_l*varpars['dt']**2./(4.*varpars['T_ratio_i']*varpars['T']*varpars['m1']**2.)
				varpars['k_wn'][layer] = torch.sqrt( varpars['M'][layer]*varpars['T']*varpars['m1']**2. - curr_var_l*(varpars['dt']/2.)**2. )

		else:
			keep_streak = True
			reset = torch.rand(1, device=self.model.device, generator=self.generator.get()).item() <= varpars["p_reset"]
			
			# 1. Reset temperatures ratio, weight masses and momenta according to the current variances
			if reset:
				print(f"\nReset of the temperatures ratios and extraction of new momenta.")
				for layer, curr_var_l in curr_var.items():
					varpars['T_ratio'][layer] = torch.full_like(curr_var_l, varpars['T_ratio_f'])
					varpars['M'][layer] = curr_var_l*varpars['dt']**2./(4.*varpars['T_ratio_f']*varpars['T']*varpars['m1']**2.)
					varpars['k_wn'][layer] = torch.sqrt( varpars['M'][layer]*varpars['T']*varpars['m1']**2. - curr_var_l*(varpars['dt']/2.)**2. )
					momenta[layer] = torch.randn(self.model.weights[layer].shape, device=self.model.device, generator=self.generator.get()) * torch.sqrt(varpars['T']*varpars['M'][layer])
					momenta[layer] -=  (momenta[layer]*self.model.weights[layer]/varpars['M'][layer]).sum()*self.model.weights[layer]*varpars['M'][layer]/ compute_mod2(self.model.weights)
					
					keep_streak *= torch.allclose(
						torch.sqrt(curr_var_l/varpars["var"][layer]), torch.ones_like(curr_var_l),
						atol=varpars['threshold_adj'],
					)
				
			# 2. Standard (controlled) update of the temperatures ratio
			else:
				print(f"\nStandard (controlled) update of the temperatures ratios and other parameters.")
				for layer, curr_var_l in curr_var.items():
					varpars["T_ratio"][layer] *= curr_var_l/varpars["var"][layer]
					mask = varpars["T_ratio"][layer] > varpars["T_ratio_max"]
					if mask.any().item():
						varpars, momenta, stats = self._increase_masses(varpars, momenta, curr_var_l, mask, layer)
						print(f"ALERT: {stats['n']} ({100*stats['f']:.1f}%) T_ratios on layer {layer} have reached the threshold value {varpars['T_ratio_max']}. Starting the update of the mass matrix M!")

					keep_streak *= torch.allclose(
						torch.sqrt(curr_var_l/varpars["var"][layer]), torch.ones_like(curr_var_l),
						atol=varpars['threshold_adj'],
					)

			if keep_streak:
				varpars['streak'] += 1
				print(f"Compatible current and previous variances: the streak is increased to {varpars['streak']}.")
			else:
				varpars['streak'] = 1
				print(f"Incompatible current and previous variances: the streak is set back to {varpars['streak']}.")
		
		varpars['adj_step'] = min([varpars['streak']*varpars['min_adj_step'], varpars['max_adj_step']])
		varpars['var'], varpars['tot_extractions'] = wcopy(curr_var), tot_extractions
		
		print(f"The current adjournment step is {varpars['adj_step']}.")
		print(f'Back to the simulation.\n')
		print(f'// {self.name} status register:')
		print(f'{self.separator}\n{self.header}\n{self.separator}')
		return varpars, momenta

	def _increase_masses(self, varpars, momenta, curr_var_l, mask, layer):
		varpars["T_ratio"][layer][mask] = varpars["T_ratio_max"]
		varpars["M"][layer][mask] = ( varpars["dt"]**2./(4.*varpars["T"]*varpars["m1"]**2) ) * curr_var_l[mask]/varpars["T_ratio_max"]
		varpars['k_wn'][layer] = torch.sqrt( varpars['M'][layer]*varpars['T']*varpars['m1']**2. - curr_var_l*(varpars['dt']/2.)**2. )

		momenta_l_shape = list(momenta[layer].shape)
		flat_momenta_l = momenta[layer].flatten()
		if varpars["mean"]:
			axis = varpars["axis"] if varpars["axis"]>=0 else max([len(momenta_l_shape)+varpars["axis"], 0])
			repeated_shape = tuple( [1]*axis + momenta_l_shape[axis:] )
			repeated_mask = mask.repeat(repeated_shape)
			flat_M_l_masked = varpars["M"][layer].repeat(repeated_shape)[repeated_mask].flatten()
			flat_momenta_l[ repeated_mask.flatten() ] = torch.randn(repeated_mask.sum(), device=self.model.device, generator=self.generator.get()) * torch.sqrt(varpars["T"]*flat_M_l_masked)
			n = repeated_mask.sum().item()
		else:
			flat_M_l_masked = varpars["M"][layer][mask].flatten()
			flat_momenta_l[ mask.flatten() ] = torch.randn(mask.sum(), device=self.model.device, generator=self.generator.get()) * torch.sqrt(varpars["T"]*flat_M_l_masked)
			n = mask.sum().item()
		momenta[layer] = flat_momenta_l.reshape(momenta_l_shape)

		N = np.cumprod(momenta_l_shape)[-1].item()
		stats = {"n":n, "f":n/N}
		return varpars, momenta, stats


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
		self.model.save(f'{settings["weights_dir"]}/weights_{data["move"]}.pt')
		self.generator.save(f'{settings["results_dir"]}/generator.npy')
		torch.save(varpars, f'{settings["weights_dir"]}/varpars_{data["move"]}.pt')
		torch.save(momenta, f'{settings["weights_dir"]}/momenta_{data["move"]}.pt')

		self.log = {
			"data": data.copy(),
			"files": {
				"weights": f'{settings["weights_dir"]}/weights_{data["move"]}.pt',
				"generator": f'{settings["results_dir"]}/generator.npy',
				"varpars": f'{settings["weights_dir"]}/varpars_{data["move"]}.pt',
				"momenta": f'{settings["weights_dir"]}/momenta_{data["move"]}.pt',
				"weights_ref": f'{settings["weights_dir"]}/weights_{data["move"]}.pt' if is_ref else self.log["files"]["weights_ref"],
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
		for name, param in self.model.NN.named_parameters():
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
			"log_zerovar": (-9, int),
			"seed": (0, int),
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
				['loss', 'U', 5],
				['cost', 'loss', 5],
				['metric', 'metric', 5],
				['metric_val', 'metric_val', 5],
				['mod2', 'mod2', 1],
				['time_h', 'time', 2],
            ],
			'efficiency':[
				['move', 'move', 0],
				['q', 'q', 5],
				['eff_v', 'eff_v', 5],
				['d2', 'd2', 3],
			],
		}

		self.separator = ''.join(['-']*(13*len(self.formatter['sampling'])+1)) + ''.join([' ']*5) + ''.join(['-']*(13*len(self.formatter['efficiency'])+1))
		self.header = ''
		for _, symbol, _ in self.formatter['sampling']: self.header = f'{self.header}|{symbol:^12}'
		self.header = f'{self.header}|' + ''.join([' ']*5)
		for _, symbol, _ in self.formatter['efficiency']: self.header = f'{self.header}|{symbol:^12}'
		self.header = f'{self.header}|'
