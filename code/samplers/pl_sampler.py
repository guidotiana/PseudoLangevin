import os

from io import StringIO
from time import process_time as ptime
from collections.abc import Callable

import math
import torch

from models.nnmodel import NNModel
from generator.custom_generator import CustomGenerator
from utils.general import create_path
from utils.operations import wcopy, compute_q, compute_d2, compute_mod2, is_subset, merge_dict, roundup, estimate_variance



### ------------------------------------ ###
### Double-Noise Pseudo-Langevin Sampler ###
### ------------------------------------ ###
class PLSampler():

	def __init__(
			self,
			model: NNModel,
			dataset: torch.utils.data.Dataset,
			Cost: Callable,
			Metric: Callable,
			name: str = 'PLSampler'
	):
		self.model = model
		self.dataset = dataset
		self.Cost = Cost
		self.Metric = Metric
		self.name = name

		self._init_attributes()


	
	def sample(
			self,
			pars: dict,
			settings: dict,
			start_fn: str | None = None,
	):
		pars_list, settings, data, varpars, momenta, steps_and_sample_list = self._setup(pars, settings, start_fn)
		del pars

		for idx, pars_idx in enumerate(pars_list):
			if idx > 0:
				data, varpars, momenta, steps_and_sample_list = self._start(pars_idx, settings, idx)
			del pars_idx

			data = self._correct_types(data, "data")
			varpars = self._correct_types(varpars, "varpars")			

			for (steps, sample) in steps_and_sample_list:
				momenta = self._integrate(momenta, varpars, steps)
				if sample:
					data = self._sample(momenta, varpars, data["move"]+steps)

				if data["move"]%settings["print_step"] == 0:
					self._print_status(data)
				if data["move"]%varpars['adj_step'] == 0:
					varpars, momenta = self._update_varpars(varpars=varpars, momenta=momenta, verbose=settings['verbose'])
				if data["move"]%settings["log_step"]==0:
					self._save_log(data, varpars, momenta, settings)

			momenta = self._integrate(momenta, varpars, steps=1)
			if idx+1 == len(pars_list):
				data = self._sample(momenta, varpars, varpars["tot_moves"])
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
		# 1. SETTINGS
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
		settings["data_step"] = roundup(multiple=settings["data_step"], divisor=settings["step_scale"])
		settings["log_step"] = roundup(multiple=settings["log_step"], divisor=settings["data_step"])
		settings["print_step"] = roundup(multiple=settings["print_step"], divisor=settings["data_step"])

		assert settings["num_threads"] <= 5, f"{self.name}.setup(): invalid value for 'num_threads' variable {settings['num_threads']}. Allowed values: num_threads <= 5."
		torch.set_num_threads(settings["num_threads"])

		settings["device"] = torch.device(settings["device"]) if isinstance(settings["device"], str) else settings["device"]
		if ("cuda" in settings["device"].type) and (not torch.cuda.is_available()):
			settings["device"] = torch.device("cpu")


		# 2. PARS
		# First the inputted pars dictionary is checked, verifying the all the necessary keys are present.
		# Then, the pars dictionary is completed, adding missing keys and checking the type for the inputted ones.
		# Then, the pars dictionary is splitted up in a list of dictionaries, where each instance of parameters must be executed 
		# when the previous one has been completed. Finally, the values are checked for each parameter instance.
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
			assert all([v>=0. for k,v in pars_list[idx].items() if k in ["max_resets", "gamma", "lamda", "bss", "threshold_est", "threshold_adj"]]), (
			    f'{self.name}._setup(): invalid value for one of the following keys ("max_resets", "gamma", "lamda", "bss", "threshold_est", "threshold_adj") at index {idx}. Allowed values: v>=0.'
			)
			assert all([v>0. for k,v in pars_list[idx].items() if k in ["stime", "moves", "tot_moves", "T", "dt", "max_extractions", "min_extractions", "max_adj_step", "min_adj_step"]]), (
			    f'{self.name}._setup(): invalid value for one of the following keys '
			    f'("stime", "moves", "tot_moves", "T", "dt", "max_extractions", "min_extractions", "max_adj_step", "min_adj_step") at index {idx}. Allowed values: v>0.'
			)
			assert all([v in [0,1] for k,v in pars_list[idx].items() if k in ["adj_ref", "mean"]]), (
			    f'{self.name}._setup(): invalid value for one of the following keys ("adj_ref", "mean") at index {idx}. Allowed values: v==0 or v==1.'
			)
			assert all([v<0 for k,v in pars_list[idx].items() if k in ["log_zerovar"]]), (
				f'{self.name}._setup(): invalid value for one of the following keys ("log_zerovar") at index {idx}. Allowed values: v<0.'
			)
			assert all([pars_list[idx][key]<=pars_list[idx]["T_ratio_max"] for key in ["T_ratio_i", "T_ratio_f"]]), (
				f'{self.name}._setup(): "T_ratio_i" ({pars_list[idx]["T_ratio_i"]}) and "T_ratio_f" ({pars_list[idx]["T_ratio_f"]}) must be lower than "T_ratio_max" ({pars_list[idx]["T_ratio_max"]}).',
				f'Check values at index {idx}.'
			)

			pars_list[idx]["min_adj_step"] = roundup(multiple=pars_list[idx]["min_adj_step"], divisor=settings["data_step"])
			pars_list[idx]["max_adj_step"] = roundup(multiple=pars_list[idx]["max_adj_step"], divisor=pars_list[idx]["min_adj_step"])
			pars_list[idx]["max_extractions"] = roundup(multiple=pars_list[idx]["max_extractions"], divisor=2*pars_list[idx]["min_extractions"])
			if pars_list[idx]["bss"] == 0:
				pars_list[idx]["bss"] = len(self.dataset)
			if idx == 0:
				pars_list[idx]["adj_ref"] = True


		# 3. CLEAN
		# First of all, load to device model dataset and generator (once an instance has been created).
		# Then, if restart is True (and everything required to restart a previous simulation exists), load the log information.
		# Otherwise, reset everything and proceed to clean every file (except for pars.txt) in results and weights directories.
		# The last three temporary attributes are here initiated, (log, weights_ref and t0).
		self.model.to(settings["device"])
		self.dataset.to(settings["device"])
		self.generator = CustomGenerator(
				seed=pars_list[0]["seed"],
				device=settings["device"],
		)

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
			self.generator.load(self.log['files']['generator'])
			varpars = torch.load(self.log["files"]["varpars"])
			momenta = torch.load(self.log["files"]["momenta"])
			self.weights_ref = torch.load(self.log['files']['weights_ref'], map_location=settings["device"], weights_only=True)

			steps_and_sample_list = self._get_steps_and_sample_list(varpars["tot_moves"], data["move"], settings["data_step"])

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
			data, varpars, momenta, steps_and_sample_list = self._start(pars_list[0].copy(), settings, 0)

		return (
			pars_list,
			settings,
			data,
			varpars,
			momenta,
			steps_and_sample_list,
		)



	def _start(self, pars_idx, settings, idx):
		self._print_pars(pars_idx, settings, idx)

		if pars_idx["adj_ref"]:
			self.weights_ref = self.model.copy(grad=False)
			q, d2 = 1., 0.
		else:
			q = compute_q(self.model.weights, self.weights_ref).item()
			d2 = compute_d2(self.model.weights, self.weights_ref).item()

		varpars, _ = self._update_varpars(varpars=pars_idx.copy(), momenta=None, verbose=settings["verbose"])
		momenta = {
				layer: torch.randn(values.shape, device=settings["device"], generator=self.generator.get()) * torch.sqrt(varpars['T']*varpars['M'][layer])
				for layer, values in self.model.weights.items()
		}

		data = {
			'move': varpars['tot_moves']-varpars['moves'],
			'time': ptime()-self.t0,
			'q': q,
			'd2': d2,
			'K': self._compute_K(momenta, varpars),
		}
		obs = self._compute_observables(lamda=varpars['lamda'], gamma=varpars['gamma'], bss=varpars['bss'])
		data = merge_dict(from_dict=obs, into_dict=data)
		self._extend_buffer(data, header=data['move']==0)
		self._save_log(data, varpars, momenta, settings, is_ref=varpars["adj_ref"])
		self._print_status(data)

		steps_and_sample_list = self._get_steps_and_sample_list(varpars["tot_moves"], data["move"], settings["data_step"])

		return data, varpars, momenta, steps_and_sample_list



	def _get_steps_and_sample_list(self, tot_moves, curr_move, step_size):
		remaining_moves = tot_moves-(curr_move+1)
		steps_and_sample_list = [(step_size, True)] * (remaining_moves//step_size)
		if remaining_moves%step_size > 0:
			steps_and_sample_list += [ (remaining_moves%step_size, False) ]
		else:
			steps_and_sample_list[-1][1] = False
		return steps_and_sample_list



	def _integrate(self, momenta, varpars, steps):
		for step in range(1, steps+1):
			old_grad = self._compute_grad(varpars)
			old_noise = self._generate_noise()

			with torch.no_grad():
				for layer in self.model.weights:
					vvd = momenta[layer]*varpars['c1']*varpars['dt']/varpars['M'][layer] - old_grad[layer]*varpars['dt']**2./(2.*varpars['M'][layer])
					bpn = old_noise[layer]*varpars['k_wn'][layer]*varpars['dt']/varpars['M'][layer]
					self.model.weights[layer] += vvd + bpn

			new_grad = self._compute_grad(varpars)
			new_noise = self._generate_noise()

			for layer in self.model.weights:
				vvd = momenta[layer]*(varpars['c1']**2.-1.) - (new_grad[layer]+old_grad[layer])*varpars['c1']*varpars['dt']/2.
				bpn = old_noise[layer]*varpars['c1']*varpars['k_wn'][layer] + new_noise[layer]*torch.sqrt( varpars['var'][layer]*(varpars['dt']*varpars['m1']/2.)**2 + varpars['k_wn'][layer]**2. )
				momenta[layer] += vvd + bpn

		return momenta



	def _sample(self, momenta, varpars, move):
		data = {
			'move': move,
			'time': ptime()-self.t0,
			'q': compute_q(self.model.weights, self.weights_ref).item(),
			'd2': compute_d2(self.model.weights, self.weights_ref).item(),
			'K': self._compute_K(momenta, varpars),
		}
		obs = self._compute_observables(lamda=varpars['lamda'], gamma=varpars['gamma'], bss=varpars['bss'])
		data = merge_dict(from_dict=obs, into_dict=data)
		self._extend_buffer(data)
		return data



	def _compute_observables(self, lamda, gamma, bss=None, x=None, y=None):
		mod2 = compute_mod2(self.model.weights)
		d2 = compute_d2(self.model.weights, self.weights_ref)

		# used during _integrate(), to compute the gradient on the current mini-batch (i.e. x, y)
		if bss is None:
			fx = self.model(x)
			cost = self.Cost(fx, y)
			loss = cost + (lamda/2.)*mod2 + (gamma/2.)*d2

			loss.backward(retain_graph=False)

			return {
				'loss': loss.detach().item(),
				'cost': cost.detach().item(),
				'mod2': mod2.detach().item(),
				'd2': d2.detach().item(),
				'metric': None,
			}
		
		# used during _sample(), to compute the values of the observables on the full-batch
		else:
			mod2 = mod2.detach().item()
			d2 = d2.detach().item()

			P = len(self.dataset)
			Nbs = P//bss if P%bss==0 else P//bss+1

			cost, metric = 0., 0.
			for ibs in range(Nbs):
				x, y = self.dataset[int(ibs*bss):int((ibs+1)*bss)]
				fx = self.model(x)
				cost_bs = self.Cost(fx, y) * len(x)/P
				cost += cost_bs.detach().item()
				metric_bs = self.Metric(fx, y) * len(x)/P
				metric += metric_bs.detach().item()

			return {
				'loss': cost + (lamda/2.)*mod2 + (gamma/2.)*d2,
				'cost': cost,
				'mod2':mod2,
				'd2': d2,
				'metric': metric,
			}

	def _compute_K(self, momenta, varpars):
		K = 0.
		for layer in momenta:
			K += (0.5*(momenta[layer]**2.)/varpars["M"][layer]).sum()
		return K.item()

	def _compute_grad(self, varpars):
		mb_mask = torch.zeros((len(self.dataset),), dtype=torch.bool)
		mb_idxs = torch.randint(low=0, high=len(self.dataset), size=(varpars['mbs'],), device=self.model.device, generator=self.generator.get())
		#WRONG: mb_idxs = torch.randperm(len(self.dataset), device=self.model.device, generator=self.generator.get())[:varpars['mbs']]
		mb_mask[mb_idxs] = True
		x, y = self.dataset[mb_mask]
		_ = self._compute_observables(lamda=varpars['lamda'], gamma=varpars['gamma'], x=x, y=y)
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
					sum_grad[layer] = value.detach().clone()
					sum2_grad[layer] = (value**2.).detach().clone()
			else:
				for layer, value in grad.items():
					sum_grad[layer] += value.detach().clone()
					sum2_grad[layer] += (value**2.).detach().clone()
		
		tot_extractions = varpars['min_extractions']
		curr_var = {}
		for layer in sum_grad:
			curr_var[layer] = estimate_variance(sum2_grad[layer].detach(), sum_grad[layer].detach(), tot_extractions, mean=varpars["mean"], axis=varpars["axis"])
			curr_var[layer][curr_var[layer] < 10**varpars["log_zerovar"]] = 10**varpars["log_zerovar"]
		if verbose:
			print(f"First estimate at {tot_extractions} extractions terminated.")

		# 2. Refine the current estimate up until the variance converges for each weight (or the maximum number of mini-batch extractions is reached)
		while tot_extractions < varpars['max_extractions']:
			for _ in range(varpars['min_extractions']):
				grad = self._compute_grad(varpars)
				for layer, value in grad.items():
					sum_grad[layer] += value.detach().clone()
					sum2_grad[layer] += (value**2.).detach().clone()
			tot_extractions += varpars['min_extractions']

			converged = []
			next_var = {}
			for layer, curr_var_l in curr_var.items():
				next_var[layer] = estimate_variance(sum2_grad[layer].detach(), sum_grad[layer].detach(), tot_extractions, mean=varpars["mean"], axis=varpars["axis"])
				next_var[layer][next_var[layer] < 10**varpars["log_zerovar"]] = 10**varpars["log_zerovar"]
				converged.append( 
						torch.all( abs( torch.sqrt(next_var[layer]/curr_var_l) - 1. ) <= varpars['threshold_est'] ).item()
				)
			if verbose:
				print(f"Further estimate at {tot_extractions} extractions. Converged: {all(converged)} ({sum(converged)}/{len(converged)}).")

			curr_var = wcopy(next_var)
			del next_var
			if all(converged):
				break

		return curr_var, tot_extractions

	def _update_varpars(self, varpars, momenta=None, verbose=True):
		print("\n!!! Update of the current mini-batch standard deviations and related parameters !!!\n")
		curr_var, tot_extractions = self._estimate_var(varpars, verbose)

		# 0. Initiate mini-batch temperatures
		if 'var' not in varpars.keys():
			varpars['c1'] = math.sqrt(1.-varpars['m1']**2.)
			varpars['M'], varpars['T_ratio'], varpars['k_wn'] = {}, {}, {}
			for layer, curr_var_l in curr_var.items():
				varpars['T_ratio'][layer] = torch.full_like(curr_var_l, varpars['T_ratio_i'])
				varpars['M'][layer] = curr_var_l*varpars['dt']**2./(4.*varpars['T_ratio_i']*varpars['T']*varpars['m1']**2.)
				varpars['k_wn'][layer] = torch.sqrt( varpars['M'][layer]*varpars['T']*varpars['m1']**2. - curr_var_l*(varpars['dt']/2.)**2. )

			varpars['streak'] = 1
			print(f"\nNew set of starting parameters: the streak is {varpars['streak']}.")

		else:
			idxs = torch.randperm(len(curr_var), device=self.model.device, generator=self.generator.get())
			perm_layers = [list(curr_var.keys())[idx.item()] for idx in idxs]
			resets = torch.rand((len(curr_var),), device=self.model.device, generator=self.generator.get()) < varpars["p_reset"]

			keep_streak = True
			counter = 0
			for perm_layer, reset in zip(perm_layers, resets):
				curr_var_pl = curr_var[perm_layer]
				curr_T_ratio_pl = varpars['T_ratio'][perm_layer] * curr_var_pl/varpars['var'][perm_layer]

				# 1.1 Reset mini-batch temperature to final value
				if reset and (counter < varpars['max_resets']):
					varpars['T_ratio'][perm_layer] = torch.full_like(curr_var_pl, varpars['T_ratio_f'])
					varpars['M'][perm_layer] *= curr_T_ratio_pl/varpars['T_ratio_f']
					momenta[perm_layer] = torch.randn(momenta[perm_layer].shape, device=self.model.device, generator=self.generator.get()) * torch.sqrt(varpars['T']*varpars['M'][perm_layer])
					counter += 1
					print(f"RESET: T_ratios on layer {perm_layer} have been reset to goal (final) value {varpars['T_ratio_f']} (counter={counter}).")

				# 1.2 Standard (controlled) update of the mini-batch temperature
				else:
					varpars['T_ratio'][perm_layer] = curr_T_ratio_pl

					mask = curr_T_ratio_pl > varpars['T_ratio_max']
					if mask.any().item():
						varpars['T_ratio'][perm_layer][mask] = varpars['T_ratio_max']
						varpars['M'][perm_layer][mask] *= curr_T_ratio_pl[mask]/varpars['T_ratio_max']

						flat_momenta_pl = momenta[perm_layer].flatten()
						if varpars["mean"]:
							repeated_shape = tuple( [1]*varpars["axis"] + list(momenta[perm_layer].shape[varpars["axis"]:]) )
							repeated_mask = mask.repeat(repeated_shape)
							flat_M_pl_masked = varpars['M'][perm_layer].repeat(repeated_shape)[repeated_mask].flatten()
							flat_momenta_pl[ repeated_mask.flatten() ] = torch.randn(repeated_mask.sum(), device=self.model.device, generator=self.generator.get()) * torch.sqrt(varpars['T']*flat_M_pl_masked)
						else:
							flat_M_pl_masked = varpars['M'][perm_layer][mask].flatten()
							flat_momenta_pl[ mask.flatten() ] = torch.randn(mask.sum(), device=self.model.device, generator=self.generator.get()) * torch.sqrt(varpars['T']*flat_M_pl_masked)
							
						momenta[perm_layer] = flat_momenta_pl.reshape(momenta[perm_layer].shape)
						print(f"ALERT: T_ratios on layer {perm_layer} have reached the threshold value {varpars['T_ratio_max']}. Starting the update of the mass matrix M!")
				
				varpars['k_wn'][perm_layer] = torch.sqrt( varpars['M'][perm_layer]*varpars['T']*varpars['m1']**2. - curr_var_pl*(varpars['dt']/2.)**2. )

				# 2. Check streak
				keep_streak *= torch.all( abs( torch.sqrt(curr_var_pl/varpars["var"][perm_layer]) - 1. ) < varpars['threshold_adj'] ).item()

			if keep_streak:
				varpars['streak'] += 1
				print(f"\nCompatible current and previous variances: the streak is increased to {varpars['streak']}.")
			else:
				varpars['streak'] = 1
				print(f"\nIncompatible current and previous variances: the streak is set back to {varpars['streak']}.")
		
		varpars['adj_step'] = min([varpars['streak']*varpars['min_adj_step'], varpars['max_adj_step']])
		print(f"The current adjournment step is {varpars['adj_step']}.")
		
		varpars['var'], varpars['tot_extractions'] = curr_var, tot_extractions

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
		if pars_idx["p_reset"] > 0.:
			lines.append(f'# final temperatures ratio:   {pars_idx["T_ratio_f"]:.1e}')
			lines.append(f'# layer reset probability:    {pars_idx["p_reset"]:.2f}')
			lines.append(f'# maximum number of resets:   {pars_idx["max_resets"]:.0f}')
		lines.append(f'# mobility:                   {pars_idx["m1"]:.2f}')
		lines.append(f'# lamda:                      {pars_idx["lamda"]:.1e}')
		lines.append(f'# gamma:                      {pars_idx["gamma"]:.1e}')
		lines.append(f'# mini-batch size:            {pars_idx["mbs"]:.0f}')
		if pars_idx["mean"]:
			lines.append(f'# mean variances:             {pars_idx["mean"]} (from axis={pars_idx["axis"]})')
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
					(int,  ['moves', 'tot_moves', 'max_resets', 'axis', 'mbs', 'max_extractions', 'min_extractions', 'max_adj_step', 'min_adj_step', 'streak', 'log_zerovar']),
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
			"p_reset": (0.0, float),
			"max_resets": (0, int),
			"dt": (1.0, float),
			"T_ratio_max": (1.0e-1, float),
			"mean": (True, bool),
			"axis": (0, int),
			"mbs": (1024, int),
			"bss": (0, int),
			"max_extractions": (5000, int),
			"min_extractions": (250, int),
			"threshold_est": (0.05, float),
			"max_adj_step": (100000, int),
			"min_adj_step": (10000, int),
			"threshold_adj": (0.05, float),
			"log_zerovar": (-9, int), 
			"seed": (0, int),
		}

		self.defsettings = {
			"results_dir": ("./results", str),
			"weights_dir": ("./results/weights", str),
			"data_step": (1000, int),
			"log_step": (10000, int),
			"print_step": (1000, int),
			"step_scale": (100, int),
			"verbose": (True, bool),
			"restart": (False, bool),
			"device": ("cpu", str),
			"num_threads": (1, int),
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
				['d2', 'd2', 3],
			],
		}

		self.separator = ''.join(['-']*(13*len(self.formatter['sampling'])+1)) + ''.join([' ']*5) + ''.join(['-']*(13*len(self.formatter['efficiency'])+1))
		self.header = ''
		for _, symbol, _ in self.formatter['sampling']: self.header = f'{self.header}|{symbol:^12}'
		self.header = f'{self.header}|' + ''.join([' ']*5)
		for _, symbol, _ in self.formatter['efficiency']: self.header = f'{self.header}|{symbol:^12}'
		self.header = f'{self.header}|'
