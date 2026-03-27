import torch
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from time import process_time as ptime



""" ################################# """
""" OPERATIONS ON WEIGHT DICTIONARIES """
""" ################################# """

# Calculate squared modulus of a weight vector
def compute_mod2(w):
    mod2 = 0.
    for name in w:
        mod2 += (w[name]**2).sum()
    return mod2

# Calculate modulus of a weight vector
def compute_mod(w):
    mod2 = compute_mod2(w)
    return torch.sqrt(mod2)

# Calculate squared distance between two weight vectors
def compute_d2(wi, wj):
    d2 = 0.
    for name in wi:
        d2 += ((wi[name]-wj[name])**2.).sum()
    return d2

# Calculate distance between two weight vectors
def compute_d(wi, wj):
    d2 = compute_d2(wi, wj)
    return torch.sqrt(d2)

# Calculate similarity between two weight vectors
def compute_q(wi, wj, mods=None):
    if mods is None:
        modi = compute_mod(wi)
        modj = compute_mod(wj)
    else:
        modi, modj = mods
    dotprod = 0.
    for name in wi:
        dotprod += (wi[name]*wj[name]).sum()
    q = dotprod/(modi*modj)
    return q

# Produce a copy of the weight vector
def wcopy(w):
    return {name: w[name].detach().clone() for name in w}



""" ########################## """
""" OPERATIONS ON DICTIONARIES """
""" ########################## """

# Merge two dictionaries
def merge_dict(from_dict, into_dict, overwrite=True):
    if overwrite:
        for key in from_dict:
            into_dict[key] = from_dict[key]
    else:
        for key in from_dict:
            if key not in into_dict.keys():
                into_dict[key] = from_dict[key]
    return into_dict

# Check that the keys of dictionary A are a subset of the keys of dictionary B
def is_subset(keys_A, keys_B):
	if len(keys_A) > len(keys_B):
		return False
	else:
		return all([k in keys_B for k in keys_A])



""" #################### """
""" OPERATIONS ON FLOATS """
""" #################### """

# Round up a x to be multiple of x0
def roundup(multiple, divisor):
    if multiple <= divisor:
        return divisor
    else:
        return round(multiple/divisor) * divisor



""" ##################### """
""" OPERATIONS ON TENSORS """
""" ##################### """

# Estimate the variance starting from the squared sum and the sum of N sampled values of the associated stochastic variable
def estimate_variance(sum2, summ, N, mean=False, axis=None):
    var = sum2/N - (summ/N)**2.
    if mean:
        if axis is not None:
            axis = torch.arange(var.ndim)[axis:].tolist() if axis<var.ndim else -1
        var = var.mean(axis=axis, keepdim=True)
    return var
