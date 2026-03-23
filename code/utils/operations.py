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

# Calculate sum of two weight vectors
def wsum(wi, wj, requires_grad=False):
    if not requires_grad:
        return {name: (wi[name]+wj[name]).detach().clone() for name in wi}
    else:
        return {name: wi[name]+wj[name] for name in wi}

# Calculate difference of two weight vectors
def wdiff(wi, wj, requires_grad=False):
    if not requires_grad:
        return {name: (wi[name]-wj[name]).detach().clone() for name in wi}
    else:
        return {name: wi[name]-wj[name] for name in wi}

# Calculate the product between two weights vectors
def wprod(wi, wj, requires_grad=False):
	if not requires_grad:
		return {name: (wi[name]*wj[name]).detach().clone() for name in wi}
	else:
		return {name: wi[name]*wj[name] for name in wi}

# Multiply weight vector by constant
def kprod(w, k, requires_grad=False):
    if not requires_grad:
        return {name: (k*w[name]).detach().clone() for name in w}
    else:
        return {name: k*w[name] for name in w}

# Elevate weight vector elements to the k-th power
def kpow(w, k, requires_grad=False):
    if not requires_grad:
        return {name: (w[name]**k).detach().clone() for name in w}
    else:
        return {name: w[name]**k for name in w}

# Rescale the norm of a weight vector
def rescale(w, new_mod, old_mod=None, requires_grad=False):
    if old_mod is None: old_mod = compute_mod(w)
    return kprod(w, new_mod/old_mod, requires_grad=requires_grad)

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

# Merge many dictionaries
def merge_dicts(from_dicts, into_dict, overwrite=True):
    for from_dict in from_dicts:
        into_dict = merge_dict(from_dict, into_dict, overwrite)
    return into_dict

# Product between the values of two dictionaries (expected to be floats, tensors or arrays)
def prod_dicts(first_dict, second_dict, keys_from:str="first"):
    if keys_from == "first":
        return {first_dict[key]*second_dict[key] for key in first_dict}
    elif keys_from == "second":
        return {first_dict[key]*second_dict[key] for key in second_dict}
    else:
        raise ValueError(f"prod_dicts(): keys_from is supposed to be a string with values ('first', 'second'), but found {keys_from}.")

# Check that the keys of dictionary A are a subset of the keys of dictionary B
def is_subset(keys_A, keys_B):
	if len(keys_A) > len(keys_B):
		return False
	else:
		return all([k in keys_B for k in keys_A])



""" #################### """
""" OPERATIONS ON FLOATS """
""" #################### """

# Get order of magnitude of input number
def get_ofm(x):
    return abs(np.log10(abs(x)).astype(int))+1

# Evaluate if two numbers are close
def isclose(a, b, rel_tol=1e-08, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

# Round up a x to be multiple of x0
def roundup(multiple, divisor):
    if multiple <= divisor:
        return divisor
    else:
        return round(multiple/divisor) * divisor

# Check the bounds of a variable x
def included(x, lim=[-math.inf, math.inf], eq=[0, 0]):
    return (x>=lim[0] if eq[0] else x>lim[0]) and (x<=lim[1] if eq[1] else x<lim[1])



""" ###################### """
""" OPERATIONS ON MATRICES """
""" ###################### """

def compute_correlation(X: torch.Tensor):
    #Function to compute the correlation C_ij among the elements of a matrix X_ij of shape (samples, dim),
    #where "dim" stands for the dimension of the stochastic variable x,
    #while "samples" correponds to the number of realizations.
    #
    #Input: X (torch.Tensor)).
    #Output: C (correlation matrix of the same shape as X).
    C = np.corrcoef(X.cpu().detach().numpy(), rowvar=False)
    return torch.tensor(C).to(X.device)

def compute_autocorrelation(X: torch.Tensor, dt_min: int = 1, normalize: bool = False):
    #Function to compute the autocorrelation Chi_i of a stochastic variable x of dimension "dim".
    #The input matrix X_ij must be of shape (time, dim), with "time" being
    #the index controlling the time evolution of the stochastic variable x.
    #
    #Input: X (torch.Tensor, input matrix);
    #       dt_min (int, minumum time interval over which averages are computed);
    #       normalize (bool, whether or not to normalize the input X matrix row by row).
    #Output: Chi (autocorrelation matrix of shape (dim, time-dt_min)).
    t_max, dt_max = len(X), len(X)-dt_min
    if normalize:
        with torch.no_grad():
            X /= (X**2).mean(axis=1, keepdim=True)

    Chi = [
        list( (X[:t_max-dt]*X[dt:]).mean(axis=0) - X[:t_max-dt].mean(axis=0)*X[dt:].mean(axis=0) ) for dt in range(dt_max+1)
    ]
    Chi = torch.tensor(Chi).to(X.device) / X.var(axis=0)
    
    #NOTE: if a variable x_i is fixed, var(x_i)=0.
    #      In that case, chi_i(t)=1. forall t
    mask = ~( ~Chi.isnan() * ~Chi.isinf() )
    Chi[mask] = 1.
    return Chi

def estimate_variance(sum2, summ, N, mean=False, axis=None):
    var = sum2/N - (summ/N)**2.
    if mean:
        if axis is not None:
            axis = torch.arange(var.ndim)[axis:].tolist() if axis<var.ndim else -1
        var = var.mean(axis=axis, keepdim=True)
    return var
