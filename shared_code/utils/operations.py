import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from time import process_time as ptime


# Find optimal permutation between two weights vectors
def find_permutation(wi, wj, check=True):
    if check:
        assert len(wi.keys())==len(wj.keys()), 'compute_q(): invalid inputs weights vectors. Dictionary lengths must coincide!'
        assert all([wi_layer in wj for wi_layer in wi]), 'compute_q(): invalid inputs weights vectors. Dictionary keys must coincide!'

    layers = [layer for layer in wi if 'bias' not in layer]
    for ilayer, layer in enumerate(layers):
        bias = f'{layer.rstrip("weight")}.bias'
        if bias in wi:
            wi[layer] = torch.concat(
                (wi[layer], wi[bias].unsqueeze(-1)),
                axis=-1
            )
            wj[layer] = torch.concat(
                (wj[layer], wj[bias].unsqueeze(-1)),
                axis=-1
            )

        wi[layer] = wi[layer].squeeze()
        wj[layer] = wj[layer].squeeze()
        assert wi[layer].ndim == 2, f'compute_q(): unexpected shape from layer {layer}: {tuple(wi[layer].shape)}. Expected (squeezed) dimensions: 2.'
        K, N_k = wi[layer].shape

        x, y = np.arange(K), np.arange(K)
        X, Y = np.meshgrid(x, y)

        cost = (wi[layer][Y.reshape(-1)]*wj[layer][X.reshape(-1)]).sum(axis=-1)
        cost = cost.reshape(K, K)
        row_ind, col_ind = linear_sum_assignment(cost.detach().numpy(), maximize=True)

        if ilayer < len(layers)-1:
            next_layer = layers[ilayer+1]
            wi[next_layer] = wi[next_layer][..., row_ind]
            wj[next_layer] = wj[next_layer][..., col_ind]

    return wi, wj

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



""" #################### """
""" OPERATIONS ON FLOATS """
""" #################### """

# Get order of magnitude of input number
def get_ofm(x):
    return abs(np.log10(abs(x)).astype(int))+1

# Evaluate if two numbers are close
def isclose(a, b, rel_tol=1e-08, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)



""" ###################################################################################################### """
""" funzioni introdotte per valutare best batch size, sono usate in examples/HMC_Batching/time_compute_obs """
""" ###################################################################################################### """

# calcolo potenziale su tutto il dataset dividendo il set in batch

def compute_observables(dataset, batch_size, model, Cost, pars, Metric):
    t0 = ptime()

    n_samples = dataset.x.size(0)
    cost_total = 0.0  
    fx_list = []      

    for i in range(0, n_samples, batch_size):
        x_batch = dataset.x[i : i + batch_size]
        y_batch = dataset.y[i : i + batch_size]
        fx_batch = model(x_batch)
        cost_batch = Cost(fx_batch, y_batch)
        cost_total += cost_batch.item() * x_batch.size(0)

        fx_list.append(fx_batch.detach())

    t1 = ptime()

    avg_cost = cost_total / n_samples
    avg_cost_tensor = torch.tensor(avg_cost, device=dataset.x.device)

    mod2 = compute_mod2(model.weights)
    reg_loss = (pars['lamda'] / 2.) * mod2

    loss_value = avg_cost + (pars['lamda'] / 2.) * mod2.item()
    loss_tensor = torch.tensor(loss_value, device=dataset.x.device)

    fx_all = torch.cat(fx_list, dim=0)
    metric = Metric(fx_all, dataset.y)

    t2 = ptime()

    return t1-t0, t2-t1



def compute_gradient(dataset, batch_size_grad, model, Cost, pars, Metric):
    num_samples = dataset.x.size(0)
    dt10, dt21 = 0., 0.

    for i in range(0, num_samples, batch_size_grad):
        t0 = ptime()

        x_batch = dataset.x[i : i + batch_size_grad]
        y_batch = dataset.y[i : i + batch_size_grad]
        fx_batch = model(x_batch)
        cost_batch = Cost(fx_batch, y_batch)

        model.zero_grad()
        cost_batch.backward(retain_graph=True)

        t1 = ptime()

        cost_batch_tensor = torch.tensor(cost_batch, device=dataset.x.device)

        mod2 = compute_mod2(model.weights)
        reg_loss = (pars['lamda'] / 2.) * mod2
        reg_loss.backward(retain_graph=True)

        loss_value = cost_batch + (pars['lamda'] / 2.) * mod2.item()
        loss_tensor = torch.tensor(loss_value, device=dataset.x.device)

        t2 = ptime()
        dt10 += t1-t0
        dt21 += t2-t1

    return dt10/(num_samples//batch_size_grad), dt21/(num_samples//batch_size_grad)



# calcolo il potenziale solo su un batch

def compute_observables_batch(dataset, batch_size_grad, model , Cost, pars, Metric):

    t0 = ptime()
    batch = int(dataset.x.size(0) /batch_size_grad)

    b = torch.randint(0, batch + 1, (1,)).item()

    x_batch = dataset.x[batch_size_grad * b : batch_size_grad * b + batch_size_grad]
    y_batch = dataset.y[batch_size_grad * b : batch_size_grad * b + batch_size_grad]
    fx_batch = model(x_batch)
    cost_batch = Cost(fx_batch, y_batch)

    model.zero_grad()
    cost_batch.backward(retain_graph=True)

    t1 = ptime()

    cost_batch_tensor = torch.tensor(cost_batch, device=dataset.x.device)

    t2 = ptime()

    mod2 = compute_mod2(model.weights)
    reg_loss = (pars['lamda'] / 2.) * mod2
    reg_loss.backward(retain_graph=True)

    t3 = ptime()

    loss_value = cost_batch + (pars['lamda'] / 2.) * mod2.item()
    loss_tensor = torch.tensor(loss_value, device=dataset.x.device)

    t4 = ptime()

    return t4 - t0


def acceptance_rate(df):
    return df.loc[len(df)-1].am / df.loc[len(df)-1].move
