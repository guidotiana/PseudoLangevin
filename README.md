# Pseudo-Langevin sampling

This repository contains the data, code, and usage templates associated with our recently published article ([PAPER](http://arxiv.org/abs/2603.15367)).



<details open><summary><b>Table of contents</b></summary>

- [Code](#code)
  - [Samplers](#samplers)
  - [Models](#models)
  - [Datasets](#datasets)
  - [Generator](#generator)
  - [Utils](#utils)
- [Usage templates](#usage)
  - [Pseudo-Langevin](#pseudoLangevin)
  - [hybrid Monte Carlo](#hybridMonteCarlo)
- [Article data](#article_data)
- [Contacts](#contacts)
</details>


---
## Code <a name="code"></a>

The `code/` directory is the core of this repository, as it contains the implementation of the sampling algorithms used in [PAPER](http://arxiv.org/abs/2603.15367).

#### samplers/ <a name="samplers"></a>
- `pl_sampler.py`: it contains the implementation of the pseudo-Langevin (pL) sampling method as the python class `PLSampler`. For usage examples, see [pL templates](pseudoLangevin).
- `hmc_sampler.py`: it contains the implementation of the hybrid Monte Carlo (hMC) algorithm as the python class `HMCSampler`. For usage examples, see [hMC templates](hybridMonteCarlo). 
- `suboptimal/`: it contains sub-optimal variants of the pL and hMC algorithms.

#### models/ <a name="models"></a>
- `nnmodel.py`: it contains the `NNModel` python class. It is derived from the `torch.nn.Module` class and it incapsulates the neural network model to be studied using the sampling classes `PLSampler` and `HMCSampler`.
- `ffn/`: the `ffn.py` python script contains the implementation of a standard three-layer feedforward neural network as the python class `FeedforwardNet` studied in [PAPER](http://arxiv.org/abs/2603.15367).

#### datasets/ <a name="datasets"></a>
- `KSpin/`: directory containing the code used to generate datasets of binary spin vectors, as described in [PAPER](http://arxiv.org/abs/2603.15367).
- `ProjFashionMNIST/`: directory containing the code used to generate datasets of projected FashionMNIST images, as described in [PAPER](http://arxiv.org/abs/2603.15367). 

#### generator/ <a name="generator"></a>
- `custom_generator.py`: it implements a custom pytorch random-number generator named `CustomGenerator` used in the sampling classes `PLSampler` and `HMCSampler`.

#### utils/ <a name="utils"></a>
- `general.py`: it contains functions for input file reading and directory management.
- `operations.py`: it contains functions which perform operations on neural network model weights, dictionaries and floats.


---
## Usage templates <a name="usage"></a>
Here are provided a few examples on how to use the implementation of the pseudo-Langevin and of the hybrid Monte Carlo algorithms.

### Pseudo-Langevin <a name="#pseudoLangevin"></a> 

### hybrid Monte Carlo <a name="#hybridMonteCarlo"></a>


---
## Article data <a name="article_data"></a>


---
## Contacts <a name="contacts"></a>

















