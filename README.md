# Pseudo-Langevin sampling

This repository contains the data, code, and usage templates associated with the published articles [PAPER-1](http://arxiv.org/abs/2603.15367) and [PAPER-2](http://arxiv.org/abs/2603.29529).


<details open><summary><b>Table of contents</b></summary>

- [Code](#Code)
  - [samplers/](#Code_samplers)
  - [models/](#Code_models)
  - [datasets/](#Code_datasets)
  - [generator/](#Code_generator)
  - [utils/](#Code_utils)
- [Usage templates](#Usage)
- [Article data](#Article-data)
</details>



---
## Code <a name="Code"></a>

The `code/` directory is the core of this repository, as it contains the implementation of the sampling algorithms, the neural network architectures and the datasets used in [PAPER-1](http://arxiv.org/abs/2603.15367) and [PAPER-2](http://arxiv.org/abs/2603.29529). Here follows a brief description of the content of each its sub-directories.

#### samplers/ <a name="Code_samplers"></a>
- `pl_sampler.py`: the implementation of the pseudo-Langevin (pL) sampling method as the python class `PLSampler`. For usage examples, see [templates](Usage).
- `cpl_sampler.py`: the implementation of the pseudo-Langevin algorithm with norm constraints (CpL) as the python class `ConstrainedPLSampler`. For usage examples, see [templates](Usage).
- `hmc_sampler.py`: the implementation of the hybrid Monte Carlo (hMC) algorithm as the python class `HMCSampler`. For usage examples, see [templates](Usage). 
- `suboptimal/`: sub-optimal variants of the pL and hMC algorithms.

#### models/ <a name="Code_models"></a>
- `nnmodel.py`: it contains the `NNModel` python class. It is derived from the `torch.nn.Module` class and it incapsulates the neural network model to be studied. It is used in the sampling classes `PLSampler`, `ConstrainedPLSampler` and `HMCSampler`.
- `cnn/`: implementation of a convolutional neural network as the python class `ConvolutionalNet`, studied in [PAPER-1](http://arxiv.org/abs/2603.15367).
- `plain_ffn/`: implementation of a standard three-layer feedforward neural network with ReLU activation functions as the python class `PlainFFNet`, studied in [PAPER-1](http://arxiv.org/abs/2603.15367).
- `pooling_ffn/`: implementation of a feedforward neural network with embedding and pooling layers for masked language modeling tasks as the python class `PoolingFFNet`, studied in [PAPER-2](http://arxiv.org/abs/2603.29529).
- `transformer/`: implementation of an L-module tranformer-based encoder for masked language modeling tasks as the python class `TFNet`, studied in [PAPER-2](http://arxiv.org/abs/2603.29529).

#### datasets/ <a name="Code_datasets"></a>
- `KSpin/`: directory containing the code used to generate datasets of binary spin vectors, as described in [PAPER-1](http://arxiv.org/abs/2603.15367).
- `MNIST/`: directory containing downloaded MNIST images and the code used to generate datasets from them, as described in [PAPER-1](http://arxiv.org/abs/2603.15367). 
- `Protein/`: directory containing masked sequences derived from the acyl–coenzyme A binding protein contact map and the code to generate datasets from them, as described in [PAPER-2](http://arxiv.org/abs/2603.29529).

#### generator/ <a name="Code_generator"></a>
- `custom_generator.py`: the implementation of a custom pytorch random-number generator named `CustomGenerator` used in the sampling classes `PLSampler`, `ConstrainedPLSampler` and `HMCSampler`.

#### utils/ <a name="Code_utils"></a>
- `general.py`: functions for input file reading and directory management.
- `operations.py`: functions for operations on neural network parameters, dictionaries and float/integer variables.



---
## Usage templates <a name="Usage"></a>
Here are provided a few examples on how to use the implementation of the pseudo-Langevin algorithm.<br>
As a prerequisite, you must have PyTorch installed to use this repository. You can use this one-liner for cloning:
```bash
git clone "https://github.com/guidotiana/PseudoLangevin"
```

Once you cloned this directory, you must define the neural network, the dataset, and the cost and metric functions for your problem. In this instance, the model is a three-layer feedforward neural network with ReLU activation functions on each layer, the dataset consists of binary spin vectors, and the cost and metric functions are the cross-entropy and the error functions, respectively.
```python
import torch
import torch.nn.functional as F

# Pseudo-Langevin code import
import sys
sys.path.append('PseudoLangevin/code')
from models.plain_ffn.ffn import PlainFFNet
from models.nnmodel import NNModel
from datasets.KSpin.kspin_dataset import generate_kspin_datasets


# Initiate a three-layer FF network, with ReLU activation functions.
net = PlainFFNet(
        input_dim=100,           #input layer dimension
        hidden_dims=[500, 100],  #first and second hidden layer dimensions
        output_dim=10,           #output layer dimension
)

# Create an instance of the NNModel class, with <net> as the associated neural network.
model = NNModel(net)

# Generate a training set of binary spin vectors.
datasets = generate_kspin_datasets(
        K=10,                    #number of classes
		d=100,                   #dimension of the spin vectors
		pflip=0.355,             #flipping probability
		P_train=100_000,         #number of examples in the training set
)

# Define the cost and metric functions to evaluate during the sampling.
Cost = lambda logits, target: F.cross_entropy(logits, target)
Metric = lambda logits, target: (logits.argmax(dim=1) == target).float().mean()
```
Now you can choose one of the algorithms present in the `samplers/` directory. As an example, let's see a rapid implementation for a pL sampling simulation.
```python
from samplers.pl_sampler import PLSampler

# Initiate the PLSampler class with the NNModel, the datasets,
# and the Cost and Metric functions you previously defined.
sampler = PLSampler(
        model=model,
		datasets=datasets,
		Cost=Cost,
		Metric=Metric,
)

# Define the parameters dictionary "pars" for the sampling.
pars = {
        "stime": 10_000,        #the total simulated time, i.e., the number of moves times the integration time step
		"T": 1.0e-6,            #the temperature of the sampling
		"T_ratio_i": 0.01,      #the initial temperatures ratio applied to each parameter of the network
		"mbs": 512,             #the size of the mini-batches used for integration
		"adj_step": 1_000,      #the number of steps after which the mini-batch gradient variances are updated
}

# Eventually, add an optional dictionary "settings",
# containing other variables that do not influence the sampling.
settings = {
	    "results_dir": "./pL_results",  #results directory
	    "device": "cuda",               #device for the simulation
}

# Start the sampling.
sampler.sample(
        pars=pars,
        settings=settings,
)
```

In the directory `templates/`, you can find examples of combinations of sampling algorithm, neural network and task (as `<algo>_<net>_<task>/` sub-directories). Within each sub-directory, you will find a python script to run the simulation (`main.py`) with default parameters (`pars.txt` and `settings.txt`). Both parameter files include detailed information on every variable used to run the simulation.



---
## Article data <a name="Article-data"></a>
This directory, `PAPER-1_data/`, contains all the data necessary to reproduce the plots shown in [PAPER-1](http://arxiv.org/abs/2603.15367), along with a brief description of how such data is distributed within each sub-directory.

---
