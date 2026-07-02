## PAPER-1 data

This repository contains the data necessary to reproduce the plots shown in [PAPER-1](http://arxiv.org/abs/2603.15367). Here is a brief description of the contant of each sub-directory.

#### Fig.3
The data is relative to the N=11160 model applied to the binary spin vector problem. It is distributed in three sub-directories, `Fig.3a/`, `Fig.3b/`, `Fig.3c/`, and `Fig.3d/`, that is, one for each panel.
- `Fig.3a/`, it contains three types of files:
  - `distribution_simulation_<i>.csv`: the normalized distribution of values ("values") of the mini-bath noise at different times ("move 0", "move 5000", ...) during a pL sampling starting from the \<i\>-th starting vector.
  - `update_simulation_<i>.csv`: the values of the mini-batch gradient variances at three different times ("move 0", "move 25000", "move 50000") during a pL sampling starting from the \<i\>-th starting vector.
  - `KSfrac_simulation_<i>.csv`:the result of the Kolmogorov-Smirnov test applied on the values in the `distribution.csv` file with respect to the standard normal distibution at different times ("move 0", "move 5000", ...) during a pL sampling starting from the \<i\>-th starting vector.
- `Fig.3b/`, it contains three types of files:
  - `distribution_minibatch-noise_simulation_<i>.csv`: the normalized distribution of values ("values") of the correlation matrix computed at different times ("move 0", "move 5000", ...) for the mini-batch noise during a pL sampling starting from the \<i\>-th starting vector.
  - `distribution_white-noise_simulation_<i>.csv`: the normalized distribution of values ("values") of the correlation matrix computed at different times ("move 0", "move 5000", ...) for the white noise during a pL sampling starting from the \<i\>-th starting vector.
  - `distribution_full-noise_simulation_<i>.csv`: the normalized distribution of values ("values") of the correlation matrix computed at different times ("move 0", "move 5000", ...) for the total noise (i.e., mini-batch noise plus white noise) during a pL sampling starting from the \<i\>-th starting vector.
- `Fig.3c`, it contains three types of files:
  - `autocorr_minibatch-noise_simulation_<i>.csv`: the average and the standard deviation (over the network components) of the autocorrelation function with respect to different starting times ("move 0, avg", "move 0, err", "move 5000, avg", "move 5000, err", ...) for the mini-batch noise during a pL sampling starting from the \<i\>-th starting vector.
  - `autocorr_white-noise_simulation_<i>.csv`: the average and the standard deviation (over the network components) of the autocorrelation function with respect to different starting times ("move 0, avg", "move 0, err", "move 5000, avg", "move 5000, err", ...) for the white-batch noise during a pL sampling starting from the \<i\>-th starting vector.
  - `autocorr_full-noise_simulation_<i>.csv`: the average and the standard deviation (over the network components) of the autocorrelation function with respect to different starting times ("move 0, avg", "move 0, err", "move 5000, avg", "move 5000, err", ...) for the full noise (i.e., mini-batch noise plus white noise) during a pL sampling starting from the \<i\>-th starting vector.
- `Fig.3d`: it contains two files, `thermodynamics_hMC.csv` and `thermodynamics_pL.csv`, which report the estimate for the average and standard deviation values of a set of observables ("loss" as loss function, "cost" as cost function, "mod2" as squared norm and "error" as training error) at different temperatures, obtained through hMC and pL simulations, respectively.

#### Fig.4
The data is relative to the binary spin vector problem. It is distributed in three sub-directories, `N_11160/`, `N_101610/` and `N_1006110/`, that is, one for each model size. Each directory contains file of the following format, `<sampler>_simulation_<i>.csv`, where "sampler" stands for the sampling scheme used ("hMC", "mbhMC", "pL", "pSGLD" or "SGHMC"). Each file contains the average ("d2, avg") and the standard deviation ("d2, err") of the squared distance computed between all the pairs of weight vectors separated by the same wall–clock time interval ("Dt\_W") sampled at equilibrium at T=10^(-6) during the \<i\>-th simulation with optimized hyperparameters for the model of size \<N\>.

#### Fig.5
The data is relative to the N=1006110 model applied to the binary spin vector problem. It is distributed in the following files:
- `Adam_best.csv`: the error function evaluated on the training set ("train\_error"), on the validation set ("val\_error") and on the test set ("test\_error") for the three best models trained with Adam as optimizer with early-stopping.
- `thermodynamics.csv`: the estimate for the average and standard deviation values of a set of observables ("loss" as loss function, "cost" as cost function, "mod2" as squared norm, "train\_error" as training error, and "test\_error" as test error) at different temperatures.
- `Adam_simulation.csv`: the error function evaluated on the training set ("train\_error"), on the validation set ("val\_error") and on the test set ("test\_error") during an Adam training without early-stopping as function of the training epoch ("epoch") and the wall-clock time ("t\_W").
- `T<T>_simulation.csv`: the error function evaluated on the training set ("train\_error") and on the test set ("test\_error") during a pL sampling at temperature \<T\> from an initialized weight vector as function of the move ("move") and the wall-clock time ("t\_W").

#### Fig.6
The data is relative to the N=1006110 model applied to the MNIST problem. Each file has the following format, `<nn_type>_<sampler>_optimization_<i>.csv`, where "nn\_type" and "sampler" represent respectively the network type ("ffn" or "cnn") and the optimizer/sampler ("Adam", "pL" or "pSGLD") used for the \<i\>-th optimization. Each file contains the values of the error function evaluated on the training set ("train\_error") and on the test set ("test\_error") during the optimization as functions of the wall-clock time "t\_W" and of the epoch/move (depending on the optimizer type).

