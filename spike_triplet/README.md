# Figure 7c-d: Spike-triplet plasticity

This folder contains the MATLAB code and experimental data used to fit the Predictive Dendrites Model to spike-pair and spike-triplet plasticity data shown in Figure 7c-d.

## 1. System requirements

The code has been tested with:

- Windows 11
- MATLAB R2023b and R2026a

No specialized hardware is required.

## 2. Installation

Install MATLAB with the Optimization Toolbox. No additional packages or installation steps are required.

## 3. Reproduction

The experimental data were extracted from Wang et al. [1] and are provided in `Triplet_Raw_Data.xlsx`.

Place `Triplet_Raw_Data.xlsx` and `fit_spike_triplet.m` in the same directory. In MATLAB, navigate to this directory and run:

```matlab
fit_spike_triplet
```

The script fits the parameters of the Predictive Dendrites Model to the experimental spike-pair and spike-triplet plasticity data. It prints the best-fit parameters, sum of squared errors, and predicted weight change for each stimulation condition. It also generates plots comparing the experimental data with the model predictions.

The optimization takes approximately 300 seconds to run.

## 4. Command-line options

The main simulation and fitting parameters can be modified directly in `fit_spike_triplet.m`.

The number of random optimization starts is set by:

```matlab
n_starts = 50;
```

By default, the optimization uses random initial parameter values. To obtain identical random initializations across repeated runs, uncomment:

```matlab
rng(61);
```

A different integer can be supplied to use a different random seed. The seed 61 is provided as an example for reproducible runs and is not necessarily the seed used to generate the results shown in the manuscript.

The bounds on the fitted parameters are specified by:

```matlab
lb = [0.01, -0.5, 0.1, 0.0, 0.0001];
ub = [0.5,   0.5, 0.9, 10.0, 2.0];
```

These correspond, in order, to the decay rate (`lambda`), plasticity threshold (`epsilon`), initial synaptic weight (`w0`), synaptic transmission delay (`tau`), and learning rate (`alpha`).

## References

[1] Wang, H.-X., Gerkin, R. C., Nauen, D. W. & Bi, G.-Q. Coactivation and timing-dependent integration of synaptic potentiation and depression. *Nature Neuroscience* **8**, 187–193 (2005). https://doi.org/10.1038/nn1387
