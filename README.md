# PDM

This repository contains code to reproduce all simulation plots from the paper 
*Predictive Dendrites as a Foundation for Biological Learning*.

## Repository structure

Each subdirectory covers a specific figure or set of figures and has its own 
self-contained environment. Follow the README in that folder for system 
requirements, installation, reproduction, and command-line options. Do not 
install a single project-wide environment.

| Directory | Figures |
|---|---|
| [`xor/`](xor/) | Figures 3c–d (firing-rate and spiking XOR) |
| [`spiking_predictive_dendrites/`](spiking_predictive_dendrites/) | Figure 5b–c (spiking predictive-dendrite dynamics and parameter effects) |
| [`spike_timing_dependent_plasticity/`](spike_timing_dependent_plasticity/) | Figure 6 (spike-timing-dependent plasticity dynamics, analytic result, and experimental-data fits) |
| [`initial_synaptic_strength/`](initial_synaptic_strength/) | Figure 7a-b (effect of initial synaptic strength on spike-timing-dependent plasticity) |
| [`spike_triplet/`](spike_triplet/) | Figure 7c-d (effect of spike patterns on spike-timing-dependent plasticity) |
| [`firing_rate_xor_weight_trajectories/`](firing_rate_xor_weight_trajectories/) | Supplementary Figure 3c (firing-rate XOR weight trajectories) |
| [`spiking_xor_weight_trajectories/`](spiking_xor_weight_trajectories/) | Supplementary Figure 3d (spiking XOR weight trajectories) |
| [`supplementary_firing_rate/`](supplementary_firing_rate/) | Supplementary Figures 2 (XOR), 4 (linear and nonlinear toy tasks), and 5 (Iris) |

## Getting started

```bash
git clone https://github.com/Bogacz-Group/PDM.git
cd PDM
```

Then enter the subdirectory for the figures you want to reproduce and follow its README.

## Citation

If you use this code, please cite:

> Predictive Dendrites as a Foundation for Biological Learning.

## License

This project is released under the Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) licence.
