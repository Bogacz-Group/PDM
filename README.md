# PDM

This repository contains code to reproduce all simulation plots from the paper 
*Predictive Dendrites as a Foundation for Biological Learning*.

## Repository structure

Each subdirectory covers a specific figure or set of figures and has its own 
self-contained environment. Follow the README in that folder for system 
requirements, installation, reproduction, and instructions for use. Do not 
install a single project-wide environment.

| Directory | Figures |
|---|---|
| [`xor/`](xor/) | Figures 3c–d (firing-rate and spiking XOR) |
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
