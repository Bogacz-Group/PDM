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
| [`xor/`](xor/) | Figures 3c–d (firing-rate and spiking XOR) and Supplementary Figure 3c (firing-rate weight trajectories) |
| [`spiking/`](spiking/) | Figures 5b–c, 6a–j, 7a–b, and Supplementary Figure 3d |
| [`supplementary_firing_rate/`](supplementary_firing_rate/) | Supplementary Figures 2 (XOR), 4 (linear and nonlinear toy tasks), and 5 (Iris) |

Figure 5a is a conceptual schematic rather than a numerical experiment, so it
does not have a reproduction script.

The machine-readable [`migration_manifest.json`](migration_manifest.json)
pins the original analysis repository to commit
`370b8daebcd6d52e37075f362f29e3a6c59e1ca3` and records the source path for
every migrated panel.

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

This project is released under the MIT License. See [`LICENSE`](LICENSE) for
details.
