# Digitized experimental data

These four CSV files are the numerical points used for the fits in Fig. 6i-j
and Fig. 7b. They were copied without numerical alteration from the analysis
workspace. They are points digitized from published figures, not newly
released source measurements; the digitization uncertainty and experimental
error bars are not available in these tables.

| File | PDM panel | Columns and transformation | Experimental source |
| --- | --- | --- | --- |
| `Bi et al. 2002.csv` | Fig. 6i | `x` is the pre-post interval in ms. `y` is normalized final weight; the script plots `(y - 1) * 100` as percentage weight change. | Fig. 3 of Bi & Wang (2002) |
| `Melanie et al. 2003.csv` | Fig. 6j | `x` is the pre-post interval in ms and `y` is already percentage weight change; no numerical transform is applied. The historical filename uses the first author's given name and is retained to preserve provenance. | Fig. 2 of Woodin, Ganguly & Poo (2003) |
| `positive_spiking.csv` | Fig. 7b | `x` is initial EPSP amplitude in pA and `y` is percentage weight change. The script assigns a pre-post interval of +5 ms to every row. | Filled-circle condition in Fig. 5 of Bi & Poo (1998) |
| `negative_spiking.csv` | Fig. 7b | `x` is initial EPSP amplitude in pA and `y` is percentage weight change. The script assigns a post-pre interval of -6 ms to every row. | Crosses condition in Fig. 5 of Bi & Poo (1998) |

## Experimental sources

- Guo-Qiang Bi and Huai-Xing Wang. “Temporal asymmetry in spike
  timing-dependent synaptic plasticity.” *Physiology & Behavior* 77 (2002),
  551–555. [doi:10.1016/S0031-9384(02)00933-2](https://doi.org/10.1016/S0031-9384(02)00933-2).
- Melanie A. Woodin, Karunesh Ganguly, and Mu-ming Poo. “Coincident pre- and
  postsynaptic activity modifies GABAergic synapses by postsynaptic changes in
  Cl- transporter activity.” *Neuron* 39 (2003), 807–820.
  [doi:10.1016/S0896-6273(03)00507-5](https://doi.org/10.1016/S0896-6273(03)00507-5).
- Guo-Qiang Bi and Mu-ming Poo. “Synaptic modifications in cultured
  hippocampal neurons: dependence on spike timing, synaptic strength, and
  postsynaptic cell type.” *Journal of Neuroscience* 18 (1998), 10464–10472.
  [doi:10.1523/JNEUROSCI.18-24-10464.1998](https://doi.org/10.1523/JNEUROSCI.18-24-10464.1998).

## Important variance-weighting provenance

The fixed Fig. 6 configurations were selected by the legacy notebooks. Those
notebooks did **not** implement the ±5 ms window described in Methods Section H
literally. They first sorted observations by `(delay, weight change)`, found
the first index equal to each delay, and calculated a population variance over
the slice `[index - 5:index + 5)`. Thus, “5” means five neighboring
observations per side (at most ten observations), the upper endpoint is
excluded, and duplicate delays all reuse the first duplicate's window.

`fit_stdp.py` preserves that behavior as the default `--weighting legacy`, so
the embedded parameters and their objective values remain comparable with the
original search. `--manuscript-weighting` (equivalent to `--weighting
manuscript`) instead uses observations whose delays fall strictly inside
`(delay - 5 ms, delay + 5 ms)`, as written in the manuscript. Several digitized
points are alone within that interval and therefore have zero variance. The
manuscript-mode implementation uses the explicitly reported
`--variance-floor` (default `1e-12`) to keep reciprocal weights finite; this
numerical floor is an additional implementation choice, and manuscript-mode
fits should report it.

The raw CSV files remain untouched in both modes.
