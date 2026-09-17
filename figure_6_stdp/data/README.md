# Digitized data for Figure 6i–j

- `Bi et al. 2002.csv`: Figure 6i. The script converts normalized final
  weights to percentage change with `(y - 1) * 100`.
- `Melanie et al. 2003.csv`: Figure 6j. Values are already percentage weight
  changes; the historical filename is retained for provenance.

These are digitized points from Bi and Wang (2002), Fig. 3, and Woodin,
Ganguly and Poo (2003), Fig. 2. The files were copied without numerical
alteration from the pinned source revision in `provenance.json`.

The original fits weight residuals by variance over neighboring observations;
the alternative Methods interpretation uses observations strictly within
±5 ms. See `README.md` and `fit_stdp.py` for the selectable definitions.
