"""Numerical model used by the spiking PDM figure scripts.

The implementation follows Eqs. 18--28 of the manuscript and intentionally
keeps the update ordering of the original ``predictive_dendrite`` notebooks.
In particular, Euler updates use values at time ``t`` and weights are updated
after the somatic and dendritic potentials for ``t + dt`` have been computed.

The public Python names spell out the manuscript parameters:

``decay_rate`` (lambda), ``transmission_delay_ms`` (tau),
``learning_rate`` (alpha), and ``plasticity_threshold`` (gamma).

Historical keyword aliases from the research notebooks are accepted by the
STDP helpers so that saved configurations remain usable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SimulationResult:
    """Complete state of a two-dendrite simulation."""

    time_ms: np.ndarray
    input_spikes: np.ndarray
    input_signal: np.ndarray
    dendritic_potential: np.ndarray
    somatic_potential: np.ndarray
    output_spikes: np.ndarray
    output_signal: np.ndarray
    weights: np.ndarray

    @property
    def dendritic_error(self) -> np.ndarray:
        """Return epsilon^d = x^out - v^d for every dendrite and time."""

        return self.output_signal[np.newaxis, :] - self.dendritic_potential

    @property
    def energy(self) -> np.ndarray:
        """Return the somato-dendritic mismatch energy over time."""

        return 0.5 * np.sum(self.dendritic_error**2, axis=0)


def sigmoid_gate(
    output_signal: np.ndarray | float,
    plasticity_threshold: float = 0.02,
    gain: float = 200.0,
) -> np.ndarray | float:
    """Evaluate ``g(x_out - gamma)`` with numerically stable exponentials."""

    z = np.clip(
        gain * (np.asarray(output_signal) - plasticity_threshold),
        -700.0,
        700.0,
    )
    result = 1.0 / (1.0 + np.exp(-z))
    if np.ndim(result) == 0:
        return float(result)
    return result


def simulate_two_dendrites(
    input_spikes: np.ndarray,
    initial_weights: Sequence[float] = (0.6, 0.3),
    *,
    dt_ms: float = 0.1,
    transmission_delay_ms: float = 5.0,
    decay_rate: float = 0.1,
    plasticity_threshold: float = 0.02,
    learning_rate: float | Sequence[float] = 0.01,
    forced_output_spike_times_ms: Iterable[float] = (),
    suppress_spontaneous_spikes: bool = False,
    gate_gain: float = 200.0,
    dendritic_mix: float = 0.5,
    spike_threshold: float = 0.5,
    reset_potential: float = -0.5,
) -> SimulationResult:
    """Simulate the historical two-dendrite Euler discretisation.

    Parameters are expressed in milliseconds and inverse milliseconds where
    appropriate. ``input_spikes`` must have shape ``(2, n_steps)``. A value of
    one represents a spike impulse, as in the original implementation.

    ``dendritic_mix`` is the fraction of somatic input assigned to dendrite 2;
    dendrite 1 receives ``1 - dendritic_mix``. The manuscript figures use 0.5.
    """

    input_spikes = np.asarray(input_spikes, dtype=float)
    if input_spikes.ndim != 2 or input_spikes.shape[0] != 2:
        raise ValueError("input_spikes must have shape (2, n_steps)")
    if input_spikes.shape[1] < 2:
        raise ValueError("input_spikes must contain at least two time steps")
    if dt_ms <= 0:
        raise ValueError("dt_ms must be positive")
    if transmission_delay_ms < 0:
        raise ValueError("transmission_delay_ms must be non-negative")
    if decay_rate <= 0:
        raise ValueError("decay_rate must be positive")
    if gate_gain <= 0:
        raise ValueError("gate_gain must be positive")
    if not 0.0 <= dendritic_mix <= 1.0:
        raise ValueError("dendritic_mix must lie between zero and one")

    n_steps = input_spikes.shape[1]
    delay_steps = int(transmission_delay_ms / dt_ms)
    if delay_steps >= n_steps - 1:
        raise ValueError("transmission delay must be shorter than the simulation")

    initial_weights_array = np.asarray(initial_weights, dtype=float)
    if initial_weights_array.shape != (2,):
        raise ValueError("initial_weights must contain exactly two values")

    learning_rates = np.asarray(learning_rate, dtype=float)
    if learning_rates.ndim == 0:
        learning_rates = np.repeat(learning_rates, 2)
    if learning_rates.shape != (2,):
        raise ValueError("learning_rate must be a scalar or contain two values")

    time_ms = np.arange(n_steps, dtype=float) * dt_ms
    input_signal = np.zeros((2, n_steps), dtype=float)
    dendritic_potential = np.zeros((2, n_steps), dtype=float)
    somatic_potential = np.zeros(n_steps, dtype=float)
    output_spikes = np.zeros(n_steps, dtype=float)
    output_signal = np.zeros(n_steps, dtype=float)
    weights = np.zeros((2, n_steps), dtype=float)
    weights[:, delay_steps] = initial_weights_array

    forced_indices = {
        int(round(float(spike_time) / dt_ms))
        for spike_time in forced_output_spike_times_ms
    }
    mix = np.array([1.0 - dendritic_mix, dendritic_mix])

    for step in range(delay_steps, n_steps - 1):
        delayed_step = step - delay_steps

        somatic_potential[step + 1] = (
            somatic_potential[step] - dt_ms * decay_rate * somatic_potential[step]
        )
        input_signal[:, step + 1] = (
            input_signal[:, step]
            + input_spikes[:, step]
            - dt_ms * decay_rate * input_signal[:, step]
        )

        instantaneous_input = weights[:, step] * input_spikes[:, delayed_step]
        dendritic_potential[:, step + 1] = (
            dendritic_potential[:, step]
            + instantaneous_input
            - dt_ms * decay_rate * dendritic_potential[:, step]
        )
        somatic_potential[step + 1] += np.dot(mix, instantaneous_input)

        error = output_signal[step] - dendritic_potential[:, step]
        weights[:, step + 1] = weights[
            :, step
        ] + learning_rates * dt_ms * error * input_signal[
            :, delayed_step
        ] * sigmoid_gate(
            output_signal[step],
            plasticity_threshold=plasticity_threshold,
            gain=gate_gain,
        )

        output_signal[step + 1] = (
            output_signal[step]
            + output_spikes[step]
            - dt_ms * decay_rate * output_signal[step]
        )
        if somatic_potential[step + 1] > spike_threshold:
            output_spikes[step + 1] = 1.0
            somatic_potential[step + 1] = reset_potential

        if step + 1 in forced_indices:
            output_spikes[step + 1] = 1.0
        elif suppress_spontaneous_spikes:
            output_spikes[step + 1] = 0.0

    return SimulationResult(
        time_ms=time_ms,
        input_spikes=input_spikes.copy(),
        input_signal=input_signal,
        dendritic_potential=dendritic_potential,
        somatic_potential=somatic_potential,
        output_spikes=output_spikes,
        output_signal=output_signal,
        weights=weights,
    )


def _coerce_historical_keywords(
    canonical: Mapping[str, object], legacy_kwargs: Mapping[str, object]
) -> dict[str, object]:
    """Translate names used by the source notebooks into public API names."""

    values = dict(canonical)
    aliases = {
        "syndelay": "transmission_delay_ms",
        "decay": "decay_rate",
        "epsilon": "plasticity_threshold",
        "gamma": "plasticity_threshold",
        "lrate": "learning_rate",
        "initialw": "initial_weight",
        "npost": "n_post",
        "ISIpost": "post_isi_ms",
        "npre": "n_pre",
        "gain": "gate_gain",
    }
    unknown = set(legacy_kwargs).difference(aliases)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise TypeError(f"unexpected keyword argument(s): {names}")

    for old_name, value in legacy_kwargs.items():
        new_name = aliases[old_name]
        if old_name == "syndelay" and isinstance(value, str):
            value = float(value.replace("ms", "").strip())
        elif old_name == "decay" and isinstance(value, str):
            # Historical configs encoded a membrane time constant as "10 ms".
            time_constant_ms = float(value.replace("ms", "").strip())
            value = 1.0 / time_constant_ms
        elif isinstance(value, str):
            value = float(value)
        values[new_name] = value
    return values


def stdp_single(
    delay_ms: float,
    transmission_delay_ms: float = 5.0,
    decay_rate: float = 0.1,
    plasticity_threshold: float = 0.02,
    learning_rate: float = 0.01,
    initial_weight: float = 0.5,
    n_post: int = 1,
    post_isi_ms: float = 10.0,
    n_pre: int = 1,
    dt_ms: float = 0.1,
    gate_gain: float = 200.0,
    return_trace: bool = False,
    **legacy_kwargs: object,
) -> float | tuple[float, SimulationResult]:
    """Run one pre/post timing protocol and return the absolute weight change.

    Positive ``delay_ms`` means pre before post. With ``return_trace=True``,
    return ``(weight_change, SimulationResult)``. Historical aliases such as
    ``syndelay``, ``epsilon``, ``lrate`` and ``initialw`` are accepted.
    """

    values = _coerce_historical_keywords(
        {
            "transmission_delay_ms": transmission_delay_ms,
            "decay_rate": decay_rate,
            "plasticity_threshold": plasticity_threshold,
            "learning_rate": learning_rate,
            "initial_weight": initial_weight,
            "n_post": n_post,
            "post_isi_ms": post_isi_ms,
            "n_pre": n_pre,
            "gate_gain": gate_gain,
        },
        legacy_kwargs,
    )
    transmission_delay_ms = float(values["transmission_delay_ms"])
    decay_rate = float(values["decay_rate"])
    plasticity_threshold = float(values["plasticity_threshold"])
    learning_rate = float(values["learning_rate"])
    initial_weight = float(values["initial_weight"])
    n_post = int(values["n_post"])
    post_isi_ms = float(values["post_isi_ms"])
    n_pre = int(values["n_pre"])
    gate_gain = float(values["gate_gain"])

    if n_post < 1 or n_pre < 1:
        raise ValueError("n_post and n_pre must be positive")
    if initial_weight <= 0:
        raise ValueError("initial_weight must be positive")

    duration_ms = 100.0 + 2.0 * abs(float(delay_ms))
    n_steps = int(duration_ms / dt_ms)
    first_post_ms = int(duration_ms / 2.0)
    post_times_ms = [
        int(first_post_ms + post_isi_ms * index) for index in range(n_post)
    ]
    input_spikes = np.zeros((2, n_steps), dtype=float)
    for index in range(n_pre):
        pre_time_ms = first_post_ms + post_isi_ms * index - float(delay_ms)
        pre_index = int(pre_time_ms / dt_ms)
        if not 0 <= pre_index < n_steps:
            raise ValueError("pre-synaptic spike falls outside the simulation")
        input_spikes[0, pre_index] = 1.0

    simulation = simulate_two_dendrites(
        input_spikes,
        initial_weights=(initial_weight, initial_weight),
        dt_ms=dt_ms,
        transmission_delay_ms=transmission_delay_ms,
        decay_rate=decay_rate,
        plasticity_threshold=plasticity_threshold,
        learning_rate=learning_rate,
        forced_output_spike_times_ms=post_times_ms,
        suppress_spontaneous_spikes=True,
        gate_gain=gate_gain,
    )
    weight_change = float(simulation.weights[0, -1] - initial_weight)
    if return_trace:
        return weight_change, simulation
    return weight_change


def stdp_curve(
    delays_ms: Sequence[float] | np.ndarray,
    *,
    percent_change: bool = False,
    **stdp_kwargs: object,
) -> np.ndarray:
    """Return an STDP curve for ``delays_ms`` using :func:`stdp_single`."""

    delays = np.asarray(delays_ms, dtype=float)
    if delays.ndim != 1:
        raise ValueError("delays_ms must be one-dimensional")
    changes = np.array(
        [float(stdp_single(float(delay), **stdp_kwargs)) for delay in delays],
        dtype=float,
    )
    if percent_change:
        values = _coerce_historical_keywords(
            {"initial_weight": 0.5},
            {key: value for key, value in stdp_kwargs.items() if key in {"initialw"}},
        )
        initial_weight = float(
            stdp_kwargs.get("initial_weight", values["initial_weight"])
        )
        changes = 100.0 * changes / initial_weight
    return changes


__all__ = [
    "SimulationResult",
    "sigmoid_gate",
    "simulate_two_dendrites",
    "stdp_single",
    "stdp_curve",
]
