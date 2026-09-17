"""Numerical PDM model used by the Figure 7 reproduction scripts.

The Euler update order matches the original ``predictive_dendrite``
notebooks. Times are in milliseconds and rates are in inverse milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class SimulationResult:
    """State recorded during one two-dendrite simulation."""

    time_ms: np.ndarray
    input_spikes: np.ndarray
    input_signal: np.ndarray
    dendritic_potential: np.ndarray
    somatic_potential: np.ndarray
    output_spikes: np.ndarray
    output_signal: np.ndarray
    weights: np.ndarray


def sigmoid_gate(
    output_signal: np.ndarray | float,
    plasticity_threshold: float = 0.02,
    gain: float = 200.0,
) -> np.ndarray | float:
    """Evaluate the plasticity gate with numerically stable exponentials."""

    z = np.clip(
        gain * (np.asarray(output_signal) - plasticity_threshold), -700.0, 700.0
    )
    result = 1.0 / (1.0 + np.exp(-z))
    return float(result) if np.ndim(result) == 0 else result


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
    """Simulate the historical two-dendrite Euler discretisation."""

    spikes = np.asarray(input_spikes, dtype=float)
    if spikes.ndim != 2 or spikes.shape[0] != 2 or spikes.shape[1] < 2:
        raise ValueError("input_spikes must have shape (2, n_steps), n_steps >= 2")
    if dt_ms <= 0 or decay_rate <= 0 or gate_gain <= 0:
        raise ValueError("dt_ms, decay_rate, and gate_gain must be positive")
    if transmission_delay_ms < 0:
        raise ValueError("transmission_delay_ms must be non-negative")
    if not 0.0 <= dendritic_mix <= 1.0:
        raise ValueError("dendritic_mix must lie between zero and one")

    n_steps = spikes.shape[1]
    delay_steps = int(transmission_delay_ms / dt_ms)
    if delay_steps >= n_steps - 1:
        raise ValueError("transmission delay must be shorter than the simulation")

    initial = np.asarray(initial_weights, dtype=float)
    if initial.shape != (2,):
        raise ValueError("initial_weights must contain exactly two values")
    rates = np.asarray(learning_rate, dtype=float)
    if rates.ndim == 0:
        rates = np.repeat(rates, 2)
    if rates.shape != (2,):
        raise ValueError("learning_rate must be scalar or contain two values")

    input_signal = np.zeros((2, n_steps), dtype=float)
    dendritic = np.zeros((2, n_steps), dtype=float)
    soma = np.zeros(n_steps, dtype=float)
    output_spikes = np.zeros(n_steps, dtype=float)
    output_signal = np.zeros(n_steps, dtype=float)
    weights = np.zeros((2, n_steps), dtype=float)
    weights[:, delay_steps] = initial
    forced = {
        int(round(float(spike_time) / dt_ms))
        for spike_time in forced_output_spike_times_ms
    }
    mix = np.array([1.0 - dendritic_mix, dendritic_mix])

    for step in range(delay_steps, n_steps - 1):
        delayed_step = step - delay_steps
        soma[step + 1] = soma[step] - dt_ms * decay_rate * soma[step]
        input_signal[:, step + 1] = (
            input_signal[:, step]
            + spikes[:, step]
            - dt_ms * decay_rate * input_signal[:, step]
        )
        instantaneous = weights[:, step] * spikes[:, delayed_step]
        dendritic[:, step + 1] = (
            dendritic[:, step] + instantaneous - dt_ms * decay_rate * dendritic[:, step]
        )
        soma[step + 1] += np.dot(mix, instantaneous)
        error = output_signal[step] - dendritic[:, step]
        weights[:, step + 1] = weights[:, step] + rates * dt_ms * error * input_signal[
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
        if soma[step + 1] > spike_threshold:
            output_spikes[step + 1] = 1.0
            soma[step + 1] = reset_potential
        if step + 1 in forced:
            output_spikes[step + 1] = 1.0
        elif suppress_spontaneous_spikes:
            output_spikes[step + 1] = 0.0

    return SimulationResult(
        time_ms=np.arange(n_steps, dtype=float) * dt_ms,
        input_spikes=spikes.copy(),
        input_signal=input_signal,
        dendritic_potential=dendritic,
        somatic_potential=soma,
        output_spikes=output_spikes,
        output_signal=output_signal,
        weights=weights,
    )


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
) -> float | tuple[float, SimulationResult]:
    """Run one timing protocol; positive delay means pre before post."""

    if n_post < 1 or n_pre < 1 or initial_weight <= 0:
        raise ValueError("spike counts and initial_weight must be positive")
    duration_ms = 100.0 + 2.0 * abs(float(delay_ms))
    n_steps = int(duration_ms / dt_ms)
    first_post_ms = int(duration_ms / 2.0)
    post_times_ms = [
        int(first_post_ms + post_isi_ms * index) for index in range(n_post)
    ]
    spikes = np.zeros((2, n_steps), dtype=float)
    for index in range(n_pre):
        pre_time_ms = first_post_ms + post_isi_ms * index - float(delay_ms)
        pre_index = int(pre_time_ms / dt_ms)
        if not 0 <= pre_index < n_steps:
            raise ValueError("pre-synaptic spike falls outside the simulation")
        spikes[0, pre_index] = 1.0

    simulation = simulate_two_dendrites(
        spikes,
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
    change = float(simulation.weights[0, -1] - initial_weight)
    return (change, simulation) if return_trace else change


def stdp_curve(
    delays_ms: Sequence[float] | np.ndarray,
    *,
    percent_change: bool = False,
    **stdp_kwargs: float,
) -> np.ndarray:
    """Return the STDP curve for a one-dimensional delay array."""

    delays = np.asarray(delays_ms, dtype=float)
    if delays.ndim != 1:
        raise ValueError("delays_ms must be one-dimensional")
    changes = np.asarray(
        [float(stdp_single(float(delay), **stdp_kwargs)) for delay in delays]
    )
    if percent_change:
        changes = 100.0 * changes / float(stdp_kwargs.get("initial_weight", 0.5))
    return changes


__all__ = [
    "SimulationResult",
    "sigmoid_gate",
    "simulate_two_dendrites",
    "stdp_single",
    "stdp_curve",
]
