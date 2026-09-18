#!/usr/bin/env python3
"""Minimal spiking XOR dynamics for Supplementary Figure 3d.

The equations are a self-contained migration of
``experiments/spiking/simulation_XOR.py``.  Arrays supplied by callers are
copied, so one simulation is a deterministic state transition and never
mutates its inputs.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

DEFAULT_DT = 0.1
DEFAULT_TMAX = 50.0
DEFAULT_DELAY = 5.0
DEFAULT_DECAY = 0.1
DEFAULT_THRESHOLD = 0.5
DEFAULT_RESET = -0.5
DEFAULT_LEARNING_RATE = 0.005


def plasticity_gate(x: np.ndarray | float, epsilon: float = 0.02):
    """Sigmoidal gate controlling whether a postsynaptic trace is plastic."""

    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-200.0 * (np.asarray(x) - epsilon)))


def spike_train(
    delay: float,
    dt: float,
    tmax: float,
    *,
    delay_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate the regular source spike train with ISI ``2 * delay``."""

    delay = float(delay)
    dt = float(dt)
    tmax = float(tmax)
    delay_std = float(delay_std)
    if delay <= 0.0 or dt <= 0.0 or tmax <= 0.0:
        raise ValueError("delay, dt, and tmax must be positive")
    if delay_std < 0.0:
        raise ValueError("delay_std must be non-negative")
    if delay_std > 0.0:
        if rng is None:
            rng = np.random.default_rng(0)
        delay += float(rng.normal(0.0, delay_std))
        if delay <= 0.0:
            raise ValueError("jittered delay must remain positive")

    n_steps = int(tmax / dt)
    inter_spike_interval = 2.0 * delay
    spikes = np.zeros(n_steps, dtype=np.float64)
    for spike_index in range(int(tmax / inter_spike_interval)):
        time_index = int(spike_index * inter_spike_interval / dt)
        if time_index < n_steps:
            spikes[time_index] = 1.0
    return spikes


def initialize_weights(
    hidden_size: int,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialize weights exactly as in the source spiking XOR script."""

    hidden_size = int(hidden_size)
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if rng is None:
        rng = np.random.default_rng(0)
    forward_input = 4.0 * rng.random((hidden_size, 2)) - 2.0
    forward_output = 5.0 * np.ones((1, hidden_size), dtype=np.float64) / hidden_size
    backward = np.ones((hidden_size, 1), dtype=np.float64)
    return forward_input, forward_output, backward


def simulate(
    inputs: Sequence[float],
    target: float,
    forward_input: np.ndarray,
    forward_output: np.ndarray,
    backward: np.ndarray,
    train: bool,
    theta: float,
    *,
    dt: float = DEFAULT_DT,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    backward_learning_rate: float = DEFAULT_LEARNING_RATE,
    output_delay_std: float = 0.0,
    tmax: float = DEFAULT_TMAX,
    delay: float = DEFAULT_DELAY,
    decay: float = DEFAULT_DECAY,
    threshold: float = DEFAULT_THRESHOLD,
    reset_potential: float = DEFAULT_RESET,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Simulate one XOR example and return updated weights and diagnostics."""

    inputs_array = np.asarray(inputs, dtype=np.float64)
    if inputs_array.shape != (2,):
        raise ValueError("inputs must contain exactly two values")
    theta = float(theta)
    dt = float(dt)
    tmax = float(tmax)
    delay = float(delay)
    decay = float(decay)
    threshold = float(threshold)
    reset_potential = float(reset_potential)
    learning_rate = float(learning_rate)
    backward_learning_rate = float(backward_learning_rate)
    if not 0.0 <= theta <= 1.0:
        raise ValueError("theta must be in [0, 1]")
    if dt <= 0.0 or tmax <= 0.0 or delay <= 0.0:
        raise ValueError("dt, tmax, and delay must be positive")
    if learning_rate < 0.0 or backward_learning_rate < 0.0:
        raise ValueError("learning rates must be non-negative")

    forward_input = np.asarray(forward_input, dtype=np.float64).copy()
    if forward_input.ndim != 2 or forward_input.shape[1] != 2:
        raise ValueError("forward_input must have shape (hidden_size, 2)")
    hidden_size = forward_input.shape[0]
    forward_output = np.asarray(forward_output, dtype=np.float64).reshape(-1).copy()
    backward = np.asarray(backward, dtype=np.float64).reshape(-1).copy()
    if forward_output.shape != (hidden_size,) or backward.shape != (hidden_size,):
        raise ValueError("forward_output and backward must match hidden_size")

    n_steps = int(tmax / dt)
    transmission_steps = int(delay / dt)
    if transmission_steps <= 0 or n_steps <= transmission_steps + 1:
        raise ValueError("simulation window must exceed the transmission delay")
    if rng is None:
        rng = np.random.default_rng(0)

    base_spikes = spike_train(delay, dt, tmax, rng=rng)
    input_spikes = inputs_array[:, None] * base_spikes[None, :]
    output_spikes = float(target) * spike_train(
        delay,
        dt,
        tmax,
        delay_std=float(output_delay_std),
        rng=rng,
    )
    target_spikes = output_spikes.copy()

    hidden_spikes = np.zeros((hidden_size, n_steps), dtype=np.float64)
    hidden_voltage = np.zeros((hidden_size, n_steps), dtype=np.float64)
    hidden_trace = np.zeros((hidden_size, n_steps), dtype=np.float64)
    hidden_dendrite = np.zeros((hidden_size, 2, n_steps), dtype=np.float64)
    input_trace = np.zeros((hidden_size, 2, n_steps), dtype=np.float64)
    feedback_trace = np.zeros((hidden_size, n_steps), dtype=np.float64)

    output_voltage = np.zeros(n_steps, dtype=np.float64)
    output_trace = np.zeros(n_steps, dtype=np.float64)
    output_dendrite = np.zeros(n_steps, dtype=np.float64)
    hidden_to_output_trace = np.zeros((hidden_size, n_steps), dtype=np.float64)

    train_scale = float(bool(train))
    hidden_lr = learning_rate * train_scale
    feedback_lr = backward_learning_rate * train_scale
    output_lr = learning_rate * train_scale

    for time_index in range(transmission_steps, n_steps - 1):
        delayed_input_spikes = input_spikes[:, time_index - transmission_steps]
        delayed_output_spike = output_spikes[time_index - transmission_steps]

        input_trace[:, :, time_index + 1] = (
            input_trace[:, :, time_index]
            + input_spikes[:, time_index][None, :]
            - dt * decay * input_trace[:, :, time_index]
        )
        feedback_trace[:, time_index + 1] = (
            feedback_trace[:, time_index]
            + output_spikes[time_index]
            - dt * decay * feedback_trace[:, time_index]
        )

        forward_current = np.sum(forward_input * delayed_input_spikes[None, :], axis=1)
        feedback_current = backward * delayed_output_spike
        hidden_dendrite[:, 0, time_index + 1] = (
            hidden_dendrite[:, 0, time_index]
            + forward_current
            - dt * decay * hidden_dendrite[:, 0, time_index]
        )
        hidden_dendrite[:, 1, time_index + 1] = (
            hidden_dendrite[:, 1, time_index]
            + feedback_current
            - dt * decay * hidden_dendrite[:, 1, time_index]
        )
        hidden_voltage[:, time_index + 1] = (
            hidden_voltage[:, time_index]
            - dt * decay * hidden_voltage[:, time_index]
            + (1.0 - theta) * forward_current
            + theta * feedback_current
        )

        gate = plasticity_gate(hidden_trace[:, time_index], epsilon=0.05)
        forward_input += (
            hidden_lr
            * dt
            * (hidden_trace[:, time_index] - hidden_dendrite[:, 0, time_index])[:, None]
            * input_trace[:, :, time_index - transmission_steps]
            * gate[:, None]
        )
        backward += (
            feedback_lr
            * dt
            * (hidden_trace[:, time_index] - hidden_dendrite[:, 1, time_index])
            * feedback_trace[:, time_index - transmission_steps]
            * gate
        )

        hidden_trace[:, time_index + 1] = (
            hidden_trace[:, time_index]
            + hidden_spikes[:, time_index]
            - dt * decay * hidden_trace[:, time_index]
        )
        hidden_spikes_next = hidden_voltage[:, time_index + 1] > threshold
        hidden_spikes[hidden_spikes_next, time_index + 1] = 1.0
        hidden_voltage[hidden_spikes_next, time_index + 1] = reset_potential

        hidden_to_output_trace[:, time_index + 1] = (
            hidden_to_output_trace[:, time_index]
            + hidden_spikes[:, time_index]
            - dt * decay * hidden_to_output_trace[:, time_index]
        )
        output_current = np.sum(
            forward_output * hidden_spikes[:, time_index - transmission_steps]
        )
        output_dendrite[time_index + 1] = (
            output_dendrite[time_index]
            + output_current
            - dt * decay * output_dendrite[time_index]
        )
        output_voltage[time_index + 1] = (
            output_voltage[time_index]
            - dt * decay * output_voltage[time_index]
            + output_current
        )
        forward_output += (
            output_lr
            * dt
            * (output_trace[time_index] - output_dendrite[time_index])
            * hidden_to_output_trace[:, time_index - transmission_steps]
            * plasticity_gate(output_trace[time_index], epsilon=-1.0)
        )

        output_trace[time_index + 1] = (
            output_trace[time_index]
            + output_spikes[time_index]
            - dt * decay * output_trace[time_index]
        )
        if output_voltage[time_index + 1] > threshold:
            output_spikes[time_index + 1] = 1.0
            output_voltage[time_index + 1] = reset_potential
        if train:
            output_spikes[time_index + 1] = target_spikes[time_index + 1]

    output_rate = float(np.sum(output_spikes[1:-1]) / (tmax / (delay * 2.0) - 1.0))
    dendritic_error = float(np.mean((output_trace - output_dendrite) ** 2))
    return (
        forward_input,
        forward_output,
        backward[:, None],
        output_rate,
        dendritic_error,
    )


__all__ = [
    "DEFAULT_DT",
    "DEFAULT_TMAX",
    "DEFAULT_DELAY",
    "DEFAULT_DECAY",
    "DEFAULT_THRESHOLD",
    "DEFAULT_RESET",
    "DEFAULT_LEARNING_RATE",
    "plasticity_gate",
    "spike_train",
    "initialize_weights",
    "simulate",
]
