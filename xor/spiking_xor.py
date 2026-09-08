#!/usr/bin/env python3
"""Train spiking XOR models and write results/spiking.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = SCRIPT_DIR / "results" / "spiking.csv"
RNG_SEED = 4
THETA = 0.4
HIDDEN_SIZE = 128
N_STEPS = 20
N_REPEATS = 10
MODELS = ("predictive_dendrites", "classic_stdp")


def g_local(x, epsilon=0.02):
    return 1 / (1 + np.exp(-200 * (x - epsilon)))


def spike_train_local(delay, dt, tmax, delay_std=0.0):
    nstep = int(tmax / dt)
    if delay_std > 0:
        delay += np.random.normal(0, delay_std)
    isi = 2 * delay
    spikes = np.zeros(nstep)
    for i in range(int(tmax / isi)):
        spikes[int(i * isi / dt)] = 1
    return spikes


def initialize_weights_local(nhid):
    wf1 = 4 * np.random.rand(nhid, 2) - 2
    wf2 = 5 * np.ones((1, nhid)) / nhid
    wb = np.ones((nhid, 1))
    return wf1, wf2, wb


def simulate_local(
    xin,
    xout,
    wf1,
    wf2,
    wb,
    train,
    theta,
    dt=0.1,
    lr=0.005,
    lr_w2b=0.005,
    spikes_output_delay_std=0.0,
):
    epsilon = 0.05
    tmax = 50
    nstep = int(tmax / dt)
    delay = 5
    tau = int(delay / dt)
    decay = 0.1
    threshold = 0.5
    vreset = -0.5
    nhid = len(wb)

    wf1 = np.asarray(wf1, dtype=float).copy()
    wf2 = np.asarray(wf2, dtype=float).reshape(-1).copy()
    wb = np.asarray(wb, dtype=float).reshape(-1).copy()

    spikes_input = np.stack(
        [xin[i] * spike_train_local(delay, dt, tmax) for i in range(2)],
        axis=0,
    )
    spikes_output = xout * spike_train_local(
        delay, dt, tmax, delay_std=spikes_output_delay_std
    )
    spikes_target = np.copy(spikes_output)

    spikes_hidden = np.zeros((nhid, nstep))
    v_hidden = np.zeros((nhid, nstep))
    x_hidden = np.zeros((nhid, nstep))
    vi_hidden = np.zeros((nhid, 2, nstep))
    xi_input = np.zeros((nhid, 2, nstep))
    xi_feedback = np.zeros((nhid, nstep))

    v_output = np.zeros(nstep)
    x_output = np.zeros(nstep)
    vi_output = np.zeros(nstep)
    xi_output = np.zeros((nhid, nstep))

    hidden_lr = lr * train
    feedback_lr = lr_w2b * train
    output_lr = lr * train

    for t in range(tau, nstep - 1):
        delayed_input_spikes = spikes_input[:, t - tau]
        delayed_output_spike = spikes_output[t - tau]

        xi_input[:, :, t + 1] = (
            xi_input[:, :, t]
            + spikes_input[:, t][None, :]
            - dt * decay * xi_input[:, :, t]
        )
        xi_feedback[:, t + 1] = (
            xi_feedback[:, t] + spikes_output[t] - dt * decay * xi_feedback[:, t]
        )

        hidden_input_forward = np.sum(wf1 * delayed_input_spikes[None, :], axis=1)
        hidden_input_feedback = wb * delayed_output_spike

        vi_hidden[:, 0, t + 1] = (
            vi_hidden[:, 0, t]
            + hidden_input_forward
            - dt * decay * vi_hidden[:, 0, t]
        )
        vi_hidden[:, 1, t + 1] = (
            vi_hidden[:, 1, t]
            + hidden_input_feedback
            - dt * decay * vi_hidden[:, 1, t]
        )

        v_hidden[:, t + 1] = (
            v_hidden[:, t]
            - dt * decay * v_hidden[:, t]
            + hidden_input_forward * (1 - theta)
            + hidden_input_feedback * theta
        )

        hidden_gate = g_local(x_hidden[:, t], epsilon)
        wf1 = wf1 + (
            hidden_lr
            * dt
            * (x_hidden[:, t] - vi_hidden[:, 0, t])[:, None]
            * xi_input[:, :, t - tau]
            * hidden_gate[:, None]
        )
        wb = wb + (
            feedback_lr
            * dt
            * (x_hidden[:, t] - vi_hidden[:, 1, t])
            * xi_feedback[:, t - tau]
            * hidden_gate
        )

        x_hidden[:, t + 1] = (
            x_hidden[:, t] + spikes_hidden[:, t] - dt * decay * x_hidden[:, t]
        )
        hidden_spikes_next = v_hidden[:, t + 1] > threshold
        spikes_hidden[hidden_spikes_next, t + 1] = 1
        v_hidden[hidden_spikes_next, t + 1] = vreset

        xi_output[:, t + 1] = (
            xi_output[:, t] + spikes_hidden[:, t] - dt * decay * xi_output[:, t]
        )
        output_input = np.sum(wf2 * spikes_hidden[:, t - tau])
        vi_output[t + 1] = vi_output[t] + output_input - dt * decay * vi_output[t]
        v_output[t + 1] = v_output[t] - dt * decay * v_output[t] + output_input

        output_gate = g_local(x_output[t], -1)
        wf2 = wf2 + (
            output_lr
            * dt
            * (x_output[t] - vi_output[t])
            * xi_output[:, t - tau]
            * output_gate
        )

        x_output[t + 1] = x_output[t] + spikes_output[t] - dt * decay * x_output[t]
        if v_output[t + 1] > threshold:
            spikes_output[t + 1] = 1
            v_output[t + 1] = vreset
        if train:
            spikes_output[t + 1] = spikes_target[t + 1]

    output = np.sum(spikes_output[1:-1]) / (tmax / (delay * 2) - 1)
    error = np.mean((x_output - vi_output) ** 2)
    return wf1, wf2, wb[:, None], output, error


def simulate_local_online_stdp(
    xin,
    xout,
    wf1,
    wf2,
    wb,
    train,
    theta,
    dt=0.1,
    lr=0.005,
    lr_w2b=0.005,
    spikes_output_delay_std=0.0,
    stdp_tau_plus=10.0,
    stdp_tau_minus=10.0,
    stdp_a_plus=1.0,
    stdp_a_minus=1.0,
):
    """
    Online STDP implementation:
    - Presynaptic trace x_j follows Eq. (1.4)
    - Postsynaptic trace y follows Eq. (1.5)
    - Weight dynamics follow Eq. (1.6), discretized with Euler steps.
    """
    tmax = 50
    nstep = int(tmax / dt)
    delay = 5
    tau = int(delay / dt)
    decay = 0.1
    threshold = 0.5
    vreset = -0.5
    nhid = len(wb)

    wf1 = np.asarray(wf1, dtype=float).copy()
    wf2 = np.asarray(wf2, dtype=float).reshape(-1).copy()
    wb = np.asarray(wb, dtype=float).reshape(-1).copy()

    spikes_input = np.stack(
        [xin[i] * spike_train_local(delay, dt, tmax) for i in range(2)],
        axis=0,
    )
    spikes_output = xout * spike_train_local(
        delay, dt, tmax, delay_std=spikes_output_delay_std
    )
    spikes_target = np.copy(spikes_output)

    spikes_hidden = np.zeros((nhid, nstep))
    v_hidden = np.zeros((nhid, nstep))
    x_hidden = np.zeros((nhid, nstep))
    vi_hidden = np.zeros((nhid, 2, nstep))

    v_output = np.zeros(nstep)
    x_output = np.zeros(nstep)
    vi_output = np.zeros(nstep)

    # STDP traces:
    # Eq (1.4): x_j traces for presynaptic spike arrivals (input and hidden presyn)
    x_pre_input = np.zeros(2)
    x_pre_hidden = np.zeros(nhid)
    x_pre_feedback = 0.0  # output -> hidden presynaptic trace
    # Eq (1.5): y traces for postsynaptic spikes (hidden and output)
    y_hidden = np.zeros(nhid)
    y_output = 0.0

    hidden_lr = lr * train
    feedback_lr = lr_w2b * train
    output_lr = lr * train

    for t in range(tau, nstep - 1):
        delayed_input_spikes = spikes_input[:, t - tau]
        delayed_output_spike = spikes_output[t - tau]

        # Synaptic filtering / membrane dynamics (kept identical to baseline).
        hidden_input_forward = np.sum(wf1 * delayed_input_spikes[None, :], axis=1)
        hidden_input_feedback = wb * delayed_output_spike

        vi_hidden[:, 0, t + 1] = (
            vi_hidden[:, 0, t]
            + hidden_input_forward
            - dt * decay * vi_hidden[:, 0, t]
        )
        vi_hidden[:, 1, t + 1] = (
            vi_hidden[:, 1, t]
            + hidden_input_feedback
            - dt * decay * vi_hidden[:, 1, t]
        )
        v_hidden[:, t + 1] = (
            v_hidden[:, t]
            - dt * decay * v_hidden[:, t]
            + hidden_input_forward * (1 - theta)
            + hidden_input_feedback * theta
        )

        x_hidden[:, t + 1] = (
            x_hidden[:, t] + spikes_hidden[:, t] - dt * decay * x_hidden[:, t]
        )
        hidden_spikes_next = v_hidden[:, t + 1] > threshold
        spikes_hidden[hidden_spikes_next, t + 1] = 1
        v_hidden[hidden_spikes_next, t + 1] = vreset

        # Output dynamics
        delayed_hidden_spikes = spikes_hidden[:, t - tau]
        output_input = np.sum(wf2 * delayed_hidden_spikes)
        vi_output[t + 1] = vi_output[t] + output_input - dt * decay * vi_output[t]
        v_output[t + 1] = v_output[t] - dt * decay * v_output[t] + output_input

        x_output[t + 1] = x_output[t] + spikes_output[t] - dt * decay * x_output[t]
        if v_output[t + 1] > threshold:
            spikes_output[t + 1] = 1
            v_output[t + 1] = vreset
        if train:
            spikes_output[t + 1] = spikes_target[t + 1]

        # --- Online STDP updates (Eqs. 1.4-1.6) ---
        # Update traces with spike *arrivals* at synapses.
        # Eq (1.4): presynaptic traces
        x_pre_input += (-x_pre_input / stdp_tau_plus) * dt
        x_pre_input += stdp_a_plus * delayed_input_spikes

        x_pre_hidden += (-x_pre_hidden / stdp_tau_plus) * dt
        x_pre_hidden += stdp_a_plus * delayed_hidden_spikes

        x_pre_feedback += (-x_pre_feedback / stdp_tau_plus) * dt
        x_pre_feedback += stdp_a_plus * delayed_output_spike

        # Eq (1.5): postsynaptic traces
        y_hidden += (-y_hidden / stdp_tau_minus) * dt
        y_hidden += stdp_a_minus * spikes_hidden[:, t]

        y_output += (-y_output / stdp_tau_minus) * dt
        y_output += stdp_a_minus * spikes_output[t]

        # Eq (1.6): weight increments at postsynaptic spikes (LTP term).
        # Hidden layer: when hidden neuron spikes at time t, potentiate wf1 and wb.
        if train:
            hidden_spikes_now = spikes_hidden[:, t]
            if np.any(hidden_spikes_now):
                wf1 += hidden_lr * (hidden_spikes_now[:, None] * x_pre_input[None, :])
                wb += feedback_lr * (hidden_spikes_now * x_pre_feedback)

            # Output layer: when output spikes at time t, potentiate wf2.
            if spikes_output[t] > 0:
                wf2 += output_lr * x_pre_hidden

        # Eq (1.6): weight decrements at presynaptic spike arrivals (LTD term).
        if train:
            # Input -> hidden depression at input spike arrivals.
            if np.any(delayed_input_spikes):
                wf1 -= hidden_lr * (y_hidden[:, None] * delayed_input_spikes[None, :])

            # Output -> hidden depression at output spike arrival.
            if delayed_output_spike > 0:
                wb -= feedback_lr * y_hidden

            # Hidden -> output depression at hidden spike arrivals.
            if np.any(delayed_hidden_spikes):
                wf2 -= output_lr * (y_output * delayed_hidden_spikes)

    output = np.sum(spikes_output[1:-1]) / (tmax / (delay * 2) - 1)
    error = np.mean((x_output - vi_output) ** 2)
    return wf1, wf2[None, :], wb[:, None], output, error


def train_one_model(
    model: str,
    *,
    theta: float,
    hidden_size: int,
    n_steps: int,
    n_repeats: int,
) -> list[dict]:
    # Reset per model so each run matches the original independent Ray trials.
    np.random.seed(RNG_SEED)
    simulate_fn = (
        simulate_local_online_stdp if model == "classic_stdp" else simulate_local
    )
    train_in = [[0, 0], [0, 1], [1, 0], [1, 1]]
    train_out = [0, 1, 1, 0]
    rows: list[dict] = []

    for rep in range(n_repeats):
        wf1, wf2, wb = initialize_weights_local(hidden_size)
        for step in range(n_steps):
            for ex in range(len(train_out)):
                wf1, wf2, wb, _, _ = simulate_fn(
                    train_in[ex],
                    train_out[ex],
                    wf1,
                    wf2,
                    wb,
                    1,
                    theta,
                    spikes_output_delay_std=0.0,
                )

            spike_error = 0.0
            for ex in range(len(train_out)):
                wf1, wf2, wb, output, _ = simulate_fn(
                    train_in[ex],
                    0,
                    wf1,
                    wf2,
                    wb,
                    0,
                    theta,
                )
                spike_error += (train_out[ex] - output) ** 2

            rows.append(
                {
                    "repeat": rep,
                    "step": step,
                    "model": model,
                    "mse": spike_error / len(train_out),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train spiking XOR models and write a long-form CSV."
    )
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    parser.add_argument("--theta", type=float, default=THETA)
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--n-steps", type=int, default=N_STEPS)
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict] = []
    for model in MODELS:
        print(f"Training model={model}")
        rows.extend(
            train_one_model(
                model,
                theta=args.theta,
                hidden_size=args.hidden_size,
                n_steps=args.n_steps,
                n_repeats=args.n_repeats,
            )
        )

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["repeat", "step", "model", "mse"])
    df.to_csv(args.results_path, index=False)
    print(f"Wrote {len(df)} rows to {args.results_path}")


if __name__ == "__main__":
    main()
