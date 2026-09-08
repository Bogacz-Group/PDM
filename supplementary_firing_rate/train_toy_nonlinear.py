"""
Train and compare PDM models on the nonlinear sum task y = ReLU(x1) + ReLU(x2)
"""

import os
import argparse
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import jpc
import optax

from utils import (
    normalize_generator_first_layer_weight_rows,
    normalize_amortiser_output_layer_weight_rows
)
from lateral_model import (
    lateral_pdm_relax_training_fixed_iters,
    lateral_pdm_init_weights,
    lateral_pdm_set_w2f_symmetric, 
    lateral_pdm_normalise_fwd_weights,
    lateral_pdm_normalise_bwd_weights,
    lateral_pdm_zero_w1b,
    lateral_pdm_mse_loss, 
    lateral_pdm_train_step
)


def generate_data(key, batch_size):
    """Generate data from y = ReLU(x1) + ReLU(x2) where x ~ Uniform(-1, 1)."""
    x = jr.uniform(key, (batch_size, 2), minval=-1.0, maxval=1.0)
    y = jnp.maximum(x[:, 0:1], 0) + jnp.maximum(x[:, 1:2], 0)
    return x, y


def generate_fixed_dataset(key, n_total=500, n_train=400):
    """
    Generate a fixed dataset and split into train/val, matching the reference setup:
    n_total samples, n_train for training, rest for validation.
    Returns (train_x, train_y, val_x, val_y) as JAX arrays.
    """
    key, data_key, shuffle_key = jr.split(key, 3)
    x_all = jr.uniform(data_key, (n_total, 2), minval=-1.0, maxval=1.0)
    y_all = jnp.maximum(x_all[:, 0:1], 0) + jnp.maximum(x_all[:, 1:2], 0)
    perm = jr.permutation(shuffle_key, n_total)
    x_all = x_all[perm]
    y_all = y_all[perm]
    train_x, val_x = x_all[:n_train], x_all[n_train:]
    train_y, val_y = y_all[:n_train], y_all[n_train:]
    return train_x, train_y, val_x, val_y


def sample_batch_from_fixed(key, train_x, train_y, batch_size):
    """Sample a random batch (with replacement) from fixed train data."""
    n_train = train_x.shape[0]
    indices = jr.choice(key, n_train, shape=(batch_size,), replace=True)
    return train_x[indices], train_y[indices]


def init_both_neurons_symmetrically(generator, w1, w2):
    """Initialize both hidden neurons with swapped pattern: neuron 0 [w1, w2], neuron 1 [w2, w1]."""
    current_weight = generator[0][1].weight
    new_weight = current_weight.at[0, :].set(jnp.array([w1, w2]))
    new_weight = new_weight.at[1, :].set(jnp.array([w2, w1]))
    return eqx.tree_at(lambda m: m[0][1].weight, generator, new_weight)


def get_first_neuron_weights(generator):
    return generator[0][1].weight[0, :]


def get_all_hidden_weights(generator):
    return generator[0][1].weight


def zero_w1b(amortiser):
    zero_weight = jnp.zeros_like(amortiser[0][1].weight)
    amortiser = eqx.tree_at(lambda m: m[0][1].weight, amortiser, zero_weight)
    return amortiser


def train_pdm(
    key,
    n_train_iters,
    batch_size,
    width,
    param_lr,
    infer_iters,
    activity_lr,
    normalise_fwd_weights,
    normalise_bwd_weights,
    zero_w1b_,
    print_every,
    save_dir,
    init_gen_weights,
    track_weight_history,
    track_activity_history,
    freeze_output_layer,
    fixed_data
):
    gen_key, amort_key, data_key = jr.split(key, 3)
    input_dim, output_dim = 2, 1
    n_hidden = 1
    generator = jpc.make_mlp(
        gen_key, 
        input_dim=input_dim, 
        width=width, 
        depth=n_hidden + 1,
        output_dim=output_dim, 
        act_fn="relu",
        use_bias=False
    )
    amortiser = jpc.make_mlp(
        amort_key,
        input_dim=output_dim, 
        width=width, 
        depth=n_hidden + 1,
        output_dim=input_dim, 
        act_fn="relu",
        use_bias=False
    )[::-1]

    if zero_w1b_:
        amortiser = zero_w1b(amortiser)
    if init_gen_weights is not None:
        generator = init_both_neurons_symmetrically(
            generator, init_gen_weights[0], init_gen_weights[1]
        )
    if freeze_output_layer:
        generator = eqx.tree_at(
            lambda m: m[-1][1].weight, generator, jnp.array([[1.0, 1.0]])
        )
        amortiser = eqx.tree_at(
            lambda m: m[-1][1].weight, amortiser, jnp.ones_like(amortiser[-1][1].weight)
        )
    initial_output_weights = jnp.array(generator[-1][1].weight) if freeze_output_layer else None
    initial_amort_input_weights = jnp.array(amortiser[-1][1].weight) if freeze_output_layer else None
    if normalise_fwd_weights:
        generator = normalize_generator_first_layer_weight_rows(generator)
    if normalise_bwd_weights:
        amortiser = normalize_amortiser_output_layer_weight_rows(amortiser)

    gen_optim = optax.adam(param_lr)
    amort_optim = optax.adam(param_lr)
    activity_optim = optax.sgd(activity_lr)
    gen_opt_state = gen_optim.init(eqx.filter(generator, eqx.is_array))
    amort_opt_state = amort_optim.init(eqx.filter(amortiser, eqx.is_array))

    train_losses = []
    val_losses = []
    gen_weight_history = [get_first_neuron_weights(generator)] if track_weight_history else None
    all_hidden_weights_history = [get_all_hidden_weights(generator)] if track_weight_history else None
    hidden_activity_history = [] if track_activity_history else None

    # Init metrics (before any param update)
    data_key, init_batch_key = jr.split(data_key)
    if fixed_data is not None:
        train_x, train_y, val_x, val_y = fixed_data
        x_init, y_init = sample_batch_from_fixed(
            init_batch_key, train_x, train_y, batch_size
        )
        x_val_init, y_val_init = val_x, val_y
    else:
        x_init, y_init = generate_data(init_batch_key, batch_size)
        data_key, val_key_init = jr.split(data_key)
        x_val_init, y_val_init = generate_data(val_key_init, 1000)

    gen_act_init = jpc.init_activities_with_ffwd(model=generator, input=x_init)
    train_losses.append(float(jpc.mse_loss(gen_act_init[-1], y_init)))
    val_act_init = jpc.init_activities_with_ffwd(model=generator, input=x_val_init)
    val_losses.append(float(jpc.mse_loss(val_act_init[-1], y_val_init)))

    if track_activity_history:
        activities = gen_act_init
        activity_opt_state = activity_optim.init(activities)
        for _ in range(infer_iters):
            res = jpc.update_pdm_activities(
                top_down_model=generator,
                bottom_up_model=amortiser,
                activities=activities,
                optim=activity_optim,
                opt_state=activity_opt_state,
                output=y_init,
                input=x_init,
            )
            activities = res["activities"]
            activity_opt_state = res["opt_state"]
        hidden_mean = jnp.mean(activities[1], axis=0)
        hidden_activity_history.append(np.array(hidden_mean))

    print(
        f"  Init     | Train loss: {train_losses[-1]:.4f} | Val. loss: {val_losses[-1]:.4f}"
    )

    for step in range(n_train_iters):
        data_key, batch_key = jr.split(data_key)
        if fixed_data is not None:
            train_x, train_y, val_x, val_y = fixed_data
            x_batch, y_batch = sample_batch_from_fixed(batch_key, train_x, train_y, batch_size)
        else:
            x_batch, y_batch = generate_data(batch_key, batch_size)
        gen_activities = jpc.init_activities_with_ffwd(model=generator, input=x_batch)
        train_loss = jpc.mse_loss(gen_activities[-1], y_batch)
        train_losses.append(float(train_loss))

        activities = gen_activities
        activity_opt_state = activity_optim.init(activities)
        for _ in range(infer_iters):
            res = jpc.update_pdm_activities(
                top_down_model=generator,
                bottom_up_model=amortiser,
                activities=activities,
                optim=activity_optim,
                opt_state=activity_opt_state,
                output=y_batch,
                input=x_batch,
            )
            activities = res["activities"]
            activity_opt_state = res["opt_state"]

        param_res = jpc.update_pdm_params(
            top_down_model=generator,
            bottom_up_model=amortiser,
            activities=activities,
            top_down_optim=gen_optim,
            bottom_up_optim=amort_optim,
            top_down_opt_state=gen_opt_state,
            bottom_up_opt_state=amort_opt_state,
            output=y_batch,
            input=x_batch,
        )
        generator, amortiser = param_res["models"]
        gen_opt_state, amort_opt_state = param_res["opt_states"]
        if freeze_output_layer and initial_output_weights is not None:
            generator = eqx.tree_at(lambda m: m[-1][1].weight, generator, initial_output_weights)
        if freeze_output_layer and initial_amort_input_weights is not None:
            amortiser = eqx.tree_at(lambda m: m[-1][1].weight, amortiser, initial_amort_input_weights)
        if zero_w1b_:
            amortiser = zero_w1b(amortiser)
        if normalise_fwd_weights:
            generator = normalize_generator_first_layer_weight_rows(generator)
        if normalise_bwd_weights:
            amortiser = normalize_amortiser_output_layer_weight_rows(amortiser)
        if track_weight_history:
            gen_weight_history.append(get_first_neuron_weights(generator))
            all_hidden_weights_history.append(get_all_hidden_weights(generator))
        if track_activity_history:
            # Record mean hidden-layer activities (over batch) at end of inference.
            # activities[1] corresponds to the single hidden layer (width = 2 in these experiments).
            hidden_mean = jnp.mean(activities[1], axis=0)
            hidden_activity_history.append(np.array(hidden_mean))

        if step % print_every == 0 or step == n_train_iters - 1:
            if fixed_data is not None:
                _, _, val_x, val_y = fixed_data
                x_val, y_val = val_x, val_y
            else:
                data_key, val_key = jr.split(data_key)
                x_val, y_val = generate_data(val_key, 1000)
            val_activities = jpc.init_activities_with_ffwd(model=generator, input=x_val)
            val_loss = jpc.mse_loss(val_activities[-1], y_val)
            val_losses.append(float(val_loss))
            print(f"  Step {step:4d} | Train loss: {train_loss:.4f} | Val. loss: {val_loss:.4f}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/train_losses.npy", np.array(train_losses))
        np.save(f"{save_dir}/val_losses.npy", np.array(val_losses))
        np.save(
            f"{save_dir}/n_train_iters.npy",
            np.array(n_train_iters, dtype=np.int64),
        )
        if track_weight_history and gen_weight_history is not None:
            np.save(f"{save_dir}/gen_weight_history.npy", np.array([np.array(w) for w in gen_weight_history]))
            np.save(f"{save_dir}/all_hidden_weights_history.npy", np.array([np.array(w) for w in all_hidden_weights_history]))
        if track_activity_history and hidden_activity_history is not None:
            np.save(f"{save_dir}/hidden_activity_history.npy", np.array(hidden_activity_history))
    
    out = {"train_losses": np.array(train_losses), "val_losses": np.array(val_losses)}
    if track_weight_history and gen_weight_history is not None:
        out["gen_weight_history"] = np.array([np.array(w) for w in gen_weight_history])
        out["all_hidden_weights_history"] = np.array([np.array(w) for w in all_hidden_weights_history])
    if track_activity_history and hidden_activity_history is not None:
        out["hidden_activity_history"] = np.array(hidden_activity_history)
    
    return out


def train_lateral_pdm(
    key,
    n_train_iters,
    batch_size,
    hidden_size,
    param_lr,
    n_relax_iters,
    thetaf,
    normalise_fwd_weights,
    normalise_bwd_weights,
    print_every,
    lateral_init_type,
    freeze_lateral,
    use_lateral,
    save_dir,
    init_w2f,
    track_weight_history,
    track_activity_history,
    zero_w1b_,
    freeze_output_layer,
    fixed_data
):
    input_size, output_size = 2, 1
    thetab = 1.0 - thetaf
    weights = lateral_pdm_init_weights(
        key, input_size, hidden_size, output_size, lateral_init_type=lateral_init_type
    )
    if use_lateral:
        print("Initial lateral connections w2_lat:")
        print(weights[4])
    if init_w2f is not None:
        weights = lateral_pdm_set_w2f_symmetric(weights, init_w2f[0], init_w2f[1])
    if normalise_fwd_weights:
        weights = lateral_pdm_normalise_fwd_weights(weights)
    if normalise_bwd_weights:
        weights = lateral_pdm_normalise_bwd_weights(weights)
    if zero_w1b_:
        weights = lateral_pdm_zero_w1b(weights)
    if freeze_output_layer:
        initial_w3_f = jnp.ones((output_size, hidden_size))
        initial_w2_b = jnp.ones((hidden_size, output_size))
        weights = (weights[0], weights[1], initial_w2_b, initial_w3_f, weights[4])
    else:
        initial_w3_f = None
        initial_w2_b = None
    if not use_lateral:
        weights = (weights[0], weights[1], weights[2], weights[3], jnp.zeros_like(weights[4]))
        freeze_lateral = True

    train_losses = []
    val_losses = []
    w2_f = weights[1]
    gen_weight_history = [np.array(w2_f[0, :])] if track_weight_history else None
    all_hidden_weights_history = [np.array(w2_f)] if track_weight_history else None
    hidden_activity_history = [] if track_activity_history else None

    data_key = key
    data_key, init_batch_key = jr.split(data_key)
    if fixed_data is not None:
        train_x, train_y, val_x, val_y = fixed_data
        x_init, y_init = sample_batch_from_fixed(
            init_batch_key, train_x, train_y, batch_size
        )
        x_val_init, y_val_init = val_x, val_y
    else:
        x_init, y_init = generate_data(init_batch_key, batch_size)
        data_key, val_key_init = jr.split(data_key)
        x_val_init, y_val_init = generate_data(val_key_init, 1000)

    init_tl = float(
        lateral_pdm_mse_loss(
            weights, x_init, y_init, thetaf, thetab, n_relax_iters
        )
    )
    init_vl = float(
        lateral_pdm_mse_loss(
            weights, x_val_init, y_val_init, thetaf, thetab, n_relax_iters
        )
    )
    train_losses.append(init_tl)
    val_losses.append(init_vl)

    if track_activity_history:
        x2_init = lateral_pdm_relax_training_fixed_iters(
            x_init, y_init, weights, thetaf, thetab, n_relax_iters
        )
        hidden_activity_history.append(
            np.array(jnp.mean(x2_init, axis=0))
        )

    print(f"  Init     | Train loss: {init_tl:.4f} | Val. loss: {init_vl:.4f}")

    for step in range(n_train_iters):
        data_key, batch_key = jr.split(data_key)
        if fixed_data is not None:
            train_x, train_y, val_x, val_y = fixed_data
            x_batch, y_batch = sample_batch_from_fixed(batch_key, train_x, train_y, batch_size)
        else:
            x_batch, y_batch = generate_data(batch_key, batch_size)
        weights = lateral_pdm_train_step(
            weights, x_batch, y_batch,
            learning_rate=param_lr,
            thetaf=thetaf,
            thetab=thetab,
            n_relax_iters=n_relax_iters,
            freeze_lateral=freeze_lateral,
        )
        if freeze_output_layer and initial_w3_f is not None:
            weights = (weights[0], weights[1], initial_w2_b, initial_w3_f, weights[4])
        if zero_w1b_:
            weights = lateral_pdm_zero_w1b(weights)
        if normalise_fwd_weights:
            weights = lateral_pdm_normalise_fwd_weights(weights)
        if normalise_bwd_weights:
            weights = lateral_pdm_normalise_bwd_weights(weights)
        if track_weight_history:
            w2_f = weights[1]
            gen_weight_history.append(np.array(w2_f[0, :]))
            all_hidden_weights_history.append(np.array(w2_f))
        if track_activity_history:
            # Record mean hidden activities (x2) at end of relaxation for current batch.
            x2 = lateral_pdm_relax_training_fixed_iters(
                x_batch, y_batch, weights, thetaf, thetab, n_relax_iters
            )
            hidden_mean = jnp.mean(x2, axis=0)
            hidden_activity_history.append(np.array(hidden_mean))

        train_loss = lateral_pdm_mse_loss(weights, x_batch, y_batch, thetaf, thetab, n_relax_iters)
        train_losses.append(float(train_loss))

        if step % print_every == 0 or step == n_train_iters - 1:
            if fixed_data is not None:
                _, _, val_x, val_y = fixed_data
                x_val, y_val = val_x, val_y
            else:
                data_key, val_key = jr.split(data_key)
                x_val, y_val = generate_data(val_key, 1000)
            val_loss = lateral_pdm_mse_loss(weights, x_val, y_val, thetaf, thetab, n_relax_iters)
            val_losses.append(float(val_loss))
            print(f"  Step {step:4d} | Train loss: {train_loss:.4f} | Val. loss: {val_loss:.4f}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/train_losses.npy", np.array(train_losses))
        np.save(f"{save_dir}/val_losses.npy", np.array(val_losses))
        np.save(
            f"{save_dir}/n_train_iters.npy",
            np.array(n_train_iters, dtype=np.int64),
        )
        if track_weight_history and gen_weight_history is not None:
            np.save(f"{save_dir}/gen_weight_history.npy", np.array(gen_weight_history))
            np.save(f"{save_dir}/all_hidden_weights_history.npy", np.array(all_hidden_weights_history))
        if track_activity_history and hidden_activity_history is not None:
            np.save(f"{save_dir}/hidden_activity_history.npy", np.array(hidden_activity_history))

    out = {
        "train_losses": np.array(train_losses),
        "val_losses": np.array(val_losses),
        "weights": weights,
    }
    if track_weight_history and gen_weight_history is not None:
        out["gen_weight_history"] = np.array(gen_weight_history)
        out["all_hidden_weights_history"] = np.array(all_hidden_weights_history)
    if track_activity_history and hidden_activity_history is not None:
        out["hidden_activity_history"] = np.array(hidden_activity_history)
        
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/toy_nonlinear")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--width", type=int, default=2)
    parser.add_argument("--param_lrs", type=float, nargs="+", default=[5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4])  # 6e-4 for grid exps., 5e-2
    parser.add_argument("--n_train_iters", type=int, default=5000)
    parser.add_argument("--infer_iters", type=int, default=20)
    parser.add_argument("--activity_lr", type=float, default=5e-1)
    parser.add_argument("--lateral_pdm_relax_iters", type=int, default=100)
    parser.add_argument("--normalise_fwd_weights", action="store_true", default=True)
    parser.add_argument("--normalise_bwd_weights", action="store_true", default=False)
    parser.add_argument("--thetaf", type=float, default=0.5)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--grid_experiment", action="store_true", default=True)
    parser.add_argument("--weight_grid_min", type=float, default=-1.0)
    parser.add_argument("--weight_grid_max", type=float, default=1.01)
    parser.add_argument("--weight_grid_step", type=float, default=0.5)
    parser.add_argument("--zero_w1b", default=False)
    parser.add_argument("--freeze_output_layer", default=False)
    parser.add_argument("--freeze_lateral", action="store_true", default=False)
    parser.add_argument("--lateral_init_type", type=str, default="glorot", choices=["glorot", "negative", "zero"])
    parser.add_argument("--use_fixed_data", action="store_true", default=True)
    parser.add_argument("--n_total", type=int, default=500)
    parser.add_argument("--n_train", type=int, default=400)
    args = parser.parse_args()

    # Optionally create one fixed dataset shared across all experiments (grid + single seeds).
    if args.use_fixed_data:
        global_data_key = jr.PRNGKey(0)
        fixed_data_global = generate_fixed_dataset(global_data_key, args.n_total, args.n_train)
    else:
        fixed_data_global = None

    for param_lr in args.param_lrs:
        print("\n" + "#" * 60)
        print(f"Running experiments for param_lr = {param_lr}")
        print("#" * 60)

        lr_results_dir = os.path.join(args.results_dir, f"lr_{param_lr}")
        os.makedirs(lr_results_dir, exist_ok=True)

        if args.grid_experiment:
            seed = 0
            print("\n" + "#" * 60)
            print(f"Running GRID experiment for seed = {seed}, param_lr = {param_lr}")
            print("#" * 60)

            key = jr.PRNGKey(seed)
            fixed_data = fixed_data_global
            # For grid experiments, store results directly under lr_results_dir so that
            # plotting scripts can find standard_pdm/unit_norm_pdm/lateral_pdm.
            run_results_dir = lr_results_dir

            weight_grid = np.arange(
                args.weight_grid_min, 
                args.weight_grid_max,
                args.weight_grid_step,
            )

            all_trajectories_1 = []
            all_trajectories_2 = []
            all_trajectories_3 = []
            for init_w11 in weight_grid:
                for init_w12 in weight_grid:
                    # Skip the (0, 0) initialisation point in the grid experiment
                    if np.isclose(init_w11, 0.0) and np.isclose(init_w12, 0.0):
                        continue
                    
                    init_gen = (float(init_w11), float(init_w12))
                    print(f"\n--- Grid point init_w=({init_w11:.2f}, {init_w12:.2f}) ---")
                    # 1) Standard PDM
                    k1, key = jr.split(key)
                    save_1 = os.path.join(
                        run_results_dir,
                        "standard_pdm",
                        f"init_w11_{init_w11:.2f}",
                        f"init_w12_{init_w12:.2f}",
                    )
                    r1 = train_pdm(
                        k1,
                        n_train_iters=args.n_train_iters,
                        batch_size=args.batch_size,
                        width=args.width,
                        param_lr=param_lr,
                        infer_iters=args.infer_iters,
                        activity_lr=args.activity_lr,
                        normalise_fwd_weights=False,
                        normalise_bwd_weights=False,
                        zero_w1b_=args.zero_w1b,
                        print_every=args.print_every,
                        save_dir=save_1,
                        init_gen_weights=init_gen,
                        track_weight_history=True,
                        track_activity_history=False,
                        freeze_output_layer=args.freeze_output_layer,
                        fixed_data=fixed_data
                    )
                    all_trajectories_1.append(
                        {
                            "gen_weight_history": r1["gen_weight_history"],
                            "all_hidden_weights_history": r1["all_hidden_weights_history"],
                            "init_w11": init_w11,
                            "init_w12": init_w12,
                        }
                    )

                    # 2) Unit-norm PDM
                    k2, key = jr.split(key)
                    save_2 = os.path.join(
                        run_results_dir,
                        "unit_norm_pdm",
                        f"init_w11_{init_w11:.2f}",
                        f"init_w12_{init_w12:.2f}",
                    )
                    r2 = train_lateral_pdm(
                        k2,
                        n_train_iters=args.n_train_iters,
                        batch_size=args.batch_size,
                        hidden_size=args.width,
                        param_lr=param_lr,
                        n_relax_iters=args.infer_iters,
                        thetaf=args.thetaf,
                        normalise_fwd_weights=args.normalise_fwd_weights,
                        normalise_bwd_weights=args.normalise_bwd_weights,
                        print_every=args.print_every,
                        lateral_init_type=args.lateral_init_type,
                        freeze_lateral=args.freeze_lateral,
                        use_lateral=False,
                        save_dir=save_2,
                        init_w2f=init_gen,
                        track_weight_history=True,
                        track_activity_history=False,
                        zero_w1b_=args.zero_w1b,
                        freeze_output_layer=args.freeze_output_layer,
                        fixed_data=fixed_data,
                    )
                    all_trajectories_2.append(
                        {
                            "gen_weight_history": r2["gen_weight_history"],
                            "all_hidden_weights_history": r2["all_hidden_weights_history"],
                            "init_w11": init_w11,
                            "init_w12": init_w12,
                        }
                    )

                    # 3) lateral pdm
                    k3, key = jr.split(key)
                    save_3 = os.path.join(
                        run_results_dir,
                        "lateral_pdm",
                        f"init_w11_{init_w11:.2f}",
                        f"init_w12_{init_w12:.2f}",
                    )
                    r3 = train_lateral_pdm(
                        k3,
                        n_train_iters=args.n_train_iters,
                        batch_size=args.batch_size,
                        hidden_size=args.width,
                        param_lr=param_lr,
                        n_relax_iters=args.lateral_pdm_relax_iters,
                        thetaf=args.thetaf,
                        normalise_fwd_weights=args.normalise_fwd_weights,
                        normalise_bwd_weights=args.normalise_bwd_weights,
                        print_every=args.print_every,
                        use_lateral=True,
                        lateral_init_type=args.lateral_init_type,
                        freeze_lateral=args.freeze_lateral,
                        save_dir=save_3,
                        init_w2f=init_gen,
                        track_weight_history=True,
                        track_activity_history=False,
                        zero_w1b_=args.zero_w1b,
                        freeze_output_layer=args.freeze_output_layer,
                        fixed_data=fixed_data,
                    )
                    all_trajectories_3.append(
                        {
                            "gen_weight_history": r3["gen_weight_history"],
                            "all_hidden_weights_history": r3["all_hidden_weights_history"],
                            "init_w11": init_w11,
                            "init_w12": init_w12,
                        }
                    )

            np.save(os.path.join(run_results_dir, "standard_pdm", "all_trajectories.npy"), all_trajectories_1)
            np.save(os.path.join(run_results_dir, "unit_norm_pdm", "all_trajectories.npy"), all_trajectories_2)
            np.save(os.path.join(run_results_dir, "lateral_pdm", "all_trajectories.npy"), all_trajectories_3)
            print(f"\nGrid experiment done. Trajectories saved under {run_results_dir}/")

        # For non-grid experiments, optionally use one fixed dataset shared across all seeds.
        if args.use_fixed_data:
            data_key_global = jr.PRNGKey(0)
            fixed_data_singles = generate_fixed_dataset(data_key_global, args.n_total, args.n_train)
        else:
            fixed_data_singles = None

        for seed in range(args.n_seeds):
            print("\n" + "#" * 60)
            print(f"Running SINGLE experiments for seed = {seed}, param_lr = {param_lr}")
            print("#" * 60)

            key = jr.PRNGKey(seed)
            fixed_data = fixed_data_singles
            run_results_dir = os.path.join(lr_results_dir, f"seed_{seed}")
            os.makedirs(run_results_dir, exist_ok=True)

            # 1) Standard PDM
            print("\n" + "=" * 60)
            print("1) Standard PDM (no unit norm, no lateral)")
            print("=" * 60)
            k1, key = jr.split(key)
            save_dir_1 = os.path.join(run_results_dir, "standard_pdm")
            result_1 = train_pdm(
                k1,
                n_train_iters=args.n_train_iters,
                batch_size=args.batch_size,
                width=args.width,
                param_lr=param_lr,
                infer_iters=args.infer_iters,
                activity_lr=args.activity_lr,
                normalise_fwd_weights=False,
                normalise_bwd_weights=False,
                zero_w1b_=args.zero_w1b,
                print_every=args.print_every,
                save_dir=save_dir_1,
                init_gen_weights=None,
                track_weight_history=True,
                track_activity_history=True,
                freeze_output_layer=args.freeze_output_layer,
                fixed_data=fixed_data,
            )

            # 2) PDM with unit norm
            print("\n" + "=" * 60)
            print("2) PDM with unit norm (train_lateral_pdm, no lateral)")
            print("=" * 60)
            k2, key = jr.split(key)
            save_dir_2 = os.path.join(run_results_dir, "unit_norm_pdm")
            result_2 = train_lateral_pdm(
                k2,
                n_train_iters=args.n_train_iters,
                batch_size=args.batch_size,
                hidden_size=args.width,
                param_lr=param_lr,
                n_relax_iters=args.infer_iters,
                thetaf=args.thetaf,
                normalise_fwd_weights=args.normalise_fwd_weights,
                normalise_bwd_weights=args.normalise_bwd_weights,
                print_every=args.print_every,
                lateral_init_type=args.lateral_init_type,
                freeze_lateral=args.freeze_lateral,
                use_lateral=False,
                save_dir=save_dir_2,
                init_w2f=None,
                track_weight_history=True,
                track_activity_history=True,
                zero_w1b_=args.zero_w1b,
                freeze_output_layer=args.freeze_output_layer,
                fixed_data=fixed_data,
            )

            # 3) lateral pdm: no norm + lateral
            print("\n" + "=" * 60)
            print("3) lateral pdm (lateral, no unit norm)")
            print("=" * 60)
            k3, key = jr.split(key)
            save_dir_3 = os.path.join(run_results_dir, "lateral_pdm")
            result_3 = train_lateral_pdm(
                k3,
                n_train_iters=args.n_train_iters,
                batch_size=args.batch_size,
                hidden_size=args.width,
                param_lr=param_lr,
                n_relax_iters=args.lateral_pdm_relax_iters,
                thetaf=args.thetaf,
                normalise_fwd_weights=args.normalise_fwd_weights,
                normalise_bwd_weights=args.normalise_bwd_weights,
                print_every=args.print_every,
                lateral_init_type=args.lateral_init_type,
                freeze_lateral=args.freeze_lateral,
                use_lateral=True,
                save_dir=save_dir_3,
                init_w2f=None,
                track_weight_history=True,
                track_activity_history=True,
                zero_w1b_=args.zero_w1b,
                freeze_output_layer=args.freeze_output_layer,
                fixed_data=fixed_data,
            )

            # Summary
            print("\n" + "=" * 60)
            print(f"Final val loss for seed {seed} (last logged value):")
            print("  Standard PDM:     ", result_1["val_losses"][-1] if len(result_1["val_losses"]) else "N/A")
            print("  Unit-norm PDM:   ", result_2["val_losses"][-1] if len(result_2["val_losses"]) else "N/A")
            print("  lateral pdm:      ", result_3["val_losses"][-1] if len(result_3["val_losses"]) else "N/A")
            # Print lateral connections (w2_lat) for lateral PDM single runs
            w2_lat = result_3["weights"][4]
            print("  lateral pdm w2_lat (lateral connections):")
            print(w2_lat)
            print("=" * 60)
            print(f"Results saved under {run_results_dir}/")


if __name__ == "__main__":
    main()
