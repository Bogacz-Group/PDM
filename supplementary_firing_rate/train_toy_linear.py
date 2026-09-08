"""
Train a PDM model on a simple linear stochastic task: y = ax + η
where x ~ N(0, 1) and η ~ N(0, σ)
"""

import os
import argparse
import numpy as np
import jax.numpy as jnp
import jax.random as jr

import equinox as eqx
import jpc
import optax


def init_first_layer_weight(model, weight_value):
    """Initialize only the first layer (input-to-hidden) weight to a constant value."""
    weight = jnp.full_like(model[0][1].weight, weight_value)
    model = eqx.tree_at(lambda m: m[0][1].weight, model, weight)
    return model


def normalize_first_layer_weight(model):
    """Normalize the first layer weight to unit norm."""
    weight = model[0][1].weight
    norm = jnp.linalg.norm(weight)
    normalized_weight = weight / jnp.maximum(norm, 1e-8)
    model = eqx.tree_at(lambda m: m[0][1].weight, model, normalized_weight)
    return model


def get_first_layer_weight(model):
    """Extract the first weight value from the input-to-hidden layer."""
    return float(model[0][1].weight.flatten()[0])


def generate_data(key, batch_size, slope=3.0, noise_std=0.5):
    """Generate data from y = slope * x + η where x ~ N(0, 1), η ~ N(0, noise_std)"""
    key_x, key_noise = jr.split(key)
    x = jr.normal(key_x, (batch_size, 1))
    noise = noise_std * jr.normal(key_noise, (batch_size, 1))
    y = slope * x + noise
    return x, y


def train(
    seed,
    # Data params
    slope,
    noise_std,
    batch_size,
    # Model params
    width,
    n_hidden,
    act_fn,
    init_gen_weight,
    init_amort_weight,
    unit_norm_w1,
    # Inference params
    infer_iters,
    activity_lr,
    # Learning params
    param_lr,
    n_train_iters,
    # Logging
    print_every,
    save_dir,
):
    key = jr.PRNGKey(seed)
    gen_key, amort_key, data_key = jr.split(key, 3)
    
    # Models
    input_dim = output_dim = 1
    depth = n_hidden + 1
    generator = jpc.make_mlp(
        gen_key,
        input_dim=input_dim,
        width=width,
        depth=depth,
        output_dim=output_dim,
        act_fn=act_fn
    )
    amortiser = jpc.make_mlp(
        amort_key,
        input_dim=output_dim,
        width=width,
        depth=depth,
        output_dim=input_dim,
        act_fn=act_fn
    )[::-1]  # Reverse for bottom-up processing
    
    if init_gen_weight is not None:
        generator = init_first_layer_weight(generator, init_gen_weight)
    if init_amort_weight is not None:
        amortiser = init_first_layer_weight(amortiser[::-1], init_amort_weight)[::-1]
    
    if unit_norm_w1:
        generator = normalize_first_layer_weight(generator)
    
    # Optimisers
    activity_optim = optax.sgd(activity_lr)
    gen_optim = optax.adam(param_lr)
    amort_optim = optax.adam(param_lr)
    
    gen_opt_state = gen_optim.init(eqx.filter(generator, eqx.is_array))
    amort_opt_state = amort_optim.init(eqx.filter(amortiser, eqx.is_array))
    
    # Metrics
    train_losses, val_losses = [], []
    gen_weights, amort_weights = [], []
    
    # Track initial weights (input-to-hidden for gen, output-to-hidden for amort)
    gen_weights.append(get_first_layer_weight(generator))
    amort_weights.append(get_first_layer_weight(amortiser[::-1]))

    # Metrics at initialization (before any train step)
    data_key, init_batch_key = jr.split(data_key)
    x_init, y_init = generate_data(init_batch_key, batch_size, slope, noise_std)
    gen_activities_init = jpc.init_activities_with_ffwd(model=generator, input=x_init)
    init_train_loss = float(jpc.mse_loss(gen_activities_init[-1], y_init))
    train_losses.append(init_train_loss)

    data_key, init_val_key = jr.split(data_key)
    x_val_init, y_val_init = generate_data(init_val_key, 1000, slope, noise_std)
    val_act_init = jpc.init_activities_with_ffwd(model=generator, input=x_val_init)
    init_val_loss = float(jpc.mse_loss(val_act_init[-1], y_val_init))
    val_losses.append(init_val_loss)

    print(f"Training PDM on y = {slope}*x + N(0, {noise_std})")
    print(f"Model: {n_hidden} hidden layers, width={width}, act_fn={act_fn}")
    print(f"Inference: {infer_iters} iters, activity_lr={activity_lr}")
    print(f"Learning: param_lr={param_lr}, batch_size={batch_size}")
    print("-" * 60)
    print(
        f"Init     | Train loss: {init_train_loss:.4f} | Val. loss: {init_val_loss:.4f}"
    )

    for step in range(n_train_iters):
        data_key, batch_key = jr.split(data_key)
        x_batch, y_batch = generate_data(batch_key, batch_size, slope, noise_std)
        
        gen_activities = jpc.init_activities_with_ffwd(
            model=generator,
            input=x_batch
        )
        train_loss = jpc.mse_loss(gen_activities[-1], y_batch)
        train_losses.append(train_loss)

        # Use amortiser initialisation for activities
        activities = gen_activities
        activity_opt_state = activity_optim.init(activities)
        
        # Inference
        for _ in range(infer_iters):
            activity_update_result = jpc.update_pdm_activities(
                top_down_model=generator,
                bottom_up_model=amortiser,
                activities=activities,
                optim=activity_optim,
                opt_state=activity_opt_state,
                output=y_batch,
                input=x_batch
            )
            activities = activity_update_result["activities"]
            activity_opt_state = activity_update_result["opt_state"]
        
        # Learning
        param_update_result = jpc.update_pdm_params(
            top_down_model=generator,
            bottom_up_model=amortiser,
            activities=activities,
            top_down_optim=gen_optim,
            bottom_up_optim=amort_optim,
            top_down_opt_state=gen_opt_state,
            bottom_up_opt_state=amort_opt_state,
            output=y_batch,
            input=x_batch
        )
        generator, amortiser = param_update_result["models"]
        gen_opt_state, amort_opt_state = param_update_result["opt_states"]
        
        if unit_norm_w1:
            generator = normalize_first_layer_weight(generator)
        
        gen_weights.append(get_first_layer_weight(generator))
        amort_weights.append(get_first_layer_weight(amortiser[::-1]))
        
        if step % print_every == 0 or step == n_train_iters - 1:
            data_key, test_key = jr.split(data_key)
            x_test, y_test = generate_data(test_key, 1000, slope, noise_std)
            
            test_gen_activities = jpc.init_activities_with_ffwd(
                model=generator, input=x_test
            )
            val_loss = jpc.mse_loss(test_gen_activities[-1], y_test)
            val_losses.append(val_loss)

            print(
                f"Step {step:4d} | "
                f"Train loss: {train_loss:.4f} | "
                f"Val. loss: {val_loss:.4f}"
            )
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/train_losses.npy", np.array(train_losses))
        np.save(f"{save_dir}/val_losses.npy", np.array(val_losses))
        np.save(f"{save_dir}/gen_weights.npy", np.array(gen_weights))
        np.save(f"{save_dir}/amort_weights.npy", np.array(amort_weights))
        np.save(
            f"{save_dir}/n_train_iters.npy",
            np.array(n_train_iters, dtype=np.int64),
        )
        print(f"\nResults saved to {save_dir}")
    
    return {
        "gen_weights": np.array(gen_weights),
        "amort_weights": np.array(amort_weights),
        "train_losses": np.array(train_losses),
        "val_losses": np.array(val_losses),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/toy_linear")
    parser.add_argument("--slope", type=float, default=3.0)
    parser.add_argument("--noise_std", type=float, default=0.5) 
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--width", type=int, default=1)
    parser.add_argument("--n_hidden", type=int, default=1)
    parser.add_argument("--act_fn", type=str, default="linear")
    parser.add_argument("--infer_iters", type=int, default=5)
    parser.add_argument("--activity_lr", type=float, default=5e-1)
    parser.add_argument("--param_lrs", type=float, nargs="+", default=[5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4])
    parser.add_argument("--n_train_iters", type=int, default=5000)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--weight_grid_min", type=float, default=-1.0)
    parser.add_argument("--weight_grid_max", type=float, default=1.0)
    parser.add_argument("--weight_grid_step", type=float, default=0.5)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--grid_experiment", action="store_true", default=True)
    
    args = parser.parse_args()

    for param_lr in args.param_lrs:
        print("\n" + "#" * 60)
        print(f"Running experiments for param_lr = {param_lr}")
        print("#" * 60)

        lr_results_dir = os.path.join(args.results_dir, f"lr_{param_lr}")

        if args.grid_experiment:
            weight_grid = np.arange(
                args.weight_grid_min,
                args.weight_grid_max + args.weight_grid_step,
                args.weight_grid_step,
            )
            for unit_norm_w1 in [False, True]:
                norm_str = "unit_norm" if unit_norm_w1 else "no_norm"
                all_trajectories = []
                for init_gen_weight in weight_grid:
                    # Skip init_gen_weight=0 for unit_norm (can't normalize zero to unit norm)
                    if unit_norm_w1 and init_gen_weight == 0:
                        continue
                    
                    for init_amort_weight in weight_grid:
                        seed = 0  # single seed for grid
                        save_dir = os.path.join(
                            lr_results_dir,
                            norm_str,
                            f"slope_{args.slope}",
                            f"noise_{args.noise_std}",
                            f"width_{args.width}",
                            f"n_hidden_{args.n_hidden}",
                            f"act_{args.act_fn}",
                            f"init_gen_{init_gen_weight:.2f}",
                            f"init_amort_{init_amort_weight:.2f}",
                            f"seed_{seed}",
                        )

                        print(f"\n{'='*60}")
                        print(
                            f"Running GRID with unit_norm_w1={unit_norm_w1}, "
                            f"init_gen={init_gen_weight:.2f}, init_amort={init_amort_weight:.2f}, seed={seed}, "
                            f"param_lr={param_lr}"
                        )
                        print(f"{'='*60}")

                        result = train(
                            seed=seed,
                            slope=args.slope,
                            noise_std=args.noise_std,
                            batch_size=args.batch_size,
                            width=args.width,
                            n_hidden=args.n_hidden,
                            act_fn=args.act_fn,
                            init_gen_weight=init_gen_weight,
                            init_amort_weight=init_amort_weight,
                            unit_norm_w1=unit_norm_w1,
                            infer_iters=args.infer_iters,
                            activity_lr=args.activity_lr,
                            param_lr=param_lr,
                            n_train_iters=args.n_train_iters,
                            print_every=args.print_every,
                            save_dir=save_dir,
                        )

                        all_trajectories.append(
                            {
                                "gen_weights": result["gen_weights"],
                                "amort_weights": result["amort_weights"],
                                "init_gen": init_gen_weight,
                                "init_amort": init_amort_weight,
                                "unit_norm_w1": unit_norm_w1,
                            }
                        )

                save_path = os.path.join(lr_results_dir, norm_str)
                os.makedirs(save_path, exist_ok=True)
                np.save(os.path.join(save_path, "all_trajectories.npy"), all_trajectories)
                print(f"\nGrid trajectories saved to {save_path}/all_trajectories.npy")

        # Non-grid experiments
        for unit_norm_w1 in [False, True]:
            norm_str = "unit_norm" if unit_norm_w1 else "no_norm"

            for seed in range(args.n_seeds):
                print(f"\n{'#'*60}")
                print(f"Running SINGLE experiments for seed = {seed}, param_lr = {param_lr}, unit_norm_w1={unit_norm_w1}")
                print(f"{'#'*60}")

                save_dir = os.path.join(
                    lr_results_dir,
                    "single_runs",
                    norm_str,
                    f"slope_{args.slope}",
                    f"noise_{args.noise_std}",
                    f"width_{args.width}",
                    f"n_hidden_{args.n_hidden}",
                    f"act_{args.act_fn}",
                    f"seed_{seed}",
                )

                result = train(
                    seed=seed,
                    slope=args.slope,
                    noise_std=args.noise_std,
                    batch_size=args.batch_size,
                    width=args.width,
                    n_hidden=args.n_hidden,
                    act_fn=args.act_fn,
                    init_gen_weight=None,
                    init_amort_weight=None,
                    unit_norm_w1=unit_norm_w1,
                    infer_iters=args.infer_iters,
                    activity_lr=args.activity_lr,
                    param_lr=param_lr,
                    n_train_iters=args.n_train_iters,
                    print_every=args.print_every,
                    save_dir=save_dir,
                )

                val_losses = result["val_losses"]
                final_val = val_losses[-1] if len(val_losses) else None
                print("\n" + "=" * 60)
                print("Final val loss (last logged value):")
                print(f"  unit_norm_w1={unit_norm_w1}: {final_val if final_val is not None else 'N/A'}")
                print("=" * 60)
