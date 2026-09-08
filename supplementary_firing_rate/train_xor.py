import os
import argparse
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

import equinox as eqx
import jpc
import optax

from utils import normalize_generator_first_layer_weight_rows


# XOR truth table: (x1, x2) -> y
XOR_INPUTS = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
XOR_TARGETS = jnp.array([0.0, 1.0, 1.0, 0.0])[:, None]


def generate_xor_data(key, batch_size, input_noise_std=0.0):
    """Generate XOR data by sampling from the 4 corners. Optionally add Gaussian noise to inputs."""
    idx_key, noise_key = jr.split(key)
    indices = jr.choice(idx_key, 4, shape=(batch_size,), replace=True)
    x = XOR_INPUTS[indices]
    y = XOR_TARGETS[indices]
    if input_noise_std > 0:
        noise = input_noise_std * jr.normal(noise_key, x.shape)
        x = x + noise
    return x, y


def accuracy(y_pred, y_true, threshold=0.5):
    """Fraction of predictions correct (threshold at 0.5)."""
    pred_class = (y_pred >= threshold).astype(jnp.float32)
    return jnp.mean(pred_class == y_true)


def infer_pdm_activities(
    generator,
    amortiser,
    activities,
    x,
    y,
    activity_optim,
    infer_iters,
    free_output=False,
):
    """Run PDM activity inference. If `free_output`, do not clamp the true
    label: after each step, refresh the output activity from the last hidden
    layer so the backward model tracks the current prediction.
    """
    output = activities[-1] if free_output else y
    activity_opt_state = activity_optim.init(activities)
    for _ in range(infer_iters):
        result = jpc.update_pdm_activities(
            top_down_model=generator,
            bottom_up_model=amortiser,
            activities=activities,
            optim=activity_optim,
            opt_state=activity_opt_state,
            output=output,
            input=x,
        )
        activities = result["activities"]
        activity_opt_state = result["opt_state"]
        if free_output:
            activities = list(activities)
            activities[-1] = vmap(generator[-1])(activities[-2])
            output = activities[-1]
    return activities


def train(
    seed,
    # Data params
    batch_size,
    input_noise_std,
    # Model params
    width,
    n_hidden,
    act_fn,
    unit_norm_first_layer,
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
    # Setup
    key = jr.PRNGKey(seed)
    gen_key, amort_key, data_key = jr.split(key, 3)

    # Model dimensions: input (x1, x2), output (y)
    input_dim, output_dim = 2, 1
    depth = n_hidden + 1

    generator = jpc.make_mlp(
        gen_key,
        input_dim=input_dim,
        width=width,
        depth=depth,
        output_dim=output_dim,
        act_fn=act_fn,
    )
    amortiser = jpc.make_mlp(
        amort_key,
        input_dim=output_dim,
        width=width,
        depth=depth,
        output_dim=input_dim,
        act_fn=act_fn,
    )[::-1]

    if unit_norm_first_layer:
        generator = normalize_generator_first_layer_weight_rows(generator)

    activity_optim = optax.sgd(activity_lr)
    gen_optim = optax.adam(param_lr)
    amort_optim = optax.adam(param_lr)

    gen_opt_state = gen_optim.init(eqx.filter(generator, eqx.is_array))
    amort_opt_state = amort_optim.init(eqx.filter(amortiser, eqx.is_array))

    train_losses, test_losses, test_infer_losses = [], [], []
    train_accs, test_accs, test_infer_accs = [], [], []

    print("Training PDM on XOR (y = x1 XOR x2)")
    print(f"Model: {n_hidden} hidden layers, width={width}, act_fn={act_fn}")
    if unit_norm_first_layer:
        print("Unit-norm first layer: enabled")
    print(f"Inference: {infer_iters} iters, activity_lr={activity_lr}")
    print(f"Learning: param_lr={param_lr}, batch_size={batch_size}")
    if input_noise_std > 0:
        print(f"Input noise std: {input_noise_std}")
    print("-" * 60)

    for step in range(n_train_iters):
        data_key, batch_key = jr.split(data_key)
        x_train, y_train = generate_xor_data(batch_key, batch_size, input_noise_std)

        activities = jpc.init_activities_with_ffwd(
            model=generator,
            input=x_train,
        )
        train_loss = jpc.mse_loss(activities[-1], y_train)
        train_acc = accuracy(activities[-1], y_train)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        activities = infer_pdm_activities(
            generator,
            amortiser,
            activities,
            x_train,
            y_train,
            activity_optim,
            infer_iters,
        )

        param_update_result = jpc.update_pdm_params(
            top_down_model=generator,
            bottom_up_model=amortiser,
            activities=activities,
            top_down_optim=gen_optim,
            bottom_up_optim=amort_optim,
            top_down_opt_state=gen_opt_state,
            bottom_up_opt_state=amort_opt_state,
            output=y_train,
            input=x_train
        )
        generator, amortiser = param_update_result["models"]
        gen_opt_state, amort_opt_state = param_update_result["opt_states"]

        if unit_norm_first_layer:
            generator = normalize_generator_first_layer_weight_rows(generator)

        if step % print_every == 0 or step == n_train_iters - 1:
            data_key, test_key = jr.split(data_key)
            x_test, y_test = generate_xor_data(
                test_key, batch_size * 4,
                input_noise_std=0.0
            )
            activities = jpc.init_activities_with_ffwd(
                model=generator, input=x_test
            )
            test_loss = jpc.mse_loss(activities[-1], y_test)
            test_acc = accuracy(activities[-1], y_test)
            inferred = infer_pdm_activities(
                generator,
                amortiser,
                activities,
                x_test,
                y_test,
                activity_optim,
                infer_iters,
                free_output=True,
            )
            test_infer_loss = jpc.mse_loss(inferred[-1], y_test)
            test_infer_acc = accuracy(inferred[-1], y_test)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            test_infer_losses.append(test_infer_loss)
            test_infer_accs.append(test_infer_acc)

            print(
                f"Step {step:4d} | "
                f"Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f} | "
                f"Train acc: {train_acc:.2%} | Test acc: {test_acc:.2%} | "
                f"Test acc (infer): {test_infer_acc:.2%}"
            )

    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/train_losses.npy", np.array(train_losses))
    np.save(f"{save_dir}/test_losses.npy", np.array(test_losses))
    np.save(f"{save_dir}/test_infer_losses.npy", np.array(test_infer_losses))
    np.save(f"{save_dir}/train_accs.npy", np.array(train_accs))
    np.save(f"{save_dir}/test_accs.npy", np.array(test_accs))
    np.save(f"{save_dir}/test_infer_accs.npy", np.array(test_infer_accs))
    print(f"\nResults saved to {save_dir}")

    return {
        "train_losses": np.array(train_losses),
        "test_losses": np.array(test_losses),
        "test_infer_losses": np.array(test_infer_losses),
        "train_accs": np.array(train_accs),
        "test_accs": np.array(test_accs),
        "test_infer_accs": np.array(test_infer_accs),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--results_dir", type=str, default="results/xor")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--input_noise_std", type=float, default=0.)
    parser.add_argument("--width", type=int, default=2)
    parser.add_argument("--n_hidden", type=int, default=1)
    parser.add_argument("--act_fn", type=str, default="relu")
    parser.add_argument("--infer_iters", type=int, default=20)
    parser.add_argument("--activity_lr", type=float, default=5e-1)
    parser.add_argument("--param_lr", type=float, default=5e-3)
    parser.add_argument("--n_train_iters", type=int, default=1000)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--n_seeds", type=int, default=5)

    args = parser.parse_args()

    for seed in range(args.n_seeds):
        for unit_norm_first_layer in [True, False]:
            save_dir = os.path.join(
                args.results_dir,
                f"width_{args.width}",
                f"n_hidden_{args.n_hidden}",
                f"act_{args.act_fn}",
                f"unit_norm_{unit_norm_first_layer}",
                f"seed_{seed}",
            )
            print(
                f"\n{'='*60}\n"
                f"Running seed={seed} with unit_norm_first_layer={unit_norm_first_layer}\n"
                f"{'='*60}"
            )
            train(
                seed=seed,
                batch_size=args.batch_size,
                input_noise_std=args.input_noise_std,
                width=args.width,
                n_hidden=args.n_hidden,
                act_fn=args.act_fn,
                infer_iters=args.infer_iters,
                activity_lr=args.activity_lr,
                param_lr=args.param_lr,
                n_train_iters=args.n_train_iters,
                print_every=args.print_every,
                save_dir=save_dir,
                unit_norm_first_layer=unit_norm_first_layer,
            )
