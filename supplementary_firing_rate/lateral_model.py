import jax
import jax.numpy as jnp
import jax.random as jr
import functools
import numpy as np


def _relu(x):
    return jnp.maximum(x, 0.0)


def _g_relu(x):
    """Gating for ReLU derivative: (x > 0)."""
    return (x > 0).astype(jnp.float32)


def lateral_pdm_init_weights(
    key,
    input_size,
    hidden_size,
    output_size,
    init_type="glorot",
    lateral_init_type="glorot",
):
    """Initialize weights. Returns (w1_b, w2_f, w2_b, w3_f, w2_lat).

    `init_type` sets non-lateral matrices (w1_b, w2_f, w2_b, w3_f).
    `lateral_init_type` sets lateral matrix `w2_lat` (off-diagonal only; diagonal is zeroed):
      - ``"glorot"``: uniform Glorot/Xavier
      - ``"kaiming"`` / ``"he"``: legacy fan-in uniform (kept for compatibility)
      - ``"lecun"``: LeCun normal (zero-mean Gaussian with Var = 1/fan_in)
      - ``"negative"``: off-diagonal -1 (inhibition template)
      - ``"zero"``: all zeros (no lateral recurrence until learning)
    """
    k1, k2, k3, k4, k5 = jr.split(key, 5)

    def _glorot_uniform(k, shape):
        fan_out, fan_in = shape  # weights used as x @ W.T
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return jr.uniform(k, shape, minval=-limit, maxval=limit)

    def _kaiming_uniform(k, shape):
        _, fan_in = shape  # weights used as x @ W.T
        relu_gain = np.sqrt(2.0)
        limit = relu_gain * np.sqrt(3.0 / fan_in)
        return jr.uniform(k, shape, minval=-limit, maxval=limit)

    def _lecun_normal(k, shape):
        _, fan_in = shape
        relu_gain = np.sqrt(2.0)
        std = relu_gain / np.sqrt(fan_in)
        return std * jr.normal(k, shape)

    if init_type == "glorot":
        init = _glorot_uniform
    elif init_type in ("kaiming", "he"):
        init = _kaiming_uniform
    elif init_type in ("lecun"):
        init = _lecun_normal
    else:
        raise ValueError(
            "init_type must be 'glorot', 'kaiming', or 'lecun', "
            f"got {init_type!r}"
        )

    # w2_f: input -> hidden, used as x1 @ w2_f.T, so shape = (hidden, input)
    w2_f = init(k2, (hidden_size, input_size))
    # w2_b: output -> hidden, used as y @ w2_b.T, so shape = (hidden, output)
    w2_b = init(k3, (hidden_size, output_size))
    # w3_f: hidden -> output, used as x2 @ w3_f.T, so shape = (output, hidden)
    w3_f = init(k4, (output_size, hidden_size))
    # w1_b: hidden -> input, used as x2 @ w1_b.T, so shape = (input, hidden)
    w1_b = init(k1, (input_size, hidden_size))

    if lateral_init_type == "glorot":
        w2_lat = _glorot_uniform(k5, (hidden_size, hidden_size))
    elif lateral_init_type in ("kaiming", "he"):
        w2_lat = _kaiming_uniform(k5, (hidden_size, hidden_size))
    elif lateral_init_type in ("lecun"):
        w2_lat = _lecun_normal(k5, (hidden_size, hidden_size))
    elif lateral_init_type == "negative":
        w2_lat = -jnp.ones((hidden_size, hidden_size))
    elif lateral_init_type == "zero":
        w2_lat = jnp.zeros((hidden_size, hidden_size))
    else:
        raise ValueError(
            "lateral_init_type must be 'glorot', 'kaiming', 'lecun', 'negative', or 'zero', "
            f"got {lateral_init_type!r}"
        )
    w2_lat = w2_lat.at[jnp.diag_indices(hidden_size)].set(0.0)
    return (w1_b, w2_f, w2_b, w3_f, w2_lat)


def lateral_pdm_zero_diagonal(w2_lat):
    return w2_lat.at[jnp.diag_indices(w2_lat.shape[0])].set(0.0)


def lateral_pdm_clamp_lateral(w2_lat, min_val=-1.0, max_val=1.0):
    return jnp.clip(w2_lat, min_val, max_val)


@functools.partial(jax.jit, static_argnums=(5,))
def lateral_pdm_relax_training_fixed_iters(x1, target, weights, thetaf, thetab, n_iters, activ_fn="relu"):
    """Relax with fixed number of iterations (JIT + scan for speed)."""
    w1_b, w2_f, w2_b, w3_f, w2_lat = weights
    v2_f = jnp.dot(x1, w2_f.T)
    v2_b = jnp.dot(target, w2_b.T)

    def step(carry, _):
        x2 = carry
        v2_lat = jnp.dot(x2, w2_lat.T)
        v2 = thetaf * v2_f + thetab * (v2_b + v2_lat)
        return _relu(v2), None

    x2_init = _relu(v2_f)
    x2_final, _ = jax.lax.scan(step, x2_init, None, length=n_iters - 1)
    return x2_final


@functools.partial(jax.jit, static_argnums=(4,))
def lateral_pdm_predict(x1, weights, thetaf, thetab, max_iters, activ_fn="relu"):
    """
    Prediction: iterate (x2, output) until fixed iters (JIT + scan for speed).
    x1: (batch_size, input_size). Returns output: (batch_size, output_size).
    """
    w1_b, w2_f, w2_b, w3_f, w2_lat = weights
    batch_size, hidden_size = x1.shape[0], w2_f.shape[0]
    output_size = w3_f.shape[0]
    x2 = jnp.zeros((batch_size, hidden_size))
    output = jnp.zeros((batch_size, output_size))

    def body(carry, _):
        x2, output = carry
        v2_f = jnp.dot(x1, w2_f.T)
        v2_b = jnp.dot(output, w2_b.T)
        v2_lat = jnp.dot(x2, w2_lat.T)
        v2 = thetaf * v2_f + thetab * (v2_b + v2_lat)
        x2_new = _relu(v2)
        output_new = jnp.dot(x2_new, w3_f.T)
        return (x2_new, output_new), None

    (x2_final, output_final), _ = jax.lax.scan(body, (x2, output), None, length=max_iters)
    return output_final


@functools.partial(jax.jit, static_argnums=(6, 8))  # n_relax_iters, freeze_lateral
def lateral_pdm_train_step(weights, x1, target, learning_rate, thetaf, thetab, n_relax_iters,
                         clamp_lat_min=-1.0, clamp_lat_max=1.0, freeze_lateral=False):
    """
    One training step: relax to get x2, then apply Hebbian updates.
    Returns updated weights (w1_b, w2_f, w2_b, w3_f, w2_lat).
    If freeze_lateral is True, w2_lat is unchanged (e.g. kept at [[0,-1],[-1,0]]).
    """
    w1_b, w2_f, w2_b, w3_f, w2_lat = weights
    x2 = lateral_pdm_relax_training_fixed_iters(x1, target, weights, thetaf, thetab, n_relax_iters)

    v1 = jnp.dot(x2, w1_b.T)
    v2_f = jnp.dot(x1, w2_f.T)
    v2_b = jnp.dot(target, w2_b.T)
    v2_lat = jnp.dot(x2, w2_lat.T)
    v3 = jnp.dot(x2, w3_f.T)

    g = _g_relu(x2)

    # Updates (batch dimension: sum over batch)
    dw1_b = jnp.dot((x1 - v1).T, x2)
    w1_b = w1_b + learning_rate * dw1_b

    dw2_f = thetaf * jnp.dot((g * (x2 - v2_f)).T, x1)
    w2_f = w2_f + learning_rate * dw2_f

    dw2_b = thetab * jnp.dot((g * (x2 - v2_b - v2_lat)).T, target)
    w2_b = w2_b + learning_rate * dw2_b

    dw3_f = thetaf * jnp.dot((target - v3).T, x2)
    w3_f = w3_f + learning_rate * dw3_f

    # Lateral: update only when freeze_lateral is False (otherwise keep fixed = [[0,-1],[-1,0]] for 2x2)
    def update_lateral(w2_lat):
        dw2_lat = jnp.dot((x2 - v2_lat - v2_b).T, x2)
        w2_lat_new = w2_lat + learning_rate * dw2_lat
        w2_lat_new = lateral_pdm_zero_diagonal(w2_lat_new)
        return lateral_pdm_clamp_lateral(w2_lat_new, clamp_lat_min, clamp_lat_max)

    w2_lat = jax.lax.cond(freeze_lateral, lambda w: w, update_lateral, w2_lat)

    return (w1_b, w2_f, w2_b, w3_f, w2_lat)


def lateral_pdm_zero_w1b(weights):
    """Zero the bottom-up input-to-hidden weights w1_b."""
    w1_b, w2_f, w2_b, w3_f, w2_lat = weights
    w1_b = jnp.zeros_like(w1_b)
    return (w1_b, w2_f, w2_b, w3_f, w2_lat)


def lateral_pdm_normalise_fwd_weights(weights):
    """Normalize rows of w2_f to unit norm (input to hidden)."""
    w1_b, w2_f, w2_b, w3_f, w2_lat = weights
    norms = jnp.linalg.norm(w2_f, axis=1, keepdims=True)
    w2_f = w2_f / jnp.maximum(norms, 1e-8)
    return (w1_b, w2_f, w2_b, w3_f, w2_lat)


def lateral_pdm_normalise_bwd_weights(weights):
    """Normalize rows of w2_b to unit norm (hidden to output)."""
    w1_b, w2_f, w2_b, w3_f, w2_lat = weights
    norms = jnp.linalg.norm(w2_b, axis=1, keepdims=True)
    w2_b = w2_b / jnp.maximum(norms, 1e-8)
    return (w1_b, w2_f, w2_b, w3_f, w2_lat)


def lateral_pdm_set_w2f_symmetric(weights, w11, w12):
    """Set w2_f to symmetric grid init: row0 = [w11, w12], row1 = [w12, w11]."""
    w1_b, w2_f, w2_b, w3_f, w2_lat = weights
    w2_f = w2_f.at[0, :].set(jnp.array([w11, w12]))
    w2_f = w2_f.at[1, :].set(jnp.array([w12, w11]))
    return (w1_b, w2_f, w2_b, w3_f, w2_lat)


@functools.partial(jax.jit, static_argnums=(5,))  # n_relax_iters must be concrete for predict scan
def lateral_pdm_mse_loss(weights, x1, target, thetaf, thetab, n_relax_iters):
    """MSE between target and Lateral PDM prediction."""
    pred = lateral_pdm_predict(x1, weights, thetaf, thetab, max_iters=n_relax_iters)
    return jnp.mean((target - pred) ** 2)
