import os
import argparse

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA

from lateral_model import (
    lateral_pdm_init_weights,
    lateral_pdm_mse_loss,
    lateral_pdm_train_step,
    lateral_pdm_normalise_fwd_weights,
    lateral_pdm_predict,
    lateral_pdm_relax_training_fixed_iters,
)


def rankme(Z: np.ndarray, eps: float = 1e-10) -> float:
    """
    RankMe: exponentiated Shannon entropy of normalized singular values.
    Z: (N, d) representation matrix. Returns scalar in (0, min(N, d)].
    """
    Z = np.asarray(Z, dtype=np.float64)
    try:
        _, s, _ = np.linalg.svd(Z, full_matrices=False)
    except np.linalg.LinAlgError:
        return float("nan")
    s = s + eps
    p = s / s.sum()
    log_p = np.log(p)
    log_p[p <= 0] = 0.0
    entropy = -np.sum(p * log_p)
    return float(np.exp(entropy))


def load_iris_dataset(test_size=0.2, random_state=0):
    """
    Load the Iris dataset once, standardise features, and return a fixed
    train/test split in both numpy (for sklearn) and JAX (for training).
    """
    iris = load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target.astype(np.int32)

    # Split first, then fit scaler on train only (avoid leakage)
    X_train_raw, X_test_raw, y_train_idx, y_test_idx = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    X_train_np = X_train
    y_train_np = y_train_idx
    X_test_np = X_test
    y_test_np = y_test_idx

    n_classes = int(np.max(y) + 1)
    eye = jnp.eye(n_classes, dtype=jnp.float32)
    y_train = eye[y_train_idx]
    y_test = eye[y_test_idx]
    train_x = jnp.asarray(X_train)
    test_x = jnp.asarray(X_test)

    fixed_data = (train_x, y_train, test_x, y_test)
    return fixed_data, (X_train_np, y_train_np, X_test_np, y_test_np)


def sample_batch_from_fixed(key, train_x, train_y, batch_size):
    """Sample a random batch (with replacement) from fixed train data."""
    n_train = train_x.shape[0]
    indices = jr.choice(key, n_train, shape=(batch_size,), replace=True)
    return train_x[indices], train_y[indices]


def save_training_metrics(
    save_dir,
    tracked_w2f_weights,
    train_losses,
    val_losses,
    val_accuracies,
    rankmes,
    n_train_iters: int,
    rankmes_fwd=None,
    tracked_lateral_weights=None,
    tracked_lateral_steps=None,
    separability_metrics: dict | None = None,
):
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/tracked_w2f_weights.npy", tracked_w2f_weights)
    np.save(f"{save_dir}/initial_w2f_weights.npy", tracked_w2f_weights[0])
    np.save(f"{save_dir}/train_losses.npy", np.array(train_losses))
    np.save(f"{save_dir}/val_losses.npy", np.array(val_losses))
    np.save(f"{save_dir}/val_accuracies.npy", np.array(val_accuracies))
    np.save(f"{save_dir}/rankme.npy", np.array(rankmes))
    if rankmes_fwd is not None:
        np.save(f"{save_dir}/rankme_fwd.npy", np.array(rankmes_fwd))
    np.save(f"{save_dir}/n_train_iters.npy", np.array(n_train_iters, dtype=np.int64))
    if tracked_lateral_weights is not None:
        np.save(f"{save_dir}/tracked_lateral_weights.npy", tracked_lateral_weights)
    if tracked_lateral_steps is not None:
        np.save(f"{save_dir}/tracked_lateral_steps.npy", tracked_lateral_steps)
    if separability_metrics is not None:
        np.savez(f"{save_dir}/separability_metrics.npz", **separability_metrics)


def _data_pca2_basis(X: np.ndarray) -> dict:
    """
    Compute a 2D PCA basis on (standardised) input data X.
    Returns components (2, d) and explained variance ratio (2,).
    """
    X = np.asarray(X, dtype=np.float64)
    pca = PCA(n_components=2)
    pca.fit(X)
    return {
        "components2": np.asarray(pca.components_, dtype=np.float32),
        "explained_var_ratio": np.asarray(pca.explained_variance_ratio_, dtype=np.float32),
    }


def _project_w2_over_steps(
    tracked_w2f_weights: np.ndarray, steps: np.ndarray, data_pca_components2: np.ndarray
) -> np.ndarray:
    """
    Project input->hidden weight vectors onto the first 2 PCs of the input data.

    tracked_w2f_weights: (T, H, d_in)
    steps: (S,)
    data_pca_components2: (2, d_in)
    returns: (S, H, 2)
    """
    W = np.asarray(tracked_w2f_weights, dtype=np.float32)
    steps = np.asarray(steps, dtype=np.int64)
    C = np.asarray(data_pca_components2, dtype=np.float32)
    # (S, H, d_in) @ (d_in, 2) -> (S, H, 2)
    return (W[steps] @ C.T).astype(np.float32)


def bp_init_params(key, input_size, hidden_size, output_size, init_type="kaiming"):
    # Reuse the exact PDM initializer draws for forward weights so BP and PDM match.
    pdm_weights = lateral_pdm_init_weights(
        key, input_size, hidden_size, output_size, init_type=init_type
    )
    _, w2_f, _, w3_f, _ = pdm_weights
    b2 = jnp.zeros((hidden_size,), dtype=jnp.float32)
    b3 = jnp.zeros((output_size,), dtype=jnp.float32)
    return (w2_f, b2, w3_f, b3)


@jax.jit
def bp_forward(params, x):
    w2_f, b2, w3_f, b3 = params
    hidden = jnp.maximum(jnp.dot(x, w2_f.T) + b2, 0.0)
    output = jnp.dot(hidden, w3_f.T) + b3
    return hidden, output


@jax.jit
def bp_mse_loss(params, x, target):
    _, preds = bp_forward(params, x)
    return jnp.mean((target - preds) ** 2)


@jax.jit
def bp_train_step(params, x, target, learning_rate):
    grads = jax.grad(bp_mse_loss)(params, x, target)
    return jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)


class _BPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        *,
        hidden_size: int,
        param_lr: float,
        n_train_iters: int,
        batch_size: int,
        init_type: str,
        seed: int,
        print_every: int = 0,
    ):
        self.hidden_size = int(hidden_size)
        self.param_lr = float(param_lr)
        self.n_train_iters = int(n_train_iters)
        self.batch_size = int(batch_size)
        self.init_type = str(init_type)
        self.seed = int(seed)
        self.print_every = int(print_every)
        self.params_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)

        n_classes = int(np.max(y) + 1)
        self.classes_ = np.arange(n_classes, dtype=np.int32)
        self.n_features_in_ = int(X.shape[1])
        y_oh = np.eye(n_classes, dtype=np.float32)[y]

        train_x = jnp.asarray(X)
        train_y = jnp.asarray(y_oh)

        key_seed = jr.PRNGKey(self.seed)
        key_seed_width = jr.fold_in(key_seed, int(self.hidden_size))
        init_key, data_key = jr.split(key_seed_width, 2)

        input_size, output_size = 4, 3
        params = bp_init_params(
            init_key, input_size, self.hidden_size, output_size, init_type=self.init_type
        )

        for step in range(self.n_train_iters):
            data_key, batch_key = jr.split(data_key)
            x_batch, y_batch = sample_batch_from_fixed(
                batch_key, train_x, train_y, self.batch_size
            )
            params = bp_train_step(params, x_batch, y_batch, self.param_lr)

            if self.print_every and (
                step % self.print_every == 0 or step == self.n_train_iters - 1
            ):
                tl = float(bp_mse_loss(params, x_batch, y_batch))
                print(f"  [CV fit] step={step} train_loss={tl:.6f}")

        self.params_ = params
        return self

    def predict(self, X):
        if self.params_ is None:
            raise RuntimeError("Estimator is not fit yet.")
        X = np.asarray(X, dtype=np.float32)
        x = jnp.asarray(X)
        _, preds = bp_forward(self.params_, x)
        return np.asarray(jnp.argmax(preds, axis=1), dtype=np.int32)

    def predict_proba(self, X):
        if self.params_ is None:
            raise RuntimeError("Estimator is not fit yet.")
        X = np.asarray(X, dtype=np.float32)
        x = jnp.asarray(X)
        _, logits = bp_forward(self.params_, x)
        probs = jax.nn.softmax(logits, axis=1)
        return np.asarray(probs, dtype=np.float32)

    def score(self, X, y):
        if self.params_ is None:
            raise RuntimeError("Estimator is not fit yet.")
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)
        pred_labels = self.predict(X)
        return float(np.mean(pred_labels == y))


class _PDMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        *,
        hidden_size: int,
        param_lr: float,
        n_train_iters: int,
        batch_size: int,
        n_relax_iters: int,
        thetaf: float,
        use_lateral: bool,
        normalise_fwd_weights: bool,
        init_type: str,
        lateral_init_type: str,
        seed: int,
        print_every: int = 0,
    ):
        self.hidden_size = int(hidden_size)
        self.param_lr = float(param_lr)
        self.n_train_iters = int(n_train_iters)
        self.batch_size = int(batch_size)
        self.n_relax_iters = int(n_relax_iters)
        self.thetaf = float(thetaf)
        self.use_lateral = bool(use_lateral)
        self.normalise_fwd_weights = bool(normalise_fwd_weights)
        self.init_type = str(init_type)
        self.lateral_init_type = str(lateral_init_type)
        self.seed = int(seed)
        self.print_every = int(print_every)
        self.weights_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)

        n_classes = int(np.max(y) + 1)
        self.classes_ = np.arange(n_classes, dtype=np.int32)
        self.n_features_in_ = int(X.shape[1])
        y_oh = np.eye(n_classes, dtype=np.float32)[y]

        train_x = jnp.asarray(X)
        train_y = jnp.asarray(y_oh)

        key_seed = jr.PRNGKey(self.seed)
        key_seed_width = jr.fold_in(key_seed, int(self.hidden_size))
        init_key, data_key = jr.split(key_seed_width, 2)

        input_size, output_size = 4, 3
        thetab = 1.0 - self.thetaf

        weights = lateral_pdm_init_weights(
            init_key,
            input_size,
            self.hidden_size,
            output_size,
            init_type=self.init_type,
            lateral_init_type=self.lateral_init_type,
        )
        if self.normalise_fwd_weights:
            weights = lateral_pdm_normalise_fwd_weights(weights)

        if not self.use_lateral:
            weights = (
                weights[0],
                weights[1],
                weights[2],
                weights[3],
                jnp.zeros_like(weights[4]),
            )

        for step in range(self.n_train_iters):
            data_key, batch_key = jr.split(data_key)
            x_batch, y_batch = sample_batch_from_fixed(
                batch_key, train_x, train_y, self.batch_size
            )
            weights = lateral_pdm_train_step(
                weights,
                x_batch,
                y_batch,
                learning_rate=self.param_lr,
                thetaf=self.thetaf,
                thetab=thetab,
                n_relax_iters=self.n_relax_iters,
                freeze_lateral=not self.use_lateral,
            )
            if self.normalise_fwd_weights:
                weights = lateral_pdm_normalise_fwd_weights(weights)

            if self.print_every and (
                step % self.print_every == 0 or step == self.n_train_iters - 1
            ):
                tl = float(
                    lateral_pdm_mse_loss(
                        weights, x_batch, y_batch, self.thetaf, thetab, self.n_relax_iters
                    )
                )
                print(f"  [CV fit] step={step} train_loss={tl:.6f}")

        self.weights_ = weights
        return self

    def predict(self, X):
        if self.weights_ is None:
            raise RuntimeError("Estimator is not fit yet.")
        X = np.asarray(X, dtype=np.float32)
        x = jnp.asarray(X)
        preds = lateral_pdm_predict(
            x, self.weights_, self.thetaf, 1.0 - self.thetaf, max_iters=self.n_relax_iters
        )
        return np.asarray(jnp.argmax(preds, axis=1), dtype=np.int32)

    def predict_proba(self, X):
        if self.weights_ is None:
            raise RuntimeError("Estimator is not fit yet.")
        X = np.asarray(X, dtype=np.float32)
        x = jnp.asarray(X)
        logits = lateral_pdm_predict(
            x, self.weights_, self.thetaf, 1.0 - self.thetaf, max_iters=self.n_relax_iters
        )
        probs = jax.nn.softmax(logits, axis=1)
        return np.asarray(probs, dtype=np.float32)

    def score(self, X, y):
        if self.weights_ is None:
            raise RuntimeError("Estimator is not fit yet.")
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)
        pred_labels = self.predict(X)
        return float(np.mean(pred_labels == y))


def _pdm_train_val_rankme(
    weights, x_batch, y_batch, val_x, val_y, thetaf, thetab, n_relax_iters
):
    w1_b, w2_f, w2_b, w3_f, w2_lat = weights
    train_loss = float(
        lateral_pdm_mse_loss(weights, x_batch, y_batch, thetaf, thetab, n_relax_iters)
    )
    val_loss = lateral_pdm_mse_loss(weights, val_x, val_y, thetaf, thetab, n_relax_iters)
    preds = lateral_pdm_predict(val_x, weights, thetaf, thetab, max_iters=n_relax_iters)
    pred_labels = jnp.argmax(preds, axis=1)
    true_labels = jnp.argmax(val_y, axis=1)
    val_acc = jnp.mean(pred_labels == true_labels)
    # Forward-pass (feedforward) hidden state used to initialize relaxation.
    x2_fwd = jnp.maximum(jnp.dot(val_x, w2_f.T), 0.0)
    x2_val = lateral_pdm_relax_training_fixed_iters(
        val_x, val_y, weights, thetaf, thetab, n_relax_iters
    )
    rankme_fwd = rankme(np.array(x2_fwd))
    rankme_val = rankme(np.array(x2_val))
    # Also return representations/logits for separability diagnostics.
    logits_relaxed = np.asarray(preds, dtype=np.float32)
    logits_fwd = np.asarray(jnp.dot(x2_fwd, w3_f.T), dtype=np.float32)
    return (
        train_loss,
        float(val_loss),
        float(val_acc),
        float(rankme_val),
        float(rankme_fwd),
        np.asarray(x2_val, dtype=np.float32),
        np.asarray(x2_fwd, dtype=np.float32),
        logits_relaxed,
        logits_fwd,
        np.asarray(w3_f, dtype=np.float32),
    )


def _bp_train_val_rankme(params, x_batch, y_batch, val_x, val_y):
    train_loss = float(bp_mse_loss(params, x_batch, y_batch))
    hidden_val, preds = bp_forward(params, val_x)
    val_loss = float(jnp.mean((val_y - preds) ** 2))
    pred_labels = jnp.argmax(preds, axis=1)
    true_labels = jnp.argmax(val_y, axis=1)
    val_acc = float(jnp.mean(pred_labels == true_labels))
    rankme_val = float(rankme(np.array(hidden_val)))
    logits = np.asarray(preds, dtype=np.float32)
    w3_f = np.asarray(params[2], dtype=np.float32)
    return train_loss, val_loss, val_acc, rankme_val, np.asarray(hidden_val, dtype=np.float32), logits, w3_f


def train_pdm(
    *,
    init_key,
    data_key,
    n_train_iters,
    batch_size,
    hidden_size,
    param_lr,
    n_relax_iters,
    thetaf,
    print_every,
    use_lateral,
    normalise_fwd_weights,
    init_type,
    lateral_init_type,
    save_dir,
    fixed_data,
    data_pca_components2: np.ndarray | None = None,
    data_pca_explained_var_ratio: np.ndarray | None = None,
    data_pca2_train: np.ndarray | None = None,
    y_train_idx: np.ndarray | None = None,
):
    input_size, output_size = 4, 3
    thetab = 1.0 - thetaf

    weights = lateral_pdm_init_weights(
        init_key,
        input_size,
        hidden_size,
        output_size,
        init_type=init_type,
        lateral_init_type=lateral_init_type,
    )
    if normalise_fwd_weights:
        weights = lateral_pdm_normalise_fwd_weights(weights)

    if not use_lateral:
        weights = (
            weights[0],
            weights[1],
            weights[2],
            weights[3],
            jnp.zeros_like(weights[4]),
        )

    tracked_w2f_weights = np.empty((n_train_iters + 1, hidden_size, input_size), dtype=np.float32)
    train_losses, val_losses, val_accuracies, rankmes, rankmes_fwd = [], [], [], [], []
    tracked_lateral_weights = []
    tracked_lateral_steps = []

    train_x, train_y, val_x, val_y = fixed_data
    tracked_w2f_weights[0] = np.asarray(weights[1], dtype=np.float32)
    if use_lateral:
        tracked_lateral_weights.append(np.asarray(weights[4], dtype=np.float32))
        tracked_lateral_steps.append(0)

    data_key, init_batch_key = jr.split(data_key)
    x_init, y_init = sample_batch_from_fixed(init_batch_key, train_x, train_y, batch_size)
    init_tl, init_vl, init_va, init_rm, init_rm_fwd, *_ = _pdm_train_val_rankme(
        weights, x_init, y_init, val_x, val_y, thetaf, thetab, n_relax_iters
    )
    train_losses.append(init_tl)
    val_losses.append(init_vl)
    val_accuracies.append(init_va)
    rankmes.append(init_rm)
    rankmes_fwd.append(init_rm_fwd)
    print(
        f"  Init     | Train loss: {init_tl:.4f} | "
        f"Val loss: {init_vl:.4f} | Val acc: {init_va:.4f} | RankMe: {init_rm:.4f}"
    )

    for step in range(n_train_iters):
        data_key, batch_key = jr.split(data_key)
        x_batch, y_batch = sample_batch_from_fixed(batch_key, train_x, train_y, batch_size)

        weights = lateral_pdm_train_step(
            weights,
            x_batch,
            y_batch,
            learning_rate=param_lr,
            thetaf=thetaf,
            thetab=thetab,
            n_relax_iters=n_relax_iters,
            freeze_lateral=not use_lateral,
        )
        if normalise_fwd_weights:
            weights = lateral_pdm_normalise_fwd_weights(weights)

        tracked_w2f_weights[step + 1] = np.asarray(weights[1], dtype=np.float32)

        train_loss, val_loss, val_acc, rankme_val, rankme_fwd, *_ = _pdm_train_val_rankme(
            weights, x_batch, y_batch, val_x, val_y, thetaf, thetab, n_relax_iters
        )
        train_losses.append(train_loss)

        if step % print_every == 0 or step == n_train_iters - 1:
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            rankmes.append(rankme_val)
            rankmes_fwd.append(rankme_fwd)
            if use_lateral:
                tracked_lateral_weights.append(np.asarray(weights[4], dtype=np.float32))
                tracked_lateral_steps.append(step + 1)
            print(
                f"  Step {step:4d} | "
                f"Train loss: {train_loss:.4f} | "
                f"Val loss: {val_loss:.4f} | "
                f"Val acc: {val_acc:.4f} | "
                f"RankMe: {rankme_val:.4f}"
            )

    if save_dir:
        # Compute separability diagnostics on the full held-out set (val_x/val_y).
        y_val_idx = np.asarray(jnp.argmax(val_y, axis=1), dtype=np.int64)

        tracked_lateral_weights_arr = None
        tracked_lateral_steps_arr = None
        if use_lateral:
            tracked_lateral_weights_arr = np.stack(tracked_lateral_weights, axis=0)
            tracked_lateral_steps_arr = np.asarray(tracked_lateral_steps, dtype=np.int64)

        # Input->hidden weight projections onto the data PCs (recorded over training).
        w2f_data_pca2 = None
        w2f_data_pca2_steps = None
        if data_pca_components2 is not None:
            if print_every and int(print_every) > 0:
                base = np.arange(0, n_train_iters + 1, int(print_every), dtype=np.int64)
                steps = np.unique(np.concatenate([base, np.array([n_train_iters], dtype=np.int64)]))
            else:
                steps = np.array([0, n_train_iters], dtype=np.int64)
            w2f_data_pca2_steps = steps
            w2f_data_pca2 = _project_w2_over_steps(tracked_w2f_weights, steps, data_pca_components2)
        save_training_metrics(
            save_dir,
            tracked_w2f_weights,
            train_losses,
            val_losses,
            val_accuracies,
            rankmes,
            n_train_iters=n_train_iters,
            rankmes_fwd=rankmes_fwd,
            tracked_lateral_weights=tracked_lateral_weights_arr,
            tracked_lateral_steps=tracked_lateral_steps_arr,
            separability_metrics={
                "model": np.array("pdm"),
                "y_val": y_val_idx.astype(np.int64),
                "x_train_pca2": (
                    np.asarray(data_pca2_train, dtype=np.float32)
                    if data_pca2_train is not None
                    else np.full((0, 2), np.nan, dtype=np.float32)
                ),
                "y_train": (
                    np.asarray(y_train_idx, dtype=np.int64)
                    if y_train_idx is not None
                    else np.full((0,), 0, dtype=np.int64)
                ),
                "data_pca_components2": (
                    np.asarray(data_pca_components2, dtype=np.float32)
                    if data_pca_components2 is not None
                    else np.full((2, input_size), np.nan, dtype=np.float32)
                ),
                "data_pca_explained_var_ratio": (
                    np.asarray(data_pca_explained_var_ratio, dtype=np.float32)
                    if data_pca_explained_var_ratio is not None
                    else np.full((2,), np.nan, dtype=np.float32)
                ),
                "w2f_data_pca2_steps": (
                    np.asarray(w2f_data_pca2_steps, dtype=np.int64)
                    if w2f_data_pca2_steps is not None
                    else np.full((0,), 0, dtype=np.int64)
                ),
                "w2f_data_pca2": (
                    np.asarray(w2f_data_pca2, dtype=np.float32)
                    if w2f_data_pca2 is not None
                    else np.full((0, hidden_size, 2), np.nan, dtype=np.float32)
                ),
            },
        )

    return {
        "train_losses": np.array(train_losses),
        "val_losses": np.array(val_losses),
        "val_accuracies": np.array(val_accuracies),
        "rankmes": np.array(rankmes),
        "rankmes_fwd": np.array(rankmes_fwd),
        "weights": weights,
        "tracked_w2f_weights": tracked_w2f_weights,
    }


def train_bp(
    *,
    init_key,
    data_key,
    n_train_iters,
    batch_size,
    hidden_size,
    param_lr,
    init_type,
    print_every,
    save_dir,
    fixed_data,
    data_pca_components2: np.ndarray | None = None,
    data_pca_explained_var_ratio: np.ndarray | None = None,
    data_pca2_train: np.ndarray | None = None,
    y_train_idx: np.ndarray | None = None,
):
    input_size, output_size = 4, 3
    params = bp_init_params(init_key, input_size, hidden_size, output_size, init_type=init_type)

    tracked_w2f_weights = np.empty((n_train_iters + 1, hidden_size, input_size), dtype=np.float32)
    train_losses, val_losses, val_accuracies, rankmes = [], [], [], []

    train_x, train_y, val_x, val_y = fixed_data
    tracked_w2f_weights[0] = np.asarray(params[0], dtype=np.float32)

    data_key, init_batch_key = jr.split(data_key)
    x_init, y_init = sample_batch_from_fixed(init_batch_key, train_x, train_y, batch_size)
    init_tl, init_vl, init_va, init_rm, *_ = _bp_train_val_rankme(
        params, x_init, y_init, val_x, val_y
    )
    train_losses.append(init_tl)
    val_losses.append(init_vl)
    val_accuracies.append(init_va)
    rankmes.append(init_rm)
    print(
        f"  Init     | Train loss: {init_tl:.4f} | "
        f"Val loss: {init_vl:.4f} | Val acc: {init_va:.4f} | RankMe: {init_rm:.4f}"
    )

    for step in range(n_train_iters):
        data_key, batch_key = jr.split(data_key)
        x_batch, y_batch = sample_batch_from_fixed(batch_key, train_x, train_y, batch_size)

        params = bp_train_step(params, x_batch, y_batch, param_lr)
        tracked_w2f_weights[step + 1] = np.asarray(params[0], dtype=np.float32)

        train_loss, val_loss, val_acc, rankme_val, *_ = _bp_train_val_rankme(
            params, x_batch, y_batch, val_x, val_y
        )
        train_losses.append(train_loss)

        if step % print_every == 0 or step == n_train_iters - 1:
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            rankmes.append(rankme_val)
            print(
                f"  Step {step:4d} | "
                f"Train loss: {train_loss:.4f} | "
                f"Val loss: {val_loss:.4f} | "
                f"Val acc: {val_acc:.4f} | "
                f"RankMe: {rankme_val:.4f}"
            )

    if save_dir:
        # Separability diagnostics on the full held-out set (val_x/val_y).
        y_val_idx = np.asarray(jnp.argmax(val_y, axis=1), dtype=np.int64)

        w2f_data_pca2 = None
        w2f_data_pca2_steps = None
        if data_pca_components2 is not None:
            if print_every and int(print_every) > 0:
                base = np.arange(0, n_train_iters + 1, int(print_every), dtype=np.int64)
                steps = np.unique(np.concatenate([base, np.array([n_train_iters], dtype=np.int64)]))
            else:
                steps = np.array([0, n_train_iters], dtype=np.int64)
            w2f_data_pca2_steps = steps
            w2f_data_pca2 = _project_w2_over_steps(tracked_w2f_weights, steps, data_pca_components2)

        save_training_metrics(
            save_dir,
            tracked_w2f_weights,
            train_losses,
            val_losses,
            val_accuracies,
            rankmes,
            n_train_iters=n_train_iters,
            separability_metrics={
                "model": np.array("backprop"),
                "y_val": y_val_idx.astype(np.int64),
                "x_train_pca2": (
                    np.asarray(data_pca2_train, dtype=np.float32)
                    if data_pca2_train is not None
                    else np.full((0, 2), np.nan, dtype=np.float32)
                ),
                "y_train": (
                    np.asarray(y_train_idx, dtype=np.int64)
                    if y_train_idx is not None
                    else np.full((0,), 0, dtype=np.int64)
                ),
                "data_pca_components2": (
                    np.asarray(data_pca_components2, dtype=np.float32)
                    if data_pca_components2 is not None
                    else np.full((2, input_size), np.nan, dtype=np.float32)
                ),
                "data_pca_explained_var_ratio": (
                    np.asarray(data_pca_explained_var_ratio, dtype=np.float32)
                    if data_pca_explained_var_ratio is not None
                    else np.full((2,), np.nan, dtype=np.float32)
                ),
                "w2f_data_pca2_steps": (
                    np.asarray(w2f_data_pca2_steps, dtype=np.int64)
                    if w2f_data_pca2_steps is not None
                    else np.full((0,), 0, dtype=np.int64)
                ),
                "w2f_data_pca2": (
                    np.asarray(w2f_data_pca2, dtype=np.float32)
                    if w2f_data_pca2 is not None
                    else np.full((0, hidden_size, 2), np.nan, dtype=np.float32)
                ),
            },
        )

    return {
        "train_losses": np.array(train_losses),
        "val_losses": np.array(val_losses),
        "val_accuracies": np.array(val_accuracies),
        "rankmes": np.array(rankmes),
        "params": params,
        "tracked_w2f_weights": tracked_w2f_weights,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/iris")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--widths", type=int, nargs="+", default=[16])
    parser.add_argument(
        "--param_lrs",
        type=float,
        nargs="+",
        default=[5e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
    )
    parser.add_argument("--n_train_iters", type=int, default=5000)
    parser.add_argument("--pdm_relax_iters", type=int, default=100)
    parser.add_argument("--thetafs", type=float, nargs="+", default=[0.5, 0.9, 0.95, 0.99, 1.0])
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument(
        "--init_type",
        type=str, 
        default="kaiming",
        choices=["glorot", "kaiming", "lecun"],
    )
    parser.add_argument("--lateral_init_type", type=str, default="zero")
    parser.add_argument("--cv_folds", type=int, default=5)
    args = parser.parse_args()

    fixed_data, (X_train_np, y_train_idx, _X_test_np, _y_test_idx) = load_iris_dataset()
    data_pca = _data_pca2_basis(X_train_np)
    data_pca_components2 = data_pca["components2"]
    data_pca_explained_var_ratio = data_pca["explained_var_ratio"]
    X_train_pca2 = (np.asarray(X_train_np, dtype=np.float32) @ data_pca_components2.T).astype(
        np.float32
    )

    for normalise_fwd_weights in [False, True]:  #False, True
        norm_name = "norm_fwd" if normalise_fwd_weights else "no_norm_fwd"
        norm_results_dir = os.path.join(args.results_dir, norm_name)
        
        #False, True
        lateral_modes = [False, True] if normalise_fwd_weights else [False]
        for use_lateral in lateral_modes:
            mode_name = "with_lateral" if use_lateral else "no_lateral"
            mode_results_dir = os.path.join(norm_results_dir, mode_name)

            for hidden_size in args.widths:
                width_results_dir = os.path.join(mode_results_dir, f"width_{hidden_size}")
                os.makedirs(width_results_dir, exist_ok=True)

                init_and_data_keys_by_seed = {}
                for seed in range(args.n_seeds):
                    key_seed = jr.PRNGKey(seed)
                    key_seed_width = jr.fold_in(key_seed, hidden_size)
                    init_key, data_key = jr.split(key_seed_width, 2)
                    init_and_data_keys_by_seed[seed] = (init_key, data_key)

                for thetaf in args.thetafs:
                    thetaf_results_dir = os.path.join(width_results_dir, f"thetaf_{thetaf:g}")
                    os.makedirs(thetaf_results_dir, exist_ok=True)

                    print("\n" + "#" * 60)
                    print(
                        f"Global LR grid-search for lateral PDM: width = {hidden_size}, "
                        f"thetaf = {thetaf}, use_lateral = {use_lateral}, "
                        f"normalise_fwd_weights = {normalise_fwd_weights}, "
                        f"init_type = {args.init_type}, lateral_init_type = {args.lateral_init_type}"
                    )
                    print("#" * 60)

                    param_lrs = list(args.param_lrs)
                    scores_by_lr = {lr: [] for lr in param_lrs}

                    for seed in range(args.n_seeds):
                        cv = StratifiedKFold(
                            n_splits=int(args.cv_folds),
                            shuffle=True,
                            random_state=int(seed),
                        )
                        estimator = _PDMClassifier(
                            hidden_size=hidden_size,
                            param_lr=float(param_lrs[0]),
                            n_train_iters=args.n_train_iters,
                            batch_size=args.batch_size,
                            n_relax_iters=args.pdm_relax_iters,
                            thetaf=thetaf,
                            use_lateral=use_lateral,
                            normalise_fwd_weights=normalise_fwd_weights,
                            init_type=args.init_type,
                            lateral_init_type=args.lateral_init_type,
                            seed=seed,
                            print_every=0,
                        )
                        gs = GridSearchCV(
                            estimator=estimator,
                            param_grid={"param_lr": param_lrs},
                            scoring="accuracy",
                            cv=cv,
                            refit=False,
                            n_jobs=1,
                            verbose=0,
                        )
                        gs.fit(X_train_np, y_train_idx)
                        cv_res = gs.cv_results_

                        n_splits = int(args.cv_folds)
                        for i, lr in enumerate(param_lrs):
                            for split_idx in range(n_splits):
                                key = f"split{split_idx}_test_score"
                                if key in cv_res:
                                    scores_by_lr[lr].append(float(cv_res[key][i]))

                    lr_means = {}
                    lr_stds = {}
                    for lr, scores in scores_by_lr.items():
                        if not scores:
                            continue
                        arr = np.asarray(scores, dtype=np.float32)
                        lr_means[lr] = float(np.mean(arr))
                        lr_stds[lr] = float(np.std(arr))

                    if not lr_means:
                        print(
                            f"No CV scores collected for width={hidden_size}, thetaf={thetaf}, "
                            f"use_lateral={use_lateral}; skipping."
                        )
                        continue

                    best_lr = max(lr_means, key=lambda lr: lr_means[lr])
                    print(
                        "Global LR selection (PDM): "
                        f"best_lr = {best_lr:g}, mean_acc = {lr_means[best_lr]:.4f}, "
                        f"std_acc = {lr_stds[best_lr]:.4f}"
                    )

                    np.save(
                        os.path.join(thetaf_results_dir, "gridsearch_lrs.npy"),
                        np.array(param_lrs, dtype=np.float32),
                    )
                    np.save(
                        os.path.join(thetaf_results_dir, "gridsearch_lr_means.npy"),
                        np.array([lr_means.get(lr, np.nan) for lr in param_lrs]),
                    )
                    np.save(
                        os.path.join(thetaf_results_dir, "gridsearch_lr_stds.npy"),
                        np.array([lr_stds.get(lr, np.nan) for lr in param_lrs]),
                    )
                    np.save(
                        os.path.join(thetaf_results_dir, "gridsearch_best_lr.npy"),
                        np.array(best_lr, dtype=np.float32),
                    )

                    for seed in range(args.n_seeds):
                        print("\n" + "#" * 60)
                        print(
                            f"Final Iris lateral PDM run for width = {hidden_size}, "
                            f"thetaf = {thetaf}, seed = {seed}, use_lateral = {use_lateral}, "
                            f"normalise_fwd_weights = {normalise_fwd_weights}, "
                            f"param_lr = {best_lr:g}"
                        )
                        print("#" * 60)

                        run_results_dir = os.path.join(thetaf_results_dir, f"seed_{seed}")
                        os.makedirs(run_results_dir, exist_ok=True)

                        lr_results_dir = os.path.join(run_results_dir, f"best_lr_{best_lr:g}")
                        os.makedirs(lr_results_dir, exist_ok=True)

                        save_dir = os.path.join(lr_results_dir, "lateral_pdm")
                        init_key, data_key = init_and_data_keys_by_seed[seed]
                        _ = train_pdm(
                            init_key=init_key,
                            data_key=data_key,
                            n_train_iters=args.n_train_iters,
                            batch_size=args.batch_size,
                            hidden_size=hidden_size,
                            param_lr=best_lr,
                            n_relax_iters=args.pdm_relax_iters,
                            thetaf=thetaf,
                            print_every=args.print_every,
                            use_lateral=use_lateral,
                            normalise_fwd_weights=normalise_fwd_weights,
                            init_type=args.init_type,
                            lateral_init_type=args.lateral_init_type,
                            save_dir=save_dir,
                            fixed_data=fixed_data,
                            data_pca_components2=data_pca_components2,
                            data_pca_explained_var_ratio=data_pca_explained_var_ratio,
                        )

    mode_results_dir = os.path.join(args.results_dir, "backprop")
    for hidden_size in args.widths:
        width_results_dir = os.path.join(mode_results_dir, f"width_{hidden_size}")
        os.makedirs(width_results_dir, exist_ok=True)

        init_and_data_keys_by_seed = {}
        for seed in range(args.n_seeds):
            key_seed = jr.PRNGKey(seed)
            key_seed_width = jr.fold_in(key_seed, hidden_size)
            init_key, data_key = jr.split(key_seed_width, 2)
            init_and_data_keys_by_seed[seed] = (init_key, data_key)

        print("\n" + "#" * 60)
        print(f"Global LR grid-search for backprop: width = {hidden_size}")
        print("#" * 60)

        param_lrs = list(args.param_lrs)
        scores_by_lr = {lr: [] for lr in param_lrs}

        for seed in range(args.n_seeds):
            cv = StratifiedKFold(
                n_splits=int(args.cv_folds),
                shuffle=True,
                random_state=int(seed),
            )
            estimator = _BPClassifier(
                hidden_size=hidden_size,
                param_lr=float(param_lrs[0]),
                n_train_iters=args.n_train_iters,
                batch_size=args.batch_size,
                init_type=args.init_type,
                seed=seed,
                print_every=0,
            )
            gs = GridSearchCV(
                estimator=estimator,
                param_grid={"param_lr": param_lrs},
                scoring="accuracy",
                cv=cv,
                refit=False,
                n_jobs=1,
                verbose=0,
            )
            gs.fit(X_train_np, y_train_idx)
            cv_res = gs.cv_results_

            n_splits = int(args.cv_folds)
            for i, lr in enumerate(param_lrs):
                for split_idx in range(n_splits):
                    key = f"split{split_idx}_test_score"
                    if key in cv_res:
                        scores_by_lr[lr].append(float(cv_res[key][i]))

        lr_means = {}
        lr_stds = {}
        for lr, scores in scores_by_lr.items():
            if not scores:
                continue
            arr = np.asarray(scores, dtype=np.float32)
            lr_means[lr] = float(np.mean(arr))
            lr_stds[lr] = float(np.std(arr))

        if not lr_means:
            print(f"No CV scores collected for backprop width={hidden_size}; skipping.")
            continue

        best_lr = max(lr_means, key=lambda lr: lr_means[lr])
        print(
            "Global LR selection (backprop): "
            f"best_lr = {best_lr:g}, mean_acc = {lr_means[best_lr]:.4f}, "
            f"std_acc = {lr_stds[best_lr]:.4f}"
        )

        np.save(os.path.join(width_results_dir, "gridsearch_lrs.npy"), np.array(param_lrs, dtype=np.float32))
        np.save(
            os.path.join(width_results_dir, "gridsearch_lr_means.npy"),
            np.array([lr_means.get(lr, np.nan) for lr in param_lrs]),
        )
        np.save(
            os.path.join(width_results_dir, "gridsearch_lr_stds.npy"),
            np.array([lr_stds.get(lr, np.nan) for lr in param_lrs]),
        )
        np.save(os.path.join(width_results_dir, "gridsearch_best_lr.npy"), np.array(best_lr, dtype=np.float32))

        for seed in range(args.n_seeds):
            print("\n" + "#" * 60)
            print(
                f"Final Iris backprop run for width = {hidden_size}, seed = {seed}, "
                f"param_lr = {best_lr:g}"
            )
            print("#" * 60)

            run_results_dir = os.path.join(width_results_dir, f"seed_{seed}")
            os.makedirs(run_results_dir, exist_ok=True)

            lr_results_dir = os.path.join(run_results_dir, f"best_lr_{best_lr:g}")
            os.makedirs(lr_results_dir, exist_ok=True)

            save_dir = os.path.join(lr_results_dir, "backprop")
            init_key, data_key = init_and_data_keys_by_seed[seed]
            _ = train_bp(
                init_key=init_key,
                data_key=data_key,
                n_train_iters=args.n_train_iters,
                batch_size=args.batch_size,
                hidden_size=hidden_size,
                param_lr=best_lr,
                init_type=args.init_type,
                print_every=args.print_every,
                save_dir=save_dir,
                fixed_data=fixed_data,
                data_pca_components2=data_pca_components2,
                data_pca_explained_var_ratio=data_pca_explained_var_ratio,
                data_pca2_train=X_train_pca2,
                y_train_idx=y_train_idx,
            )


if __name__ == "__main__":
    main()
