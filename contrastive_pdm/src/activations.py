import torch.nn.functional as F


def hard_sigmoid(x):
    # Source : https://github.com/Laborieux-Axel/Equilibrium-Propagation/
    return (1+F.hardtanh(2*x-1))*0.5


def relu(x):
    return F.relu(x)


# Derivatives take the post-activation state s = f(u) and return f'(u).
def hard_sigmoid_derivative(s):
    return ((s > 0) & (s < 1)).to(s.dtype)


def relu_derivative(s):
    return (s > 0).to(s.dtype)


ACTIVATIONS = {
    "hard_sigmoid": hard_sigmoid,
    "relu": relu,
}

ACTIVATION_DERIVATIVES = {
    "hard_sigmoid": hard_sigmoid_derivative,
    "relu": relu_derivative,
    hard_sigmoid: hard_sigmoid_derivative,
    relu: relu_derivative,
}


def get_activation(name):
    if callable(name):
        return name
    try:
        return ACTIVATIONS[name]
    except KeyError:
        raise ValueError(
            f"Unknown activation '{name}'. Available: {sorted(ACTIVATIONS)}"
        )


def get_activation_derivative(activation):
    try:
        return ACTIVATION_DERIVATIVES[activation]
    except KeyError:
        raise ValueError(
            f"No derivative for {activation!r}. "
            f"Available: {sorted(k for k in ACTIVATION_DERIVATIVES if isinstance(k, str))}"
        )
