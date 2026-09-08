import jax.numpy as jnp
import equinox as eqx


def normalize_generator_first_layer_weight_rows(generator):
    """Normalise forward weights of the generator's first layer to unit-norm rows."""
    weight = generator[0][1].weight
    norms = jnp.linalg.norm(weight, axis=1, keepdims=True)
    normalized_weight = weight / jnp.maximum(norms, 1e-8)
    return eqx.tree_at(lambda m: m[0][1].weight, generator, normalized_weight)


def normalize_amortiser_output_layer_weight_rows(amortiser):
    """Normalise backward weights of the amortiser's output layer to unit-norm rows."""
    weight = amortiser[0][1].weight
    norms = jnp.linalg.norm(weight, axis=1, keepdims=True)
    normalized_weight = weight / jnp.maximum(norms, 1e-8)
    return eqx.tree_at(lambda m: m[0][1].weight, amortiser, normalized_weight)
