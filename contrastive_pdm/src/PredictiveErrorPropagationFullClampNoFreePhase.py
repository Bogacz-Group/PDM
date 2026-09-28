import torch
import numpy as np

from activations import hard_sigmoid, get_activation_derivative
from torch_utils import outer_prod_broadcasting, get_device, mark_step_if_xla

torch.pi = torch.acos(torch.zeros(1)).item() * 2


class PredictiveErrorPropagationFullClampNoFreePhase:
    """
    Predictive error propagation — no free phase, full output clamping, no contrastive rule.

    Training:
        - No free phase.
        - The output layer is hard-clamped to the one-hot label at every neural-dynamics
          iteration.  Only hidden layers run dynamics, driven by feedforward and feedback
          prediction errors (feedback coming from the clamped output).
        - Weight update: local Hebbian terms from the clamped state only — no phase
          difference, no 1/beta scaling.

    Evaluation:
        - Free inference: all layers (including output) are initialised to zero and
          run unconstrained neural dynamics.
        - Classification by argmax of the output neuron activities.

    Args:
        architecture:   List of layer sizes [N_0, N_1, ..., N_L].
        gamma_forward:  Mixing weight on the feedforward prediction (with
                        gamma_backward, should sum to 1).
        gamma_backward: Mixing weight on the feedback prediction.
        activation:     Pointwise activation (default: hard_sigmoid).
        use_gating:     If True, gate each synaptic update by g = f'(.) at the
                        clamped post-synaptic state (per sample).
        device:         torch.device or None (auto-selects xla / cuda:0 / cpu).
    """

    def __init__(
        self,
        architecture,
        gamma_forward,
        gamma_backward,
        activation=hard_sigmoid,
        device=None,
        use_gating=False,
    ):
        self.architecture = architecture
        self.activation = activation
        self.use_gating = use_gating
        self.activation_derivative = (
            get_activation_derivative(activation) if use_gating else None
        )

        self.device = get_device() if device is None else device

        # Feedforward synapses: Wff[j] maps layer j -> layer j+1
        Wff = []
        for idx in range(len(architecture) - 1):
            weight = torch.randn(
                architecture[idx + 1],
                architecture[idx],
                requires_grad=False,
            ).to(self.device)
            torch.nn.init.xavier_uniform_(weight)
            Wff.append({"weight": weight})
        self.Wff = np.array(Wff)

        # Feedback synapses: Wfb[j] maps layer j+1 -> layer j.
        # Wfb[0] is unused because the input layer is always clamped.
        Wfb = []
        for idx in range(len(architecture) - 1):
            weight = torch.eye(
                architecture[idx],
                architecture[idx + 1],
                requires_grad=False,
            ).to(self.device)
            torch.nn.init.xavier_uniform_(weight)
            Wfb.append({"weight": weight})
        self.Wfb = np.array(Wfb)

        # Mixing weights of the two predictions (should sum to 1).
        self.gamma_forward = gamma_forward
        self.gamma_backward = gamma_backward

        self.forward_backward_angles = []

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def copy_neurons(self, neurons):
        return [torch.empty_like(n).copy_(n.data) for n in neurons]

    def init_neurons(self, mbs, random_initialize=False, device=None):
        if device is None:
            device = self.device
        neurons = []
        for size in self.architecture[1:]:
            if random_initialize:
                neurons.append(torch.randn((mbs, size), requires_grad=False, device=device).T)
            else:
                neurons.append(torch.zeros((mbs, size), requires_grad=False, device=device).T)
        return neurons

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def angle_between_two_matrices(self, A, B):
        denom = torch.sqrt(torch.trace(A @ A.T) * torch.trace(B @ B.T)) + 1e-12
        cosine = torch.clamp(torch.trace(A @ B.T) / denom, -1.0, 1.0)
        return (180 / torch.pi) * torch.acos(cosine)

    # ------------------------------------------------------------------
    # Neural dynamics
    # ------------------------------------------------------------------
    def run_neural_dynamics(
        self,
        x,
        y,
        neurons,
        neural_lr_start,
        neural_lr_stop,
        lr_rule="constant",
        lr_decay_multiplier=0.1,
        neural_dynamic_iterations=10,
        output_clamp=False,
    ):
        """
        Neural dynamics with optional hard output clamping.

        Args:
            x:    Input batch, shape (N_0, batch_size).
            y:    One-hot label batch, shape (N_L, batch_size).
                  Only used when output_clamp=True; pass 0 otherwise.
            neurons:
                  List of non-input layer activities, each (N_l, batch_size).
            output_clamp (bool):
                  True  → training mode: neurons[-1] is fixed to y throughout;
                          only hidden layers run dynamics.
                  False → evaluation mode: all layers relax freely.
        """
        Wff = self.Wff
        Wfb = self.Wfb
        gamma_forward = self.gamma_forward
        gamma_backward = self.gamma_backward

        with torch.no_grad():
            if output_clamp:
                neurons[-1] = y.detach().clone()

            for iter_count in range(neural_dynamic_iterations):

                if lr_rule == "constant":
                    neural_lr = neural_lr_start
                elif lr_rule == "divide_by_loop_index":
                    neural_lr = max(neural_lr_start / (iter_count + 1), neural_lr_stop)
                elif lr_rule == "divide_by_slow_loop_index":
                    neural_lr = max(
                        neural_lr_start / (iter_count * lr_decay_multiplier + 1),
                        neural_lr_stop,
                    )
                else:
                    raise ValueError(f"Unknown lr_rule: {lr_rule!r}")

                if output_clamp:
                    neurons[-1] = y.detach().clone()

                layers = [x] + neurons

                for jj in range(len(neurons)):

                    if output_clamp and jj == len(neurons) - 1:
                        # Output is clamped; skip its dynamics.
                        neurons[-1] = y.detach().clone()
                        layers = [x] + neurons
                        continue

                    if jj == len(neurons) - 1:
                        # Free output dynamics (evaluation): only feedforward error.
                        gradient_neurons = (
                            - gamma_forward
                            * (layers[jj + 1] - Wff[jj]["weight"] @ layers[jj])
                        )
                    else:
                        # Hidden layers: convex mix of feedforward and feedback predictions.
                        gradient_neurons = (
                            - (
                                layers[jj + 1]
                                - gamma_forward * (Wff[jj]["weight"] @ layers[jj])
                                - gamma_backward * (Wfb[jj + 1]["weight"] @ layers[jj + 2])
                            )
                        )

                    neurons[jj] = self.activation(neurons[jj] + neural_lr * gradient_neurons)
                    layers = [x] + neurons

                mark_step_if_xla()

        return neurons

    # ------------------------------------------------------------------
    # One minibatch update: full clamp, no free phase
    # ------------------------------------------------------------------
    def batch_step_hopfield(
        self,
        x,
        y,
        lr,
        neural_lr_start,
        neural_lr_stop,
        neural_lr_rule="constant",
        neural_lr_decay_multiplier=0.1,
        neural_dynamic_iterations_nudged=10,
        take_debug_logs=False,
        weight_decay=False,
    ):
        """
        One minibatch update with full output clamping.

        The output is hard-clamped to the one-hot label y.  Hidden layers relax
        under feedforward and feedback prediction errors from the clamped output.
        Weight update (no contrastive subtraction):

            ΔWff[j] += lr_ff[j] * mean( e_ff_clamped[j]  ⊗  layers_clamped[j]^T )
            ΔWfb[j] += lr_fb[j] * mean( e_fb_clamped[j]  ⊗  layers_clamped[j+1]^T )
        """
        Wff, Wfb = self.Wff, self.Wfb

        neurons = self.init_neurons(x.size(1), device=self.device)

        # Clamped phase: output fixed to label, hidden layers relax.
        neurons = self.run_neural_dynamics(
            x,
            y,
            neurons,
            neural_lr_start,
            neural_lr_stop,
            neural_lr_rule,
            neural_lr_decay_multiplier,
            neural_dynamic_iterations_nudged,
            output_clamp=True,
        )

        neurons_clamped = self.copy_neurons(neurons)
        layers_clamped = [x] + neurons_clamped

        # Forward prediction errors (clamped state)
        forward_errors_clamped = [
            layers_clamped[jj + 1] - (Wff[jj]["weight"] @ layers_clamped[jj])
            for jj in range(len(Wff))
        ]

        # Backward prediction errors (clamped state)
        backward_errors_clamped = [
            layers_clamped[jj] - (Wfb[jj]["weight"] @ layers_clamped[jj + 1])
            for jj in range(1, len(Wfb))
        ]

        with torch.no_grad():
            # Feedforward weight updates
            for jj in range(len(Wff)):
                ff_clamped = forward_errors_clamped[jj]
                if self.use_gating:
                    # Gate each sample's update by g = f'(.) at the clamped
                    # post-synaptic state (layer jj+1); zeroes saturated units.
                    ff_clamped = self.activation_derivative(layers_clamped[jj + 1]) * ff_clamped
                Wff[jj]["weight"] += lr["ff"][jj] * torch.mean(
                    outer_prod_broadcasting(
                        ff_clamped.T,
                        layers_clamped[jj].T,
                    ),
                    dim=0,
                )
                if weight_decay:
                    Wff[jj]["weight"] -= lr["ff"][jj] * Wff[jj]["weight"]

            # Feedback weight updates
            for jj in range(1, len(Wfb)):
                bwd_clamped = backward_errors_clamped[jj - 1]
                if self.use_gating:
                    # Gate each sample's update by g = f'(.) at the clamped
                    # post-synaptic state (layer jj).
                    bwd_clamped = self.activation_derivative(layers_clamped[jj]) * bwd_clamped
                Wfb[jj]["weight"] += lr["fb"][jj] * torch.mean(
                    outer_prod_broadcasting(
                        bwd_clamped.T,
                        layers_clamped[jj + 1].T,
                    ),
                    dim=0,
                )
                if weight_decay:
                    Wfb[jj]["weight"] -= lr["fb"][jj] * Wfb[jj]["weight"]

        self.Wff, self.Wfb = Wff, Wfb

        if take_debug_logs:
            instant_forward_backward_angles = []
            for jj in range(1, len(Wff)):
                instant_forward_backward_angles.append(
                    self.angle_between_two_matrices(
                        self.Wff[jj]["weight"],
                        self.Wfb[jj]["weight"].T,
                    ).item()
                )
            self.forward_backward_angles.append(instant_forward_backward_angles)

        mark_step_if_xla()
        return neurons_clamped
