import torch
import numpy as np

from activations import hard_sigmoid, get_activation_derivative
from torch_utils import outer_prod_broadcasting, get_device, mark_step_if_xla

torch.pi = torch.acos(torch.zeros(1)).item() * 2


class PredictiveErrorPropagationWeakClampNoFreePhase:
    """
    No-free-phase predictive error propagation.

    Difference from ContrastivePredictiveErrorPropagationWeakClamp:
        - There is no free phase.
        - The network runs only a label-nudged phase.
        - Weight updates use local predictive-error Hebbian terms from the nudged state only.
        - There is no contrastive subtraction and no 1 / beta scaling.

    Recommended training use:
        beta = 1.0
        use_random_sign_beta = False
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

        # Feedforward synapses
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

        # Feedback synapses
        # Wfb[0] is unused because the input layer is clamped.
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

    ###############################################################
    # Helper methods
    ###############################################################
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

    ###############################################################
    # Debugging diagnostics
    ###############################################################
    def angle_between_two_matrices(self, A, B):
        """
        Computes the angle in degrees between two matrices.
        """
        denom = torch.sqrt(torch.trace(A @ A.T) * torch.trace(B @ B.T)) + 1e-12
        cosine = torch.clamp(torch.trace(A @ B.T) / denom, -1.0, 1.0)
        angle = (180 / torch.pi) * torch.acos(cosine)
        return angle

    ###############################################################
    # Neural dynamics
    ###############################################################
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
        beta=1,
    ):
        """
        Runs neural dynamics with optional output nudging.

        Args:
            beta = 0:
                Free inference/evaluation.
            beta > 0:
                Label-nudged inference.
        """
        Wff = self.Wff
        Wfb = self.Wfb
        gamma_forward = self.gamma_forward
        gamma_backward = self.gamma_backward

        layers = [x] + neurons

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
                raise ValueError(f"Unknown lr_rule: {lr_rule}")

            with torch.no_grad():
                for jj in range(len(neurons)):

                    if jj == len(neurons) - 1:
                        # Output layer: feedforward prediction plus supervised nudging.
                        gradient_neurons = (
                            - gamma_forward
                            * (layers[jj + 1] - Wff[jj]["weight"] @ layers[jj])
                            - beta * (layers[jj + 1] - y)
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

                    # Refresh layer list after each update, matching the original implementation.
                    layers = [x] + neurons

            mark_step_if_xla()

        return neurons

    ###############################################################
    # One minibatch update: no free phase
    ###############################################################
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
        beta=1,
        take_debug_logs=False,
        weight_decay=False,
    ):
        """
        One minibatch update without a free phase.

        The network runs only the label-nudged dynamics, then updates:
            Wff <- Wff + lr_ff * <forward_error_nudged pre_activity_nudged^T>
            Wfb <- Wfb + lr_fb * <backward_error_nudged top_activity_nudged^T>
        """
        Wff, Wfb = self.Wff, self.Wfb

        # For this no-free-phase model, nudging should attract the output toward the label.
        # This avoids accidental negative nudging if an old script still samples random-sign beta.
        beta_eff = abs(beta)

        neurons = self.init_neurons(x.size(1), device=self.device)

        # Only nudged phase
        neurons = self.run_neural_dynamics(
            x,
            y,
            neurons,
            neural_lr_start,
            neural_lr_stop,
            neural_lr_rule,
            neural_lr_decay_multiplier,
            neural_dynamic_iterations_nudged,
            beta=beta_eff,
        )

        neurons_nudged = self.copy_neurons(neurons)
        layers_nudged = [x] + neurons_nudged

        # Forward prediction errors
        forward_errors_nudged = [
            layers_nudged[jj + 1] - (Wff[jj]["weight"] @ layers_nudged[jj])
            for jj in range(len(Wff))
        ]

        # Backward prediction errors
        backward_errors_nudged = [
            layers_nudged[jj] - (Wfb[jj]["weight"] @ layers_nudged[jj + 1])
            for jj in range(1, len(Wfb))
        ]

        with torch.no_grad():

            # Feedforward weight updates
            for jj in range(len(Wff)):
                ff_nudged = forward_errors_nudged[jj]
                if self.use_gating:
                    # Gate each sample's update by g = f'(.) at the nudged
                    # post-synaptic state (layer jj+1); zeroes saturated units.
                    ff_nudged = self.activation_derivative(layers_nudged[jj + 1]) * ff_nudged
                Wff[jj]["weight"] += lr["ff"][jj] * torch.mean(
                    outer_prod_broadcasting(
                        ff_nudged.T,
                        layers_nudged[jj].T,
                    ),
                    dim=0,
                )

                if weight_decay:
                    Wff[jj]["weight"] -= lr["ff"][jj] * Wff[jj]["weight"]

            # Feedback weight updates
            for jj in range(1, len(Wfb)):
                bwd_nudged = backward_errors_nudged[jj - 1]
                if self.use_gating:
                    # Gate each sample's update by g = f'(.) at the nudged
                    # post-synaptic state (layer jj).
                    bwd_nudged = self.activation_derivative(layers_nudged[jj]) * bwd_nudged
                Wfb[jj]["weight"] += lr["fb"][jj] * torch.mean(
                    outer_prod_broadcasting(
                        bwd_nudged.T,
                        layers_nudged[jj + 1].T,
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
        return neurons_nudged


