import math
import torch
import numpy as np

from activations import hard_sigmoid, get_activation_derivative
from torch_utils import outer_prod_broadcasting, get_device, mark_step_if_xla

torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

class ContrastivePredictiveErrorPropagationFullClamp:
    """
    Predictive Entropy Maximization with strong output clamping.

    This class implements the "no contrastive rule / strong-clamp" variant:
    - Free phase: output is not clamped and beta is effectively zero.
    - Nudged phase: output layer is fixed exactly to the one-hot label.
    - Learning rules are still phase-difference rules, but there is no weak beta scaling.

    Notation:
        architecture = [N_0, N_1, ..., N_L]
        Wff[j] maps layer j -> layer j+1
        Wfb[k] maps layer k+1 -> layer k, for k=1,...,L-1
        C[j] and mu[j] store second-order statistics for non-input layer j+1
    """

    def __init__(
        self,
        architecture,
        lambda_=0.99999,
        epsilon=0.15,
        gamma_forward=0.5,
        gamma_backward=0.5,
        activation=hard_sigmoid,
        device=None,
        use_gating=False,
    ):
        self.architecture = list(architecture)
        self.lambda_ = lambda_
        self.epsilon = epsilon
        # Mixing weights of the two predictions (should sum to 1).
        self.gamma_forward = gamma_forward
        self.gamma_backward = gamma_backward
        self.activation = activation
        self.device = get_device() if device is None else device

        # Optional per-phase gating of prediction errors by g = f'(x) at that
        # phase's post-synaptic state. Off by default.
        self.use_gating = use_gating
        self.activation_derivative = (
            get_activation_derivative(activation) if use_gating else None
        )

        self.num_non_input_layers = len(self.architecture) - 1

        self.Wff = []
        self.Wfb = [None for _ in range(self.num_non_input_layers)]
        self.C = []
        self.mu = []

        self.forward_backward_angles = []

        # Feedforward Synapses Initialization
        Wff = []
        for idx in range(len(architecture)-1):
            weight = torch.randn(architecture[idx + 1], architecture[idx], requires_grad = False).to(self.device)
            torch.nn.init.xavier_uniform_(weight)

            Wff.append({'weight': weight})
        self.Wff = np.array(Wff)
        
        # Feedback Synapses Initialization
        Wfb = []
        for idx in range(len(architecture)-1):
            weight = torch.eye(architecture[idx], architecture[idx + 1], requires_grad = False).to(self.device)
            torch.nn.init.xavier_uniform_(weight)

            Wfb.append({'weight': weight})
        self.Wfb = np.array(Wfb)

        # Replace B with C for local updates (compared to CorInfoMaxHopfield model above)
        C = []
        mu = []
        for idx in range(len(architecture)-1):
            # Using 1.0 * identity for covariance initialization
            weight_C = 0 * torch.zeros(architecture[idx + 1], architecture[idx + 1], requires_grad=False).to(self.device)
            # torch.nn.init.xavier_uniform_(weight_C)
            # weight_C = weight_C @ weight_C.T
            C.append({'weight': weight_C})
            mu.append({'weight': torch.zeros(architecture[idx + 1], 1, requires_grad=False).to(self.device)})
        self.C = np.array(C)
        self.mu = np.array(mu)

        self.forward_backward_angles = []

    ###############################################################
    ############### HELPER METHODS ################################
    ###############################################################
    def copy_neurons(self, neurons):
        copy = []
        for n in neurons:
            copy.append(torch.empty_like(n).copy_(n.data))#.requires_grad_())
        return copy
        
    def init_neurons(self, mbs, random_initialize = False, device = None):
        if device is None:
            device = self.device
        # Initializing the neurons
        if random_initialize:
            neurons = []
            append = neurons.append
            for size in self.architecture[1:]:  
                append(torch.randn((mbs, size), requires_grad=False, device=device).T)       
        else:
            neurons = []
            append = neurons.append
            for size in self.architecture[1:]:  
                append(torch.zeros((mbs, size), requires_grad=False, device=device).T)
        return neurons

    ###############################################################
    ############### REQUIRED FUNCTIONS FOR DEBUGGING ##############
    ###############################################################
    def angle_between_two_matrices(self, A, B):
        """Computes the angle between two matrices A and B.

        Args:
            A (torch.Tensor): Pytorch tensor of size m times n
            B (torch.Tensor): Pytorch tensor of size m times n

        Returns:
            angle: angle between the matrices A and B. The formula is given by the following:
                (180/pi) * acos[ Tr(A @ B.T) / sqrt(Tr(A @ A.T) * Tr(B @ B.T))] 
        """

        angle = (180 / torch.pi) * torch.acos(torch.trace(A @ B.T) / torch.sqrt(torch.trace(A @ A.T) * torch.trace(B @ B.T)))
        return angle

    @staticmethod
    def angle_between_two_matrices(A, B, eps=1e-12):
        """
        Diagnostic angle in degrees between two matrices.
        """
        a = A.reshape(-1)
        b = B.reshape(-1)

        denom = torch.norm(a) * torch.norm(b) + eps
        cosang = torch.clamp(torch.dot(a, b) / denom, -1.0, 1.0)
        return torch.rad2deg(torch.acos(cosang))

    # ------------------------------------------------------------------
    # Fast neural dynamics
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
        Fast neural dynamics.

        Args:
            x:
                Input batch, shape (N_0, batch_size).
            y:
                One-hot label batch, shape (N_L, batch_size).
            neurons:
                List of non-input layer activities.
            output_clamp:
                If False, run the free phase.
                If True, run the strong-clamp nudged phase where the output
                layer is fixed to y at every neural-dynamics iteration.

        Returns:
            Updated neurons.
        """
        Wff = self.Wff
        Wfb = self.Wfb
        C = self.C
        mu = self.mu

        lambda_ = self.lambda_
        epsilon = self.epsilon
        gamma_forward = self.gamma_forward
        gamma_backward = self.gamma_backward

        # neurons_intermediate = self.copy_neurons(neurons)

        with torch.no_grad():

            # In strong-clamp phase, clamp output immediately so that hidden
            # layers receive target-driven feedback from the beginning.
            if output_clamp:
                neurons[-1] = y.detach().clone()
                # neurons_intermediate[-1] = y.detach().clone()

            for iter_count in range(neural_dynamic_iterations):

                if lr_rule == "constant":
                    neural_lr = neural_lr_start
                elif lr_rule == "divide_by_loop_index":
                    neural_lr = max(neural_lr_start / (iter_count + 1), neural_lr_stop)
                elif lr_rule == "divide_by_slow_loop_index":
                    neural_lr = max(
                        neural_lr_start / (iter_count * lr_decay_multiplier + 1.0),
                        neural_lr_stop,
                    )
                else:
                    raise ValueError(f"Unknown lr_rule: {lr_rule}")

                # Keep output clamped at every iteration.
                if output_clamp:
                    neurons[-1] = y.detach().clone()

                layers = [x] + neurons

                for jj in range(len(neurons)):

                    # Skip output-layer dynamics during strong-clamp phase.
                    if output_clamp and (jj == len(neurons) - 1):
                        neurons[-1] = y.detach().clone()
                        layers = [x] + neurons
                        continue

                    y_layer = layers[jj + 1]

                    mu_y = mu[jj]["weight"]
                    C_y = C[jj]["weight"]

                    y_bar = y_layer - mu_y

                    D_y = torch.diag(C_y).view(-1, 1)
                    O_y = C_y - torch.diag(D_y.squeeze())

                    D_reg = D_y + epsilon

                    # PEM entropy drive:
                    # self/variance expansion minus covariance-based inhibition.
                    lateral_term = (1.0 - lambda_) * (
                        y_bar / D_reg
                        -
                        (O_y @ (y_bar / D_reg)) / D_reg
                    )

                    if jj == len(neurons) - 1:
                        # Free-phase output dynamics.
                        gradient_neurons = (
                            lateral_term
                            -
                            gamma_forward
                            * (layers[jj + 1] - Wff[jj]["weight"] @ layers[jj])
                        )

                    else:
                        # Hidden-layer dynamics: feedforward prediction,
                        # feedback prediction, and entropy drive.
                        gradient_neurons = (
                            2.0 * lateral_term
                            -
                            (
                                layers[jj + 1]
                                - gamma_forward * (Wff[jj]["weight"] @ layers[jj])
                                - gamma_backward * (Wfb[jj + 1]["weight"] @ layers[jj + 2])
                            )
                        )

                    # neurons_intermediate[jj] = neurons_intermediate[jj] + neural_lr * gradient_neurons
                    # neurons[jj] = self.activation(neurons_intermediate[jj])
                    neurons[jj] = self.activation(neurons[jj] + neural_lr * gradient_neurons)

                    layers = [x] + neurons

                mark_step_if_xla()

        return neurons

    # ------------------------------------------------------------------
    # One minibatch training step
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
        neural_dynamic_iterations_free=20,
        neural_dynamic_iterations_nudged=10,
        beta=None,
        take_debug_logs=False,
        weight_decay=False,
    ):
        """
        One minibatch update using free phase + strong-clamped nudged phase.

        beta is accepted only for compatibility with old training scripts.
        It is not used in this strong-clamp variant.
        """
        Wff = self.Wff
        Wfb = self.Wfb
        C = self.C
        mu = self.mu

        lambda_ = self.lambda_
        epsilon = self.epsilon

        batch_size = x.size(1)
        neurons = self.init_neurons(batch_size, device=self.device)

        # --------------------------------------------------------------
        # Free phase: no label information
        # --------------------------------------------------------------
        neurons = self.run_neural_dynamics(
            x=x,
            y=y,
            neurons=neurons,
            neural_lr_start=neural_lr_start,
            neural_lr_stop=neural_lr_stop,
            lr_rule=neural_lr_rule,
            lr_decay_multiplier=neural_lr_decay_multiplier,
            neural_dynamic_iterations=neural_dynamic_iterations_free,
            output_clamp=False,
        )

        neurons_free = self.copy_neurons(neurons)

        # --------------------------------------------------------------
        # Strong-clamped phase: output fixed to label
        # --------------------------------------------------------------
        neurons = self.run_neural_dynamics(
            x=x,
            y=y,
            neurons=neurons,
            neural_lr_start=neural_lr_start,
            neural_lr_stop=neural_lr_stop,
            lr_rule=neural_lr_rule,
            lr_decay_multiplier=neural_lr_decay_multiplier,
            neural_dynamic_iterations=neural_dynamic_iterations_nudged,
            output_clamp=True,
        )

        neurons_clamped = self.copy_neurons(neurons)

        layers_free = [x] + neurons_free
        layers_clamped = [x] + neurons_clamped

        # --------------------------------------------------------------
        # Forward prediction errors
        # --------------------------------------------------------------
        forward_errors_free = [
            layers_free[jj + 1] - (Wff[jj]["weight"] @ layers_free[jj])
            for jj in range(len(Wff))
        ]

        forward_errors_clamped = [
            layers_clamped[jj + 1] - (Wff[jj]["weight"] @ layers_clamped[jj])
            for jj in range(len(Wff))
        ]

        # --------------------------------------------------------------
        # Backward prediction errors
        # --------------------------------------------------------------
        backward_errors_free = [
            layers_free[jj] - (Wfb[jj]["weight"] @ layers_free[jj + 1])
            for jj in range(1, len(Wfb))
        ]

        backward_errors_clamped = [
            layers_clamped[jj] - (Wfb[jj]["weight"] @ layers_clamped[jj + 1])
            for jj in range(1, len(Wfb))
        ]

        with torch.no_grad():

            # ----------------------------------------------------------
            # Feedforward weight updates
            # ----------------------------------------------------------
            for jj in range(len(Wff)):
                ff_free = forward_errors_free[jj]
                ff_clamped = forward_errors_clamped[jj]
                if self.use_gating:
                    # Gate each phase's error by g = f'(.) at that phase's
                    # post-synaptic state (layer jj+1): Δw ~ (e_x g^clamp - e_x g^free).
                    ff_free = self.activation_derivative(layers_free[jj + 1]) * ff_free
                    ff_clamped = self.activation_derivative(layers_clamped[jj + 1]) * ff_clamped

                free_update = torch.mean(
                    outer_prod_broadcasting(ff_free.T, layers_free[jj].T),
                    dim=0,
                )

                clamped_update = torch.mean(
                    outer_prod_broadcasting(ff_clamped.T, layers_clamped[jj].T),
                    dim=0,
                )

                # Strong-clamp phase-difference rule.
                Wff[jj]["weight"] += lr["ff"][jj] * (clamped_update - free_update)

                if weight_decay:
                    Wff[jj]["weight"] -= lr["ff"][jj] * epsilon * Wff[jj]["weight"]

            # ----------------------------------------------------------
            # Feedback weight updates
            # ----------------------------------------------------------
            for jj in range(1, len(Wfb)):
                bwd_free = backward_errors_free[jj - 1]
                bwd_clamped = backward_errors_clamped[jj - 1]
                if self.use_gating:
                    # Gate each phase's error by g = f'(.) at that phase's
                    # post-synaptic state (layer jj): Δw ~ (e_x g^clamp - e_x g^free).
                    bwd_free = self.activation_derivative(layers_free[jj]) * bwd_free
                    bwd_clamped = self.activation_derivative(layers_clamped[jj]) * bwd_clamped

                free_update = torch.mean(
                    outer_prod_broadcasting(bwd_free.T, layers_free[jj + 1].T),
                    dim=0,
                )

                clamped_update = torch.mean(
                    outer_prod_broadcasting(bwd_clamped.T, layers_clamped[jj + 1].T),
                    dim=0,
                )

                Wfb[jj]["weight"] += lr["fb"][jj] * (clamped_update - free_update)

                if weight_decay:
                    Wfb[jj]["weight"] -= lr["fb"][jj] * epsilon * Wfb[jj]["weight"]

            # ----------------------------------------------------------
            # Lateral second-order statistics and mean updates
            # ----------------------------------------------------------
            for jj in range(len(C)):
                C_update = torch.mean(
                    outer_prod_broadcasting(neurons_clamped[jj].T, neurons_clamped[jj].T),
                    dim=0,
                )

                C[jj]["weight"] = lambda_ * C[jj]["weight"] + (1.0 - lambda_) * C_update

                mu[jj]["weight"] = (
                    lambda_ * mu[jj]["weight"]
                    +
                    (1.0 - lambda_) * torch.mean(neurons_clamped[jj], dim=1, keepdim=True)
                )

        self.Wff = Wff
        self.Wfb = Wfb
        self.C = C
        self.mu = mu

        if take_debug_logs:
            instant_forward_backward_angles = []

            for jj in range(1, len(Wff)):
                if Wfb[jj] is not None:
                    instant_forward_backward_angles.append(
                        self.angle_between_two_matrices(
                            self.Wff[jj]["weight"],
                            self.Wfb[jj]["weight"].T,
                        ).item()
                    )

            self.forward_backward_angles.append(instant_forward_backward_angles)

        mark_step_if_xla()
        return neurons_clamped

    # ------------------------------------------------------------------
    # Inference / prediction utilities
    # ------------------------------------------------------------------
    def infer_free(
        self,
        x,
        neural_lr_start,
        neural_lr_stop,
        lr_rule="constant",
        lr_decay_multiplier=0.1,
        neural_dynamic_iterations=30,
    ):
        """
        Free-phase inference used for evaluation.
        """
        batch_size = x.size(1)
        dummy_y = torch.zeros(self.architecture[-1], batch_size, device=self.device)

        neurons = self.init_neurons(batch_size, device=self.device)

        neurons = self.run_neural_dynamics(
            x=x,
            y=dummy_y,
            neurons=neurons,
            neural_lr_start=neural_lr_start,
            neural_lr_stop=neural_lr_stop,
            lr_rule=lr_rule,
            lr_decay_multiplier=lr_decay_multiplier,
            neural_dynamic_iterations=neural_dynamic_iterations,
            output_clamp=False,
        )

        return neurons

    def predict(
        self,
        x,
        neural_lr_start,
        neural_lr_stop,
        lr_rule="constant",
        lr_decay_multiplier=0.1,
        neural_dynamic_iterations=30,
    ):
        """
        Return class predictions from free-phase output activities.

        Args:
            x: Tensor of shape (N_0, batch_size)
        """
        neurons = self.infer_free(
            x=x,
            neural_lr_start=neural_lr_start,
            neural_lr_stop=neural_lr_stop,
            lr_rule=lr_rule,
            lr_decay_multiplier=lr_decay_multiplier,
            neural_dynamic_iterations=neural_dynamic_iterations,
        )

        output_activity = neurons[-1]
        return torch.argmax(output_activity, dim=0)