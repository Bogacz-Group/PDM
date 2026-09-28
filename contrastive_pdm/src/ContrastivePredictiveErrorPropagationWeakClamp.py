import torch
import numpy as np

from activations import hard_sigmoid, get_activation_derivative
from torch_utils import outer_prod_broadcasting, get_device, mark_step_if_xla

torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

class ContrastivePredictiveErrorPropagationWeakClamp():

    def __init__(self, architecture, 
                 gamma_forward,
                 gamma_backward,
                 activation = hard_sigmoid,
                 device = None,
                 use_gating = False):

        self.architecture = architecture
        self.activation = activation
        # Optional per-phase gating of prediction errors by g = f'(x) at that
        # phase's post-synaptic state. Off by default (no behaviour change).
        self.use_gating = use_gating
        self.activation_derivative = (
            get_activation_derivative(activation) if use_gating else None
        )
        self.device = get_device() if device is None else device
        
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

        # Mixing weights of the two predictions (should sum to 1).
        self.gamma_forward = gamma_forward
        self.gamma_backward = gamma_backward

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
    
    def run_neural_dynamics(self, x, y, neurons, neural_lr_start, neural_lr_stop, 
                            lr_rule = "constant", lr_decay_multiplier = 0.1, 
                            neural_dynamic_iterations = 10, beta = 1):

    
        Wff = self.Wff
        Wfb = self.Wfb
        gamma_forward = self.gamma_forward
        gamma_backward = self.gamma_backward

        # neurons_intermediate = self.copy_neurons(neurons)
        layers = [x] + neurons  # concatenate the input to other layers
        for iter_count in range(neural_dynamic_iterations):

            if lr_rule == "constant":
                neural_lr = neural_lr_start
            elif lr_rule == "divide_by_loop_index":
                neural_lr = max(neural_lr_start / (iter_count + 1), neural_lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                neural_lr = max(neural_lr_start / (iter_count * lr_decay_multiplier + 1), neural_lr_stop)

            with torch.no_grad():       
                for jj in range(len(neurons)):
                    y_layer = layers[jj + 1]

                    if jj == len(neurons) - 1:
                        gradient_neurons = - gamma_forward * (layers[jj + 1] - Wff[jj]['weight'] @ layers[jj]) - beta * (layers[jj + 1] - y)
                        neurons[jj] = self.activation(neurons[jj] + neural_lr * gradient_neurons)
                        
                    else:
                        gradient_neurons = - (
                            layers[jj + 1]
                            - gamma_forward * (Wff[jj]['weight'] @ layers[jj])
                            - gamma_backward * (Wfb[jj + 1]['weight'] @ layers[jj + 2])
                        )
                        neurons[jj] = self.activation(neurons[jj] + neural_lr * gradient_neurons)
                    layers = [x] + neurons  # concatenate the input to other layers
            mark_step_if_xla()

        return neurons

    def batch_step_hopfield(self, x, y, lr, neural_lr_start, neural_lr_stop, neural_lr_rule = "constant", 
                            neural_lr_decay_multiplier = 0.1, neural_dynamic_iterations_free = 20, 
                            neural_dynamic_iterations_nudged = 10, beta = 1,
                            take_debug_logs = False, weight_decay = False):

        Wff, Wfb = self.Wff, self.Wfb

        neurons = self.init_neurons(x.size(1), device = self.device)

        # Free Phase
        neurons = self.run_neural_dynamics( x, y, neurons, neural_lr_start, 
                                            neural_lr_stop, neural_lr_rule, 
                                            neural_lr_decay_multiplier, 
                                            neural_dynamic_iterations_free, 
                                            beta = 0, )
        neurons1 = neurons.copy()
        
        # Nudged Phase
        neurons = self.run_neural_dynamics( x, y, neurons, neural_lr_start, 
                                            neural_lr_stop, neural_lr_rule, 
                                            neural_lr_decay_multiplier, 
                                            neural_dynamic_iterations_nudged, 
                                            beta = beta, )
        neurons2 = neurons.copy()

        layers_free = [x] + neurons1
        layers_nudged = [x] + neurons2

        ## Compute forward errors
        forward_errors_free = [layers_free[jj + 1] - (Wff[jj]['weight'] @ layers_free[jj]) for jj in range(len(Wff))]
        forward_errors_nudged = [layers_nudged[jj + 1] - (Wff[jj]['weight'] @ layers_nudged[jj]) for jj in range(len(Wff))]
        ## Compute backward errors
        backward_errors_free = [(layers_free[jj]) - (Wfb[jj]['weight'] @ layers_free[jj + 1]) for jj in range(1, len(Wfb))]
        backward_errors_nudged = [(layers_nudged[jj]) - (Wfb[jj]['weight'] @ layers_nudged[jj + 1]) for jj in range(1, len(Wfb))]

        ### Update Synapses ###
        
        ### Feedforward Weight Updates
        for jj in range(len(Wff)):
            ff_free = forward_errors_free[jj]
            ff_nudged = forward_errors_nudged[jj]
            if self.use_gating:
                # Gate each phase's error by g = f'(.) at that phase's
                # post-synaptic state (layer jj+1): Δw ~ (e_x g^nudge - e_x g^free).
                ff_free = self.activation_derivative(layers_free[jj + 1]) * ff_free
                ff_nudged = self.activation_derivative(layers_nudged[jj + 1]) * ff_nudged
            Wff[jj]['weight'] += -(1/(beta)) * lr['ff'][jj] * torch.mean(outer_prod_broadcasting(ff_free.T, layers_free[jj].T) - outer_prod_broadcasting(ff_nudged.T, layers_nudged[jj].T), axis = 0)

        ### Feedback Weight Updates 
        for jj in range(1, len(Wfb)):
            bwd_free = backward_errors_free[jj - 1]
            bwd_nudged = backward_errors_nudged[jj - 1]
            if self.use_gating:
                # Gate each phase's error by g = f'(.) at that phase's
                # post-synaptic state (layer jj): Δw ~ (e_x g^nudge - e_x g^free).
                bwd_free = self.activation_derivative(layers_free[jj]) * bwd_free
                bwd_nudged = self.activation_derivative(layers_nudged[jj]) * bwd_nudged
            Wfb[jj]['weight'] += -(1/(beta)) * lr['fb'][jj] * torch.mean(outer_prod_broadcasting(bwd_free.T, layers_free[jj + 1].T) - outer_prod_broadcasting(bwd_nudged.T, layers_nudged[jj + 1].T), axis = 0)

        self.Wff, self.Wfb = Wff, Wfb

        if take_debug_logs:
            instant_forward_backward_angles = []
            for jj in range(1, len(Wff)):
                instant_forward_backward_angles.append(self.angle_between_two_matrices(self.Wff[jj]['weight'], self.Wfb[jj]['weight'].T).item())
            
            self.forward_backward_angles.append(instant_forward_backward_angles)

        mark_step_if_xla()
        return neurons2

