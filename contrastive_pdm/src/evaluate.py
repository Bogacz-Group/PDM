import torch

from torch_utils import mark_step_if_xla

def evaluateContrastivePredictiveErrorPropagationWeakClamp(model, 
                                                    loader, 
                                                    neural_lr_start, 
                                                    neural_lr_stop, 
                                                    neural_lr_rule, 
                                                    neural_lr_decay_multiplier,
                                                    T, device, printing = True):
    # Evaluate the Contrastive Predictive Error Propagation model on a dataloader with T steps for the dynamics for the classification task
    correct = 0
    # getattr keeps this working when loader.dataset is a Subset (train/val split), which has no `.train`.
    phase = 'Train' if getattr(loader.dataset, "train", False) else 'Test'
    
    for x, y in loader:
        x = x.view(x.size(0),-1).to(device).T
        y = y.to(device)
        
        neurons = model.init_neurons(x.size(1), device = model.device)
        
        # dynamics for T time steps
        neurons = model.run_neural_dynamics(x, 0, neurons, neural_lr_start, neural_lr_stop, neural_lr_rule, neural_lr_decay_multiplier, T, beta = 0) 
        
        pred = torch.argmax(neurons[-1], dim=0).squeeze()
        mark_step_if_xla()
        correct += (y == pred).sum().item()

    acc = correct/len(loader.dataset) 
    if printing:
        print(phase+' accuracy :\t', acc)   
    return acc

def evaluatePredictiveErrorPropagationWeakClampNoFreePhase(
    model,
    loader,
    neural_lr_start,
    neural_lr_stop,
    neural_lr_rule,
    neural_lr_decay_multiplier,
    T,
    device,
    printing=True,
):
    """
    Evaluation uses beta=0, i.e. free inference.
    """
    correct = 0
    phase = "Train" if getattr(loader.dataset, "train", False) else "Test"

    with torch.no_grad():
        for x, y in loader:
            x = x.view(x.size(0), -1).to(device).T
            y = y.to(device)

            neurons = model.init_neurons(x.size(1), device=model.device)

            neurons = model.run_neural_dynamics(
                x,
                0,
                neurons,
                neural_lr_start,
                neural_lr_stop,
                neural_lr_rule,
                neural_lr_decay_multiplier,
                T,
                beta=0,
            )

            pred = torch.argmax(neurons[-1], dim=0).squeeze()
            mark_step_if_xla()
            correct += (y == pred).sum().item()

    acc = correct / len(loader.dataset)

    if printing:
        print(phase + " accuracy :\t", acc)

    return acc

def evaluateContrastivePredictiveEntropyMax(model, 
                                            loader, 
                                            neural_lr_start, 
                                            neural_lr_stop, 
                                            neural_lr_rule, 
                                            neural_lr_decay_multiplier,
                                            T, device, printing = True):
    # Evaluate the Contrastive Predictive Entropy Max model on a dataloader with T steps for the dynamics for the classification task
    correct = 0
    phase = 'Train' if loader.dataset.train else 'Test'
    
    for x, y in loader:
        x = x.view(x.size(0),-1).to(device).T
        y = y.to(device)
        
        neurons = model.init_neurons(x.size(1), device = model.device)
        
        # dynamics for T time steps
        neurons = model.run_neural_dynamics(x, 0, neurons, neural_lr_start, neural_lr_stop, neural_lr_rule, neural_lr_decay_multiplier, T, beta = 0) 
        
        pred = torch.argmax(neurons[-1], dim=0).squeeze()
        mark_step_if_xla()
        correct += (y == pred).sum().item()

    acc = correct/len(loader.dataset) 
    if printing:
        print(phase+' accuracy :\t', acc)   
    return acc


def evaluatePredictiveErrorPropagationFullClampNoFreePhase(
    model,
    loader,
    neural_lr_start,
    neural_lr_stop,
    neural_lr_rule,
    neural_lr_decay_multiplier,
    T,
    device,
    printing=True,
):
    """
    Evaluate PredictiveErrorPropagationFullClampNoFreePhase.

    During evaluation there is no label available, so the output layer is not
    clamped.  All layers (including the output) relax freely from zero
    initialisation (output_clamp=False).  Classification is by argmax of the
    output activities after T neural-dynamics iterations.
    """
    correct = 0
    phase = "Train" if getattr(loader.dataset, "train", False) else "Test"

    with torch.no_grad():
        for x, y in loader:
            x = x.view(x.size(0), -1).to(device).T
            y = y.to(device)

            neurons = model.init_neurons(x.size(1), device=model.device)

            # Free inference: output_clamp=False, y argument is unused.
            neurons = model.run_neural_dynamics(
                x,
                0,
                neurons,
                neural_lr_start,
                neural_lr_stop,
                neural_lr_rule,
                neural_lr_decay_multiplier,
                T,
                output_clamp=False,
            )

            pred = torch.argmax(neurons[-1], dim=0).squeeze()
            mark_step_if_xla()
            correct += (y == pred).sum().item()

    acc = correct / len(loader.dataset)
    if printing:
        print(phase + " accuracy:\t", acc)
    return acc


def evaluateContrastivePredictiveErrorPropagationFullClamp(
    model,
    loader,
    neural_lr_start,
    neural_lr_stop,
    neural_lr_rule,
    neural_lr_decay_multiplier,
    T,
    device,
    printing=True,
):
    """
    Evaluate the strong-clamp Predictive Entropy Max model on a dataloader.

    Evaluation uses the free phase only:
        output_clamp = False

    Args:
        model: ContrastivePredictiveErrorPropagationFullClamp model.
        loader: PyTorch dataloader.
        neural_lr_start: Initial neural dynamics learning rate.
        neural_lr_stop: Minimum neural dynamics learning rate.
        neural_lr_rule: Neural dynamics learning-rate rule.
        neural_lr_decay_multiplier: Neural dynamics decay multiplier.
        T: Number of free-phase neural dynamics iterations.
        device: torch device.
        printing: Whether to print accuracy.

    Returns:
        acc: classification accuracy.
    """
    correct = 0
    total = 0

    phase = "Train" if getattr(loader.dataset, "train", False) else "Test"

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # Flatten images and put data in shape (input_dim, batch_size)
            x = x.view(x.size(0), -1).T

            # Dummy target is unused in free-phase dynamics, but passed for API consistency
            dummy_y = torch.zeros(
                model.architecture[-1],
                x.size(1),
                device=model.device,
                dtype=x.dtype,
            )

            neurons = model.init_neurons(x.size(1), device=model.device)

            neurons = model.run_neural_dynamics(
                x=x,
                y=dummy_y,
                neurons=neurons,
                neural_lr_start=neural_lr_start,
                neural_lr_stop=neural_lr_stop,
                lr_rule=neural_lr_rule,
                lr_decay_multiplier=neural_lr_decay_multiplier,
                neural_dynamic_iterations=T,
                output_clamp=False,
            )

            pred = torch.argmax(neurons[-1], dim=0).squeeze()
            mark_step_if_xla()
            correct += (pred == y).sum().item()
            total += y.numel()

    acc = correct / total

    if printing:
        print(f"{phase} accuracy:\t{acc}")

    return acc
