import torch.nn as nn
global visualization, hooks

visualization = {}
hooks = []

def hook_act(m, i, o):
    min, _ = o.flatten(start_dim=2).min(dim=-1)
    max, _ = o.flatten(start_dim=2).max(dim=-1)
    visualization[m] = (o - min.view(o.size(0), -1, 1, 1)) / (max.view(o.size(0), -1, 1, 1) - min.view(o.size(0), -1, 1, 1) + 1e-8)

def hook_forward_output(m, i, o):
    visualization[m] = o

def hook_forward_norm(m, i, o):
    visualization[m] = o.flatten(start_dim=1).norm(p=2, dim=1).mean()

def hook_backward_norm(m, i, o):
    visualization[m] = i[1].flatten(start_dim=1).norm(p=2, dim=1).mean() # i[1]: weights' grad

def backward_norms_hook(model, instance=nn.Conv2d):
    for name, layer in model._modules.items():
        if isinstance(layer, nn.Sequential):
            backward_norms_hook(layer, instance)
        elif isinstance(layer, instance):
            hooks.append(layer.register_backward_hook(hook_backward_norm))

def forward_output_hook(model, instance=(nn.ReLU, nn.LeakyReLU, nn.Linear)):
    for name, layer in model._modules.items():
        if isinstance(layer, (nn.Sequential)):
            forward_output_hook(layer, instance)
        elif isinstance(layer, instance):
            layer.register_forward_hook(hook_forward_output)

def forward_activations_hook(model, instance=(nn.ReLU, nn.LeakyReLU)):
    for name, layer in model._modules.items():
        if isinstance(layer, (nn.Sequential)):
            forward_activations_hook(layer, instance)
        elif isinstance(layer, instance):
            layer.register_forward_hook(hook_act)

def get_visualization():
    return visualization

def remove_hooks():
    for i, hook in enumerate(hooks):
        hook.remove()
        # del hooks[i]
