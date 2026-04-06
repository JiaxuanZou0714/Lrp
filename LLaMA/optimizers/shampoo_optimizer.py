# Shampoo optimizer with group-based parameter handling
# Similar to Muon's approach: Shampoo (Adam + NS orthogonalization) for hidden layers, Adam for others

import torch
import torch.nn.functional as F
import math


def zeropower_via_newtonschulz5(G, steps=5):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Shampoo_Grouped(torch.optim.Optimizer):
    """
    Shampoo optimizer with separate handling for matrix and scalar parameters.
    Matrix params: Adam + Newton-Schulz orthogonalization
    Scalar params: AdamW
    """

    def __init__(self, param_groups, lr_shampoo=0.005, lr_adam=0.001,
                 betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0, ns_steps=5):
        defaults = dict(lr_shampoo=lr_shampoo, lr_adam=lr_adam,
                        betas=betas, eps=eps, weight_decay=weight_decay, ns_steps=ns_steps)
        super(Shampoo_Grouped, self).__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            betas = group.get('betas', (0.9, 0.95))
            eps = group.get('eps', 1e-10)
            weight_decay = group.get('weight_decay', 0.0)
            ns_steps = group.get('ns_steps', 5)
            is_shampoo = group.get('is_shampoo', True)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                param_state = self.state.setdefault(p, {})

                if 'exp_avg' not in param_state:
                    param_state['exp_avg'] = torch.zeros_like(grad)
                    param_state['exp_avg_sq'] = torch.zeros_like(grad)
                    param_state['step'] = 0

                exp_avg = param_state['exp_avg']
                exp_avg_sq = param_state['exp_avg_sq']
                param_state['step'] += 1
                step = param_state['step']

                beta1, beta2 = betas
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # Update momentum and squared gradient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute the Adam update
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1

                update = exp_avg / denom

                if is_shampoo and update.ndim >= 2:
                    # Orthogonalize the update via Newton-Schulz
                    update = zeropower_via_newtonschulz5(update, steps=ns_steps)

                # Apply the update
                p.data.add_(update, alpha=-step_size)

                # Apply weight decay
                if weight_decay > 0:
                    p.data.mul_(1 - lr * weight_decay)

        return loss


def get_shampoo_optimizer(model, lr_shampoo=0.005, lr_adam=0.001, weight_decay=0.1):
    """
    Returns a Shampoo optimizer with separate learning rates.
    Uses Muon-style parameter grouping logic:
    - Shampoo for hidden layers (ndim >= 2 and not embed/lm_head)
    - Adam for other parameters (ndim < 2 or embed/lm_head)
    """
    shampoo_params = []
    adam_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.ndim >= 2 and 'embed' not in name and 'lm_head' not in name:
                shampoo_params.append(param)
            else:
                adam_params.append(param)

    param_groups = [
        dict(params=shampoo_params, lr=lr_shampoo, lr_shampoo=lr_shampoo, lr_adam=lr_adam,
             weight_decay=weight_decay, is_shampoo=True),
        dict(params=adam_params, lr=lr_adam, lr_shampoo=lr_shampoo, lr_adam=lr_adam,
             weight_decay=weight_decay, is_shampoo=False)
    ]
    optimizer = Shampoo_Grouped(param_groups)
    return optimizer
