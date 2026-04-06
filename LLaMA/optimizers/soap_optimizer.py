# SOAP optimizer with group-based parameter handling
# Similar to Muon's approach: SOAP for hidden layers, Adam for others

import torch
import torch.nn.functional as F
import math
from itertools import chain


class SOAP_Grouped(torch.optim.Optimizer):
    """
    SOAP optimizer with separate handling for matrix and scalar parameters.
    Matrix params: SOAP preconditioning (eigenbase projection + Adam)
    Scalar params: AdamW
    Based on https://arxiv.org/abs/2409.11321
    """

    def __init__(self, param_groups, lr_soap=0.003, lr_adam=0.001,
                 betas=(0.9, 0.95), shampoo_beta=-1, eps=1e-10,
                 weight_decay=0.0, precondition_frequency=10,
                 max_precond_dim=10000, merge_dims=False,
                 normalize_grads=False, correct_bias=True):
        defaults = dict(lr_soap=lr_soap, lr_adam=lr_adam,
                        betas=betas, shampoo_beta=shampoo_beta, eps=eps,
                        weight_decay=weight_decay,
                        precondition_frequency=precondition_frequency,
                        max_precond_dim=max_precond_dim,
                        merge_dims=merge_dims,
                        normalize_grads=normalize_grads,
                        correct_bias=correct_bias)
        super(SOAP_Grouped, self).__init__(param_groups, defaults)

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
            is_soap = group.get('is_soap', True)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                param_state = self.state.setdefault(p, {})

                if is_soap and grad.dim() >= 2:
                    # SOAP for 2D+ parameters
                    self._soap_step(p, grad, param_state, group)
                else:
                    # AdamW for 1D/0D parameters
                    self._adam_step(p, grad, param_state, lr, betas, eps, weight_decay, group.get('correct_bias', True))

        return loss

    def _adam_step(self, p, grad, state, lr, betas, eps, weight_decay, correct_bias):
        if 'exp_avg' not in state:
            state['exp_avg'] = torch.zeros_like(grad)
            state['exp_avg_sq'] = torch.zeros_like(grad)
            state['step'] = 0

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        state['step'] += 1
        step = state['step']

        exp_avg.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
        exp_avg_sq.mul_(betas[1]).addcmul_(grad, grad, value=1 - betas[1])

        step_size = lr
        if correct_bias:
            bias_correction1 = 1 - betas[0] ** step
            bias_correction2 = 1 - betas[1] ** step
            step_size = lr * math.sqrt(bias_correction2) / bias_correction1

        denom = exp_avg_sq.sqrt().add_(eps)
        adam_update = exp_avg / denom

        if weight_decay > 0:
            p.data.mul_(1 - step_size * weight_decay)

        p.data.add_(adam_update, alpha=-step_size)

    def _soap_step(self, p, grad, state, group):
        lr = group['lr']
        betas = group.get('betas', (0.9, 0.95))
        eps = group.get('eps', 1e-10)
        weight_decay = group.get('weight_decay', 0.0)
        precondition_frequency = group.get('precondition_frequency', 10)
        max_precond_dim = group.get('max_precond_dim', 10000)
        merge_dims = group.get('merge_dims', False)
        normalize_grads = group.get('normalize_grads', False)
        correct_bias = group.get('correct_bias', True)
        shampoo_beta = group.get('shampoo_beta', -1)
        if shampoo_beta < 0:
            shampoo_beta = betas[1]

        if 'step' not in state:
            state['step'] = 0

        if 'exp_avg' not in state:
            state['exp_avg'] = torch.zeros_like(grad, dtype=torch.float32)
            state['exp_avg_sq'] = torch.zeros_like(grad, dtype=torch.float32)

        if 'Q' not in state:
            self._init_preconditioner(grad, state, precondition_frequency, shampoo_beta, max_precond_dim, merge_dims)
            self._update_preconditioner(grad, state, max_precond_dim, merge_dims)
            return  # skip first step

        # Project gradient to eigenbasis
        grad_projected = self._project(grad, state, max_precond_dim)
        grad_projected = grad_projected.to(state['exp_avg'].dtype)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        state['step'] += 1

        exp_avg.mul_(betas[0]).add_(grad_projected, alpha=1.0 - betas[0])
        exp_avg_sq.mul_(betas[1]).add_(grad_projected.square(), alpha=1.0 - betas[1])

        denom = exp_avg_sq.sqrt().add_(eps)

        step_size = lr
        if correct_bias:
            bias_correction1 = 1.0 - betas[0] ** state['step']
            bias_correction2 = 1.0 - betas[1] ** state['step']
            step_size = lr * (bias_correction2 ** 0.5) / bias_correction1

        # Project back
        norm_grad = self._project_back(exp_avg / denom, state, max_precond_dim)

        if normalize_grads:
            norm_grad = norm_grad / (1e-30 + torch.mean(norm_grad ** 2) ** 0.5)

        norm_grad = norm_grad.to(p.dtype)
        p.data.add_(norm_grad, alpha=-step_size)

        if weight_decay > 0.0:
            p.data.add_(p.data, alpha=-lr * weight_decay)

        self._update_preconditioner(grad, state, max_precond_dim, merge_dims)

    def _init_preconditioner(self, grad, state, precondition_frequency, shampoo_beta, max_precond_dim, merge_dims):
        state['GG'] = []
        for sh in grad.shape:
            if sh > max_precond_dim:
                state['GG'].append([])
            else:
                state['GG'].append(torch.zeros(sh, sh, device=grad.device, dtype=torch.float32))
        state['Q'] = None
        state['precondition_frequency'] = precondition_frequency
        state['shampoo_beta'] = shampoo_beta

    def _project(self, grad, state, max_precond_dim):
        for mat in state['Q']:
            if len(mat) > 0:
                if grad.dtype != mat.dtype:
                    grad = grad.to(mat.dtype)
                grad = torch.tensordot(grad, mat, dims=[[0], [0]])
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)
        return grad

    def _project_back(self, grad, state, max_precond_dim):
        for mat in state['Q']:
            if len(mat) > 0:
                if grad.dtype != mat.dtype:
                    grad = grad.to(mat.dtype)
                grad = torch.tensordot(grad, mat, dims=[[0], [1]])
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)
        return grad

    def _update_preconditioner(self, grad, state, max_precond_dim, merge_dims):
        if state['Q'] is not None:
            state['exp_avg'] = self._project_back(state['exp_avg'], state, max_precond_dim)

        for idx, sh in enumerate(grad.shape):
            if sh <= max_precond_dim:
                outer_product = torch.tensordot(
                    grad, grad,
                    dims=[[*chain(range(idx), range(idx + 1, len(grad.shape)))]] * 2,
                )
                outer_product = outer_product.to(state['GG'][idx].dtype)
                state['GG'][idx].lerp_(outer_product, 1 - state['shampoo_beta'])

        if state['Q'] is None:
            state['Q'] = self._get_orthogonal_matrix(state['GG'])
        elif state['step'] > 0 and state['step'] % state['precondition_frequency'] == 0:
            state['Q'] = self._get_orthogonal_matrix_QR(state, max_precond_dim)

        if state['step'] > 0:
            state['exp_avg'] = self._project(state['exp_avg'], state, max_precond_dim)

    def _get_orthogonal_matrix(self, mat):
        final = []
        for m in mat:
            if len(m) == 0:
                final.append([])
                continue
            m_float = m.data.float()
            try:
                _, Q = torch.linalg.eigh(m_float + 1e-30 * torch.eye(m_float.shape[0], device=m_float.device))
            except:
                _, Q = torch.linalg.eigh(m_float.to(torch.float64) + 1e-30 * torch.eye(m_float.shape[0], device=m_float.device))
                Q = Q.to(torch.float32)
            Q = torch.flip(Q, [1]).float()
            final.append(Q)
        return final

    def _get_orthogonal_matrix_QR(self, state, max_precond_dim):
        precond_list = state['GG']
        orth_list = state['Q']

        orig_shape = state['exp_avg_sq'].shape
        exp_avg_sq = state['exp_avg_sq']

        final = []
        for ind, (m, o) in enumerate(zip(precond_list, orth_list)):
            if len(m) == 0:
                final.append([])
                continue
            m_f = m.data.float()
            o_f = o.data.float()
            est_eig = torch.diag(o_f.T @ m_f @ o_f)
            sort_idx = torch.argsort(est_eig, descending=True)
            exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)
            o_f = o_f[:, sort_idx]
            power_iter = m_f @ o_f
            Q, _ = torch.linalg.qr(power_iter)
            Q = Q.float()
            final.append(Q)

        state['exp_avg_sq'] = exp_avg_sq
        return final


def get_soap_optimizer(model, lr_soap=0.003, lr_adam=0.001, weight_decay=0.1,
                       precondition_frequency=10):
    """
    Returns a SOAP optimizer with separate learning rates.
    Uses Muon-style parameter grouping logic:
    - SOAP for hidden layers (ndim >= 2 and not embed/lm_head)
    - Adam for other parameters (ndim < 2 or embed/lm_head)
    """
    soap_params = []
    adam_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.ndim >= 2 and 'embed' not in name and 'lm_head' not in name:
                soap_params.append(param)
            else:
                adam_params.append(param)

    param_groups = [
        dict(params=soap_params, lr=lr_soap, lr_soap=lr_soap, lr_adam=lr_adam,
             weight_decay=weight_decay, precondition_frequency=precondition_frequency,
             is_soap=True),
        dict(params=adam_params, lr=lr_adam, lr_soap=lr_soap, lr_adam=lr_adam,
             weight_decay=weight_decay, is_soap=False)
    ]
    optimizer = SOAP_Grouped(param_groups)
    return optimizer
