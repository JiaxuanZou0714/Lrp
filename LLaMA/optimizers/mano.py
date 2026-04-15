import torch
import torch.nn.functional as F
import math

class new_optimizer(torch.optim.Optimizer):

    def __init__(self, param_groups, lr_rmnp=0.005, lr_adam=0.001, r=1.833, momentum=0.95, beta=0.95, 
                 weight_decay=0.0, betas=(0.9, 0.95), eps=1e-10):
        # 保持接口参数名不变
        defaults = dict(lr_rmnp=lr_rmnp, lr_adam=lr_adam, r=r, momentum=momentum, beta=beta, 
                       weight_decay=weight_decay, betas=betas, eps=eps)
        super(new_optimizer, self).__init__(param_groups, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']  
            momentum = group.get('momentum', 0.95)
            weight_decay = group.get('weight_decay', 0.0)
            betas = group.get('betas', (0.9, 0.95))
            eps = group.get('eps', 1e-10)
            is_rmnp = group.get('is_rmnp', True)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                param_state = self.state.setdefault(p, {})
                
                # 全局维护 step，因为 Mano 需要用到 t 来做交替 (t mod 2)
                if 'step' not in param_state:
                    param_state['step'] = 0
                param_state['step'] += 1
                t = param_state['step']
                
                if is_rmnp and grad.dim() >= 2:
                    # ==========================================
                    # --- Mano Optimizer 核心逻辑 ---
                    # ==========================================
                    
                    if 'momentum_buffer' not in param_state:
                        buf = torch.zeros_like(grad)
                    else:
                        buf = param_state['momentum_buffer']
                    
                    # 1. 动量更新: M_t = \mu M_{t-1} + g_t (遵循 Mano 论文公式)
                    buf.mul_(momentum).add_(grad)
                    M = buf 
                    
                    # 2. 旋转流形机制: 交替维度 k = t mod 2
                    # k=0 对应行方向(dim=-1), k=1 对应列方向(dim=-2)
                    k = t % 2
                    dim_k = -1 if k == 0 else -2
                    n_k = p.data.size(dim_k)
                    
                    # 3. 强制参数归一化: \hat{\theta}_t = \theta_t / ||\theta_t||_{2,k}
                    theta_hat = F.normalize(p.data, p=2, dim=dim_k, eps=eps)
                    
                    # 4. 计算切空间动量: v_t = M_t - \hat{\theta}_t * <M_t, \hat{\theta}_t>_k
                    dot_product = torch.sum(M * theta_hat, dim=dim_k, keepdim=True)
                    v = M - dot_product * theta_hat
                    
                    # 5. 动量流形归一化: \hat{v}_t = v_t / ||v_t||_{2,k}
                    v_hat = F.normalize(v, p=2, dim=dim_k, eps=eps)
                    
                    # 6. 尺度放缩: 0.2 * sqrt(n_k) (对齐 AdamW 的更新幅度)
                    scale = 0.2 * math.sqrt(n_k)
                    update_direction = v_hat * scale
                    
                    # 7. 应用 Weight Decay 和梯度更新
                    if weight_decay > 0:
                        p.data.mul_(1 - lr * weight_decay)
                    
                    p.data.add_(update_direction, alpha=-lr)
                    
                    # 更新状态缓存
                    param_state['momentum_buffer'] = buf
                    
                else:
                    # ==========================================
                    # --- AdamW 逻辑 (保持完全不变) ---
                    # ==========================================
                    if 'exp_avg' not in param_state:
                        param_state['exp_avg'] = torch.zeros_like(grad)
                        param_state['exp_avg_sq'] = torch.zeros_like(grad)
                    
                    exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']
                    
                    exp_avg.mul_(betas[0]).add_(grad, alpha=1-betas[0])
                    exp_avg_sq.mul_(betas[1]).addcmul_(grad, grad, value=1-betas[1])
                    
                    bias_correction1 = 1 - betas[0] ** t
                    bias_correction2 = 1 - betas[1] ** t
                    step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                    
                    denom = exp_avg_sq.sqrt().add_(eps)
                    adam_update = exp_avg / denom
                    
                    if weight_decay > 0:
                        p.data.mul_(1 - step_size * weight_decay)
                    
                    p.data.add_(adam_update, alpha=-step_size)
                    
        return loss

def get_new_optimizer(model, lr_rmnp=0.005, lr_adam=0.001, r=1.833, weight_decay=0.1, momentum=0.95, beta=0.95):
    rmnp_params = []
    adam_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.ndim >= 2 and 'embed' not in name and 'lm_head' not in name:
                rmnp_params.append(param)
            else:
                adam_params.append(param)
    
    param_groups = [
        dict(params=rmnp_params, lr=lr_rmnp, lr_rmnp=lr_rmnp, lr_adam=lr_adam, r=r,
             weight_decay=weight_decay, momentum=momentum, beta=beta, is_rmnp=True),
        dict(params=adam_params, lr=lr_adam, lr_rmnp=lr_rmnp, lr_adam=lr_adam, r=r,
             weight_decay=weight_decay, momentum=momentum, beta=beta, is_rmnp=False)
    ]
    optimizer = new_optimizer(param_groups)
    return optimizer