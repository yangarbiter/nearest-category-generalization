"""
https://raw.githubusercontent.com/yaodongyu/TRADES/master/trades.py
"""
import ipdb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss

from joblib import Parallel, delayed, parallel_backend
from .trades import project_delta, project_x_ball
from .utils import project_epsilon


def scale_gradient(grad, epsilon):
    """"""
    #for i in range(len(grad)):
    #    grad[i] = epsilon[i].scale_fn(grad[i])
        #grad[i] = torch.dot(grad[i].T, epsilon[i].invBB)
    return grad

def project_epsilon(epsilon, max_eps, x_natural=None):
    """
    max_eps: [batch_size, 1]
    """
    if isinstance(max_eps, float):
        epsilon = torch.clamp(epsilon, 0, max_eps)
    elif torch.is_tensor(max_eps):
        device = max_eps.get_device()
        epsilon = torch.min(epsilon, max_eps)
        epsilon = torch.max(epsilon, torch.zeros_like(epsilon).to(device))
    elif callable(epsilon[0]):
        def _helper(x, fn):
            return fn(x.reshape(len(x), -1)).reshape(-1)
        x_adv = x_natural + epsilon
        device = x_adv.get_device()
        x_adv = x_adv.cpu().numpy()
        with parallel_backend("loky", inner_max_num_threads=4):
            ret = Parallel(n_jobs=8)(delayed(_helper)(x, epsilon[i]) for i, x in enumerate(x_adv))
        ret = torch.from_numpy(np.array(ret)).float().to(device)
        return ret - x_natural
    else:
        raise ValueError
    return epsilon

def cusgradtrades_ellipsoid_loss(
        loss_name, model, loss_fn, x_natural, y, norm, optimizer, current_eps, eta,
        threshold, clip_min=None, clip_max=None, step_size=0.003, epsilon=0.031,
        perturb_steps=10, beta=1.0, version=None, n_jobs=4):
    device = x_natural.get_device()

    if version not in [None]:
        raise ValueError(f"[TRADES] not supported version {version}")

    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='none')

    current_eps = current_eps + eta
    if callable(epsilon[0]):
        pass
    else:
        current_eps = project_epsilon(current_eps, epsilon.view(-1, 1))

    if torch.is_tensor(step_size):
        shape = [-1] + [1]*(len(x_natural.shape)-1)
        step_size = step_size.view(*shape)
    if torch.is_tensor(epsilon):
        shape = [-1] + [1]*(len(x_natural.shape)-1)
        epsilon = epsilon.view(*shape)
    if torch.is_tensor(current_eps):
        shape = [-1] + [1]*(len(x_natural.shape)-1)
        current_eps = current_eps.view(*shape)

    batch_size = len(x_natural)

    if 'trn' in loss_name:
        model.train()
    else:
        model.eval()

    clean_output = model(x_natural).detach()
    # generate adversarial example
    if norm in [np.inf]:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(clean_output, dim=1)).sum(dim=1)
                loss_kl = torch.mean(loss_kl)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            grad = scale_gradient(grad, epsilon)
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = project_x_ball(x_adv, x_natural, current_eps, n_jobs=n_jobs, norm=norm)

            if clip_min is not None and clip_max is not None:
                x_adv = torch.clamp(x_adv, clip_min, clip_max)
    elif norm == 2:
        delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()

        for _ in range(perturb_steps):
            delta.requires_grad_(True)
            x_adv = x_natural + delta

            # optimize
            #optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss_kl = (-1) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                           F.softmax(clean_output, dim=1)).sum(dim=1)
                loss_kl = torch.mean(loss_kl)

            dgrad = torch.autograd.grad(loss_kl, [delta])[0]
            grad_norms = dgrad.view(batch_size, -1).norm(p=2, dim=1)
            while len(grad_norms.shape) < len(dgrad.shape):
                grad_norms = grad_norms.unsqueeze(-1)
            dgrad.div_(grad_norms)
            if (grad_norms == 0).any():
                #dgrad[grad_norms == 0] = torch.randn_like(dgrad[grad_norms == 0])
                dgrad[grad_norms.flatten() == 0] = torch.randn_like(dgrad[grad_norms.flatten() == 0])
            dgrad = scale_gradient(dgrad, epsilon)
            delta = delta - step_size * dgrad

            # projection
            if clip_min is not None and clip_max is not None:
                delta.data.add_(x_natural)
                delta.data.clamp_(clip_min, clip_max).sub_(x_natural)
            delta = project_delta(delta, epsilon, norm=2, x_natural=x_natural)
    else:
        raise ValueError(f"Not supported Norm {norm}")

    if clip_min is not None and clip_max is not None:
        x_adv = torch.clamp(x_adv, clip_min, clip_max)
    x_adv = Variable(x_adv, requires_grad=False)

    model.train()
    optimizer.zero_grad()

    outputs = model(x_natural)
    loss_natural = loss_fn(outputs, y)

    loss_robust = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                               F.softmax(model(x_natural), dim=1)).sum(dim=1)
    loss = loss_natural + beta * loss_robust

    idx = (loss_robust > threshold).clone().detach()
    current_eps = current_eps.clone().detach()
    current_eps.requires_grad_(False)
    if 'twotimes' in loss_name:
        if torch.is_tensor(eta):
            current_eps[idx] = (current_eps[idx] - 2*eta[idx])
        else:
            current_eps[idx] = (current_eps[idx] - 2*eta)
    else:
        if torch.is_tensor(eta):
            current_eps[idx] = (current_eps[idx] - eta[idx])
        else:
            current_eps[idx] = (current_eps[idx] - eta)

    if callable(epsilon[0]):
        current_eps = current_eps.view(len(current_eps), 1)
    else:
        current_eps = project_epsilon(current_eps, epsilon).view(len(current_eps), 1)

    return outputs, loss, current_eps
