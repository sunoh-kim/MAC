import torch
from utils import one_hot_embedding
# from models.model import *
import torch.nn.functional as F
import torch.nn as nn
import functools
from autoattack import AutoAttack
from func import clip_img_preprocessing, multiGPU_CLIP, multiGPU_CLIP_image_logits

lower_limit, upper_limit = 0, 1
def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

import torch.optim as optim
import torch.nn.functional as F
import time
from torch import amp


def tanh_space(w):
    # map R -> (0,1)
    return 0.5 * (torch.tanh(w) + 1.0)

def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))

def inverse_tanh_space_(x01):
    # x in [0,1] -> R
    x = torch.clamp(x01 * 2.0 - 1.0, min=-1.0, max=1.0)
    return atanh(x)

def cw_margin(outputs, labels, targeted=False, kappa=0.0):
    # outputs: [N, C] logits, labels: [N]
    N, C = outputs.shape
    one_hot = F.one_hot(labels, num_classes=C).float()
    real = torch.sum(one_hot * outputs, dim=1)               # Z(x')_y
    other = torch.max((1 - one_hot) * outputs - 1e4 * one_hot, dim=1).values  # max_{i!=y} Z(x')_i

    if targeted:
        return torch.clamp(other - real, min=-kappa)
    else:
        return torch.clamp(real - other, min=-kappa)



def cw_f(outputs, labels, targeted=False, kappa=0.0):
    num_classes = outputs.shape[1]
    one_hot = torch.eye(num_classes, device=outputs.device)[labels]  # (N,C)

    other = torch.max((1 - one_hot) * outputs, dim=1).values
    real = torch.max(one_hot * outputs, dim=1).values

    if targeted:
        return torch.clamp(other - real, min=-kappa)
    else:
        return torch.clamp(real - other, min=-kappa)




def inverse_tanh_space(x, eps=1e-6):
    # x ∈ [0,1] -> R
    x = torch.clamp(x, eps, 1 - eps)
    z = x * 2 - 1
    return 0.5 * torch.log((1 + z) / (1 - z))

def cw_f(logits, labels, targeted=False, kappa=0.0):
    # logits: (B, C), labels: (B,)
    one_hot = torch.nn.functional.one_hot(labels, num_classes=logits.size(1)).float()
    real = (one_hot * logits).sum(dim=1)
    other = (logits - 1e10 * one_hot).max(dim=1).values
    if targeted:
        # make target more likely than others by margin kappa
        return torch.clamp(other - real, min=-kappa)
    else:
        # make real less likely than best-other by margin kappa
        return torch.clamp(real - other, min=-kappa)

def attack_CW(
    args, prompter, model, model_text, model_image, add_prompter, criterion, 
    X, target, text_tokens,
    alpha, attack_iters, norm,  
    restarts=1, early_stop=True, epsilon=0,
    c=1.0, kappa=0.0, targeted=False,
    dataset_name=None,
):

    device = X.device
    X = X.detach()  

    w = inverse_tanh_space(X).to(device).detach()
    w.requires_grad_(True)

    mse = nn.MSELoss(reduction="none")
    flatten = nn.Flatten()
    optimizer = optim.Adam([w], lr=alpha)

    best_adv = X.clone().detach()
    best_L2 = torch.full((X.size(0),), 1e10, device=device)
    prev_cost = 1e10
    dim = len(X.shape)

    tunable_param_names = []
    if hasattr(model, "module"):
        named_params = model.module.named_parameters()
    else:
        named_params = model.named_parameters()
    for n, p in named_params:
        if p.requires_grad:
            tunable_param_names.append(n)
            p.requires_grad = False

    try:
        for step in range(attack_iters):
            adv = tanh_space(w)  # [0,1]

            _images = clip_img_preprocessing(adv)
            prompted_images = prompter(_images)
            prompt_token = add_prompter()
            logits, _, _, _ = multiGPU_CLIP(
                args, model_image, model_text, model,
                prompted_images,
                text_tokens=text_tokens,
                prompt_token=prompt_token,
                dataset_name=dataset_name
            )

            current_L2 = mse(flatten(adv), flatten(X)).sum(dim=1)  # per-sample
            L2_loss = current_L2.sum()
            f_loss = cw_f(logits, target, targeted=targeted, kappa=kappa).sum()
            cost = L2_loss + c * f_loss

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            pred = logits.detach().argmax(dim=1)
            if targeted:
                cond = (pred == target).float()
            else:
                cond = (pred != target).float()

            mask = cond * (best_L2 > current_L2.detach())
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2
            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv = mask * adv.detach() + (1 - mask) * best_adv

            if early_stop and (step % max(attack_iters // 10, 1) == 0):
                if cost.item() > prev_cost:
                    return (best_adv - X).detach()
                prev_cost = cost.item()

    finally:
        if hasattr(model, "module"):
            named_params = model.module.named_parameters()
        else:
            named_params = model.named_parameters()
        for n, p in named_params:
            if n in tunable_param_names:
                p.requires_grad = True

    return (best_adv - X).detach()


def attack_CW_noprompt(args, prompter, model, model_text, model_image, criterion, X, target, text_tokens, alpha,
                       attack_iters, norm, restarts=1, early_stop=True, epsilon=0):
    delta = torch.zeros_like(X).cuda()
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    for _ in range(attack_iters):
        # output = model(normalize(X ))

        _images = clip_img_preprocessing(X + delta)
        # output, _ = model(_images, text_tokens)

        output, _, _, _ = multiGPU_CLIP(args, model_image, model_text, model, _images, text_tokens, None)

        num_class = output.size(1)
        label_mask = one_hot_embedding(target, num_class)
        label_mask = label_mask.cuda()

        correct_logit = torch.sum(label_mask * output, dim=1)
        wrong_logit, _ = torch.max((1 - label_mask) * output - 1e4 * label_mask, axis=1)

        # loss = criterion(output, target)
        loss = - torch.sum(F.relu(correct_logit - wrong_logit + 50))

        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()

    return delta

def attack_unlabelled(model, X, prompter, add_prompter, alpha, attack_iters, norm="l_inf", epsilon=0,
                      visual_model_orig=None):
    delta = torch.zeros_like(X)
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError

    # turn off model parameters temmporarily
    tunable_param_names = []
    for n,p in model.module.named_parameters():
        if p.requires_grad: 
            tunable_param_names.append(n)
            p.requires_grad = False

    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True

    if attack_iters <= 0: 
        return delta

    prompt_token = add_prompter()
    with torch.no_grad():
        if visual_model_orig is None: # use the model itself as anchor
            X_ori_reps = model.module.encode_image(
                prompter(clip_img_preprocessing(X)), prompt_token
            )
        else: # use original frozen model as anchor
            X_ori_reps = visual_model_orig.module(
                prompter(clip_img_preprocessing(X)), prompt_token
            )

    for _ in range(attack_iters):

        prompted_images = prompter(clip_img_preprocessing(X + delta))

        X_att_reps = model.module.encode_image(prompted_images, prompt_token)
        # l2_loss = ((((X_att_reps - X_ori_reps)**2).sum(1))**(0.5)).sum()
        l2_loss = ((((X_att_reps - X_ori_reps)**2).sum(1))).sum()

        grad = torch.autograd.grad(l2_loss, delta)[0]
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d

    # # Turn on model parameters
    for n,p in model.module.named_parameters():
        if n in tunable_param_names:
            p.requires_grad = True

    return delta

#### opposite update direction of attack_unlabelled()
def attack_unlabelled_opp(model, X, prompter, add_prompter, alpha, attack_iters, norm="l_inf", epsilon=0,
                      visual_model_orig=None):
    delta = torch.zeros_like(X)
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError

    # turn off model parameters temmporarily
    tunable_param_names = []
    for n,p in model.module.named_parameters():
        if p.requires_grad: 
            tunable_param_names.append(n)
            p.requires_grad = False

    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True

    prompt_token = add_prompter()
    with torch.no_grad():
        if visual_model_orig is None: # use the model itself as anchor
            X_ori_reps = model.module.encode_image(
                prompter(clip_img_preprocessing(X)), prompt_token
            )
        else: # use original frozen model as anchor
            X_ori_reps = visual_model_orig.module(
                prompter(clip_img_preprocessing(X)), prompt_token
            )

    for _ in range(attack_iters):

        prompted_images = prompter(clip_img_preprocessing(X + delta))

        X_att_reps = model.module.encode_image(prompted_images, prompt_token)
        # l2_loss = ((((X_att_reps - X_ori_reps)**2).sum(1))**(0.5)).sum()
        l2_loss = ((((X_att_reps - X_ori_reps)**2).sum(1))).sum()

        grad = torch.autograd.grad(l2_loss, delta)[0]
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            # d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            d = torch.clamp(d - alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            # d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
            d = (d - scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d

    # # Turn on model parameters
    for n,p in model.module.named_parameters():
        if n in tunable_param_names:
            p.requires_grad = True

    return delta

def attack_unlabelled_cosine(model, X, prompter, add_prompter, alpha, attack_iters, norm="l_inf", epsilon=0,
                      visual_model_orig=None):
    # unlabelled attack to maximise cosine similarity between the attacked image
    # and the original image, computed by PGD
    delta = torch.zeros_like(X)
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError

    # turn off model parameters temmporarily
    tunable_param_names = []
    for n,p in model.module.named_parameters():
        if p.requires_grad: 
            tunable_param_names.append(n)
            p.requires_grad = False

    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True

    prompt_token = add_prompter()
    with torch.no_grad():
        if visual_model_orig is None: # use the model itself as anchor
            X_ori_reps = model.module.encode_image(
                prompter(clip_img_preprocessing(X)), prompt_token
            )
        else: # use original frozen model as anchor
            X_ori_reps = visual_model_orig.module(
                prompter(clip_img_preprocessing(X)), prompt_token
            )
        # X_ori_reps_norm = X_ori_reps / X_ori_reps.norm(dim=-1, keepdim=True)

    for _ in range(attack_iters):

        prompted_images = prompter(clip_img_preprocessing(X + delta))

        X_att_reps = model.module.encode_image(prompted_images, prompt_token) # [bs, d_out]
        # X_att_reps_norm = X_att_reps / X_att_reps.norm(dim=-1, keepdim=True)
        
        cos_loss = 1 - F.cosine_similarity(X_att_reps, X_ori_reps) # [bs]

        grad = torch.autograd.grad(cos_loss, delta)[0]
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d

    # # Turn on model parameters
    for n,p in model.module.named_parameters():
        if n in tunable_param_names:
            p.requires_grad = True

    return delta


def attack_pgd(args, prompter, model, model_text, model_image, add_prompter, criterion, X, target, alpha,
               attack_iters, norm, text_tokens=None, restarts=1, early_stop=True, epsilon=0, dataset_name=None):
    delta = torch.zeros_like(X).cuda()
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True

    # turn off model parameters temmporarily
    tunable_param_names = []
    for n,p in model.module.named_parameters():
        if p.requires_grad: 
            tunable_param_names.append(n)
            p.requires_grad = False

    for iter in range(attack_iters):

        prompted_images = prompter(clip_img_preprocessing(X + delta))
        prompt_token = add_prompter()

        output, _, _, _ = multiGPU_CLIP(args, model_image, model_text, model, prompted_images, 
                                  text_tokens=text_tokens, prompt_token=prompt_token, dataset_name=dataset_name)

        loss = criterion(output, target)

        grad = torch.autograd.grad(loss, delta)[0]

        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        # delta.grad.zero_()

    # # Turn on model parameters
    for n,p in model.module.named_parameters():
        if n in tunable_param_names:
            p.requires_grad = True

    return delta


def attack_pgd_noprompt(args, prompter, model, model_text, model_image, criterion, X, target, text_tokens, alpha,
                        attack_iters, norm, restarts=1, early_stop=True, epsilon=0):
    delta = torch.zeros_like(X).cuda()
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    for _ in range(attack_iters):

        _images = clip_img_preprocessing(X + delta)
        output, _, _, _ = multiGPU_CLIP(args, model_image, model_text, model, _images, text_tokens, None)

        loss = criterion(output, target)

        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()

    return delta

def attack_auto(model, images, target, text_tokens, prompter, add_prompter,
                         attacks_to_run=['apgd-ce', 'apgd-dlr'], epsilon=0):

    forward_pass = functools.partial(
        multiGPU_CLIP_image_logits,
        model=model, text_tokens=text_tokens,
        prompter=None, add_prompter=None
    )

    adversary = AutoAttack(forward_pass, norm='Linf', eps=epsilon, version='standard', verbose=False)
    adversary.attacks_to_run = attacks_to_run
    x_adv = adversary.run_standard_evaluation(images, target, bs=images.shape[0])
    return x_adv

def attack_difgsm(
    args, prompter, model, model_text, model_image, add_prompter, criterion,
    X, target, text_tokens, alpha, attack_iters, norm,
    epsilon=0, resize_rate=0.9, diversity_prob=0.5, decay=0.0, random_start=False,
    dataset_name=None,
):

    device = X.device
    delta = torch.zeros_like(X, device=device)

    if random_start:
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / (n + 1e-12) * epsilon
        else:
            raise ValueError(f"Unsupported norm: {norm}")

    delta = clamp(delta, lower_limit - X, upper_limit - X)
    momentum = torch.zeros_like(X, device=device)

    tunable_param_names = []
    for n, p in model.module.named_parameters():
        if p.requires_grad:
            tunable_param_names.append(n)
            p.requires_grad = False

    def input_diversity(imgs):
        B, C, H, W = imgs.shape
        img_size = W
        img_resize = int(img_size * resize_rate)

        if resize_rate < 1.0:
            img_size, img_resize = img_resize, W

        if img_resize <= img_size:
            return imgs if torch.rand(1, device=device) < diversity_prob else imgs

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), device=device).item()
        if rnd <= 0:
            return imgs

        rescaled = F.interpolate(imgs, size=(rnd, rnd), mode="bilinear", align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd

        if h_rem > 0:
            pad_top = torch.randint(low=0, high=h_rem, size=(1,), device=device).item()
        else:
            pad_top = 0
        if w_rem > 0:
            pad_left = torch.randint(low=0, high=w_rem, size=(1,), device=device).item()
        else:
            pad_left = 0

        pad_bottom = h_rem - pad_top
        pad_right  = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left, pad_right, pad_top, pad_bottom], value=0.0)
        return padded if torch.rand(1, device=device) < diversity_prob else imgs

    for _ in range(attack_iters):
        delta.requires_grad_(True)

        _images = clip_img_preprocessing(X + delta)
        prompted_images = prompter(_images)
        prompt_token = add_prompter()

        diversified = input_diversity(prompted_images)

        logits, _, _, _ = multiGPU_CLIP(
            args, model_image, model_text, model,
            diversified, text_tokens=text_tokens, prompt_token=prompt_token,
            dataset_name=dataset_name
        )

        loss = criterion(logits, target)
        grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]

        grad = grad / (torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-12)
        grad = grad + decay * momentum
        momentum = grad

        d = delta
        if norm == "l_inf":
            d = d + alpha * torch.sign(grad)
            d = torch.clamp(d, min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = grad / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        else:
            raise ValueError(f"Unsupported norm: {norm}")

        d = clamp(d, lower_limit - X, upper_limit - X)
        delta = d.detach()

    for n, p in model.module.named_parameters():
        if n in tunable_param_names:
            p.requires_grad = True

    return delta

