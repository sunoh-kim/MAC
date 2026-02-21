from __future__ import print_function

import argparse
import os
import time
import random
import logging
from tqdm import tqdm
from copy import deepcopy as dcopy
from PIL import Image
import numpy as np

import torch
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms
try:
    from torchvision.transforms import InterpolationMode
    INTERP_BILINEAR = InterpolationMode.BILINEAR
except Exception:
    INTERP_BILINEAR = Image.BILINEAR
    
from replace import clip
from models.prompters import TokenPrompter, NullPrompter
from utils import *
from attacks import *
from func import clip_img_preprocessing, multiGPU_CLIP

def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--evaluate', type=bool, default=True) # eval mode
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')  # 32
    parser.add_argument('--cache', type=str, default='./cache')    

    # test setting
    parser.add_argument('--test_set', default=[], type=str, nargs='*') # defaults to 17 datasets, if not specified
    parser.add_argument('--test_attack_type', type=str, default="pgd", choices=['pgd', 'cw', 'aa','di'])
    parser.add_argument('--test_eps', type=float, default=4,help='test attack budget')
    parser.add_argument('--test_numsteps', type=int, default=100)
    parser.add_argument('--test_stepsize', type=float, default=1)

    # model
    parser.add_argument('--model', type=str, default='clip')
    parser.add_argument('--arch', type=str, default='ViT-B/32')
    parser.add_argument('--method', type=str, default='null_patch',
                        choices=['null_patch'], help='choose visual prompting method')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--prompt_size', type=int, default=30, help='size for visual prompts')
    parser.add_argument('--add_prompt_size', type=int, default=0, help='size for additional visual prompts')

    # data
    parser.add_argument('--root', type=str, default='../data', help='dataset path')
    parser.add_argument('--dataset', type=str, default='tinyImageNet', help='dataset used for AFT methods')
    parser.add_argument('--image_size', type=int, default=224, help='image size')
    
    # MAC config
    parser.add_argument('--seed', type=int, default=0, help='seed for initializing training')
    parser.add_argument('--victim_resume', type=str, default=None, help='model weights of victim to attack.')
    parser.add_argument('--outdir', type=str, default=None, help='output directory for results')
    parser.add_argument('--tau_thres', type=float, default=0.7)
    parser.add_argument('--mac_eps', type=float, default=8.)
    parser.add_argument('--mac_numsteps', type=int, default=4)
    
    parser.add_argument('--tau_temp', type=float, default=0.01)
    parser.add_argument('--num_views', type=int, default=2,
                    help='Number of views per image, including the original')
    
    args = parser.parse_args()
    return args

class AdditiveNoise(torch.nn.Module):
    def __init__(self, sigma_min=0.0, sigma_max=0.02):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    def forward(self, x):
        sigma = torch.empty(1, device=x.device).uniform_(self.sigma_min, self.sigma_max)
        noise = torch.randn_like(x) * sigma
        return (x + noise).clamp(0, 1)
    
def build_aug(
    rotation=5.0,
    translate=0.04,
    scale_var=0.08, 
    brightness=0.08,
    contrast=0.08,
    saturation=0.08,
    hue=0.02,
    blur_sigma=(0.5, 1.2),
    noise_sigma=0.02,
):
    affine = transforms.RandomAffine(
        degrees=rotation,
        translate=(translate, translate),
        scale=(1.0 - scale_var, 1.0 + scale_var),
        shear=0.0,
        interpolation=INTERP_BILINEAR   
    )

    color_jitter = transforms.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue
    )

    aug = transforms.Compose([
        transforms.RandomApply([affine], p=1.0), 
        transforms.RandomApply([color_jitter], p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=blur_sigma)], p=0.5),
        transforms.RandomApply([AdditiveNoise(0.0, noise_sigma)], p=0.5),
    ])
    return aug

def build_views(images, num_views, aug):
    if num_views <= 1 or aug is None:
        return images
    views = [images]
    with torch.no_grad():
        for _ in range(num_views - 1):
            imgs_aug = torch.stack([aug(img) for img in images], dim=0)
            views.append(imgs_aug)
    return torch.cat(views, dim=0)

def aggregate_views(logits, num_views):
    if num_views <= 1:
        return logits
    V = num_views
    Bv, C = logits.shape
    assert Bv % V == 0, f"Logits batch {Bv} not divisible by num_views {V}"
    N = Bv // V
    return logits.view(V, N, C).mean(dim=0)

def multiview_guided_counterattack(model, X, prompter, add_prompter, alpha, attack_iters, 
                           norm="l_inf", epsilon=0, visual_model_orig=None,
                           tau_thres:float=None, tau_temp:float=None, clip_visual=None, aug=None):
    delta = torch.zeros_like(X)
    if epsilon <= 0.:
        return delta

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

    if attack_iters == 0:
        return delta.data

    # Freeze model parameters
    tunable_param_names = []
    for n,p in model.module.named_parameters():
        if p.requires_grad: 
            tunable_param_names.append(n)
            p.requires_grad = False

    prompt_token = add_prompter()
    with torch.no_grad():
        X_ori_reps = model.module.encode_image(
                prompter(clip_img_preprocessing(X)), prompt_token
        )
        X_ori_norm = torch.norm(X_ori_reps, dim=-1)

    if aug is None:
        aug = build_aug() 
        
    with torch.no_grad():
        X_cpu = X.detach().cpu()
        X_aug_list = [aug(img) for img in X_cpu]
        X_aug = torch.stack(X_aug_list, dim=0).to(X.device)

        X_aug_reps = model.module.encode_image(
            prompter(clip_img_preprocessing(X_aug)), prompt_token
        )
        corpt_degree = (X_aug_reps - X_ori_reps).norm(dim=-1) / (X_ori_reps.norm(dim=-1) + 1e-10)

    soft_weight = torch.sigmoid((corpt_degree - tau_thres) / tau_temp).view(-1, 1, 1, 1)

    for _step_id in range(attack_iters):

        prompted_images = prompter(clip_img_preprocessing(X + delta))
        X_att_reps = model.module.encode_image(prompted_images, prompt_token)
        
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

    delta = delta * soft_weight
    
    # Unfreeze model parameters. Only for completeness of code
    for n,p in model.module.named_parameters():
        if n in tunable_param_names:
            p.requires_grad = True

    return delta.detach()

def validate(args, val_dataset_name, model, model_text, model_image,
             prompter, add_prompter, criterion, visual_model_orig=None,
             clip_visual=None, aug=None
    ):
    
    logging.info(f"Evaluate with Attack method: {args.test_attack_type}")

    dataset_num = len(val_dataset_name)
    all_clean_mac, all_adv_mac = {},{}

    test_stepsize = args.test_stepsize

    mac_eps = args.mac_eps
    mac_numsteps = args.mac_numsteps
    mac_stepsize = args.mac_stepsize
    tau_thres = args.tau_thres
    tau_temp = args.tau_temp

    for cnt in range(dataset_num):
        dataset_start_time = time.time()
        
        val_dataset, val_loader = load_val_dataset(args, val_dataset_name[cnt])
        dataset_name = val_dataset_name[cnt]
        texts = get_text_prompts_val([val_dataset], [dataset_name])[0]

        binary = ['PCAM', 'hateful_memes']
        attacks_to_run=['apgd-ce', 'apgd-dlr']
        if dataset_name in binary:
            attacks_to_run=['apgd-ce']

        batch_time = AverageMeter('Time', ':6.3f')

        losses = AverageMeter('Loss', ':.4e')
        top1_org_mac = AverageMeter('MAC Acc@1', ':6.2f')
        top1_adv_mac = AverageMeter('Adv MAC Acc@1', ':6.2f')

        # switch to evaluation mode
        prompter.eval()
        add_prompter.eval()
        model.eval()

        text_tokens = clip.tokenize(texts).to(device)
        end = time.time()
        num_samples = len(val_dataset)


        for i, (images, target) in enumerate(tqdm(val_loader)):
  
            V = args.num_views
            
            with autocast():

                if V > 1:
                    images_views = build_views(images, V, aug).to(device)
                    images = images.to(device)
                else:
                    images_views = images.to(device)
                    images = images.to(device)
                target = target.to(device)

                # MAC on clean images
                mac_delta_clean = multiview_guided_counterattack(
                    model, images_views, prompter, add_prompter,
                    alpha=mac_stepsize, attack_iters=mac_numsteps,
                    norm='l_inf', epsilon=mac_eps, visual_model_orig=None,
                    tau_thres=tau_thres, tau_temp=tau_temp,
                    clip_visual=clip_visual, aug=aug
                )
                with torch.no_grad():
                    clean_out_mac_views,_,_,_ = multiGPU_CLIP(
                        None, None, None, model, prompter(clip_img_preprocessing(images_views+mac_delta_clean)),
                        text_tokens = text_tokens,
                        prompt_token = None, dataset_name = dataset_name
                    )
                    clean_out_mac = aggregate_views(clean_out_mac_views, V)
                    clean_acc_mac = accuracy(clean_out_mac, target, topk=(1,))
                    top1_org_mac.update(clean_acc_mac[0].item(), images.size(0))

                # generate adv samples for this batch
                torch.cuda.empty_cache()
                if args.test_attack_type == "pgd":
                    delta_prompt = attack_pgd(args, prompter, model, model_text, model_image, add_prompter, criterion,
                                              images, target, test_stepsize, args.test_numsteps, 'l_inf',
                                              text_tokens=text_tokens, epsilon=args.test_eps, dataset_name=dataset_name)
                    attacked_images = images + delta_prompt
                elif args.test_attack_type == "cw":
                    delta_prompt = attack_CW(
                        args, prompter, model, model_text, model_image, add_prompter, criterion,
                        images, target, text_tokens,
                        alpha=0.01, attack_iters=500, norm='l_2',
                        c=3.0, kappa=0.0, targeted=False,
                        early_stop=True
                    )
                    attacked_images = images + delta_prompt
                elif args.test_attack_type == "aa":
                    attacked_images = attack_auto(model, images, target, text_tokens,
                        None, None, epsilon=args.test_eps, attacks_to_run=attacks_to_run)
                elif args.test_attack_type == "di":
                    delta_prompt = attack_difgsm(
                        args, prompter, model, model_text, model_image, add_prompter, criterion,
                        images, target, text_tokens, alpha=test_stepsize, attack_iters=args.test_numsteps, norm='l_inf',
                        epsilon=args.test_eps, resize_rate=0.9, diversity_prob=0.5, decay=0.0, random_start=True,
                        dataset_name=dataset_name
                    )
                    attacked_images = images + delta_prompt
                    
                if V > 1:
                    attacked_images_views = build_views(attacked_images, V, aug).to(device)
                else:
                    attacked_images_views = attacked_images.to(device)
                    
                mac_delta_adv = multiview_guided_counterattack(
                    model, attacked_images_views.data, prompter, add_prompter,
                    alpha=mac_stepsize, attack_iters=mac_numsteps,
                    norm='l_inf', epsilon=mac_eps, visual_model_orig=None,
                    tau_thres=tau_thres, tau_temp=tau_temp,
                    clip_visual=clip_visual, aug=aug
                )
                with torch.no_grad():
                    adv_out_mac_views,_,_,_ = multiGPU_CLIP(
                        None,None,None, model, prompter(clip_img_preprocessing(attacked_images_views+mac_delta_adv)),
                        text_tokens, prompt_token=None, dataset_name=dataset_name 
                    )
                    adv_output_mac = aggregate_views(adv_out_mac_views, V)
                    adv_output_acc = accuracy(adv_output_mac, target, topk=(1,))
                    top1_adv_mac.update(adv_output_acc[0].item(), images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
        
        torch.cuda.empty_cache()
        
        dataset_elapsed = time.time() - dataset_start_time
        avg_per_sample = dataset_elapsed / num_samples

        show_text = f"\n\nDataset: {dataset_name}:"
        show_text += f"\n{dataset_name} processed in {dataset_elapsed:.2f} sec "
        show_text += f"\n({avg_per_sample:.7f} s/sample for {num_samples} samples)"
        print(show_text)
        logging.info(show_text)      
          
        clean_mac_acc = top1_org_mac.avg
        adv_mac_acc = top1_adv_mac.avg

        all_clean_mac[dataset_name] = clean_mac_acc
        all_adv_mac[dataset_name] = adv_mac_acc

        show_text = f"\t- clean acc. {clean_mac_acc:.2f}\n"
        show_text += f"\t- robust acc. {adv_mac_acc:.2f}\n"
        
        print(show_text)
        logging.info(show_text)

    all_clean_mac_avg = np.mean([all_clean_mac[name] for name in all_clean_mac]).item()
    all_adv_mac_avg = np.mean([all_adv_mac[name] for name in all_adv_mac]).item()
    show_text = f"===== SUMMARY ACROSS {dataset_num} DATASETS =====\n\t"
    show_text += f"AVG acc. {all_clean_mac_avg:.2f}\n\t"
    show_text += f"AVG acc. {all_adv_mac_avg:.2f}"
    logging.info(show_text)

    return all_clean_mac_avg, all_adv_mac_avg

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():

    args = parse_options()

    outdir = args.outdir if args.outdir is not None else "MAC_results"
    outdir = os.path.join(outdir, f"{args.test_attack_type}_eps_{args.test_eps}_numsteps_{args.test_numsteps}")
    os.makedirs(outdir, exist_ok=True)

    args.test_eps = args.test_eps / 255.
    args.test_stepsize = args.test_stepsize / 255.

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    log_filename = ""
    log_filename += "log.log"
    log_filename = os.path.join(outdir, log_filename)
    logging.basicConfig(
        filename = log_filename,
        level = logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info(args)

    args.mac_eps = args.mac_eps / 255.
    args.mac_stepsize = args.mac_eps / 4.

    imagenet_root = './data/ImageNet'
    args.imagenet_root = imagenet_root

    # load model
    model, _ = clip.load(args.arch, device, jit=False, prompt_len=0)  #'ViT-B/32'
    for p in model.parameters():
        p.requires_grad = False
    convert_models_to_fp32(model)

    if args.victim_resume: # employ MAC on AFT checkpoints
        clip_visual = dcopy(model.visual)
        model = load_checkpoints2(args, args.victim_resume, model, None)
    else:                  # employ MAC on the original CLIP
        clip_visual = None

    model = torch.nn.DataParallel(model)
    model.eval()
    prompter = NullPrompter()
    add_prompter = TokenPrompter(0)
    prompter = torch.nn.DataParallel(prompter).cuda()
    add_prompter = torch.nn.DataParallel(add_prompter).cuda()
    logging.info("done loading model.")

    if len(args.test_set) == 0:
        test_set = DATASETS
    else:
        test_set = args.test_set

    # criterion to compute attack loss, the reduction of 'sum' is important for effective attacks
    criterion_attack = torch.nn.CrossEntropyLoss(reduction='sum').to(device)

    aug = build_aug() if args.num_views > 1 else None
    validate(
        args, test_set, model, None, None, prompter,
        add_prompter, criterion_attack, None, clip_visual, aug=aug
    )

if __name__ == "__main__":
    main()
