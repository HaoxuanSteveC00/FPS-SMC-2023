import os

from functools import partial
import argparse
import yaml

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

# from piq import psnr, ssim # Original psnr, ssim from piq
# from piq.perceptual import LPIPS # Original LPIPS from piq
from skimage.metrics import peak_signal_noise_ratio # For PSNR as in compute_metric.py
import lpips # For LPIPS as in compute_metric.py
import csv # For logging metrics

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from guided_diffusion.svd_replacement import Deblurring, Deblurring2D
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator, Blurkernel #, _transform
from util.logger import get_logger


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--c_rate', type=float, default=0.95)
    parser.add_argument('--particle_size', type=int, default=5)
    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Initialize LPIPS model
    loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)

    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
   
    #assert model_config['learn_sigma'] == diffusion_config['learn_sigma'], \
    #"learn_sigma must be the same for model and diffusion configuartion."
    
    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
   
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config, c_rate=args.c_rate, particle_size=args.particle_size) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
   
    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Setup metrics logging
    metrics_log_dir = os.path.join(out_path, 'live_metrics_log')
    os.makedirs(metrics_log_dir, exist_ok=True)
    metrics_log_path = os.path.join(metrics_log_dir, 'live_metrics.csv')
    
    # Write header to CSV if file is new
    if not os.path.exists(metrics_log_path) or os.path.getsize(metrics_log_path) == 0:
        with open(metrics_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_index', 'psnr', 'lpips'])
            
    psnr_all = []
    lpips_all = []

    # Prepare dataloader
    data_config = task_config['data']
    batch_size = 1  # Do not change this value. Larger batch size is not available for particle size > 1.
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.CenterCrop((256, 256)),
                                    transforms.Resize((256, 256)),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # Preprocessing shared by FFHQ and ImageNet.
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=batch_size, num_workers=0, train=False)

    # (Exception) In case of inpainting, we need to generate a mask 
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )
        
    # Do inference
    
    pbar = tqdm(enumerate(loader), total=len(loader), desc="Processing images")
    for i, ref_img in pbar:

        # logger.info(f"Inference for image {i}")
        fnames = [str(j).zfill(5) + '.png' for j in range(i * batch_size, (i+1) * batch_size)]
        ref_img = ref_img.to(device)

        if measure_config['operator'] ['name'] == 'inpainting':
            # Masks only exist in the inpainting tasks.
            mask = mask_gen(ref_img)
            mask = mask[0, 0, :, :].unsqueeze(dim=0).unsqueeze(dim=0)
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn, operator = operator, op = 'inpainting', mask = mask)

            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)
        
        elif measure_config['operator'] ['name'] == 'gaussian_blur':

            sample_fn = partial(sample_fn, operator = operator, op = 'gaussian_blur', mask = None)
            
            kernel = operator.get_kernel().type(torch.float64).reshape(61,61)
            kernel = kernel[30,:] / torch.sqrt(kernel[30,30])
            task_Svd = Deblurring(kernel=kernel, channels=3, img_dim=256, device=device)

            y = task_Svd.forward(ref_img)
            y_n = noiser(y)
            
        elif measure_config['operator'] ['name'] == 'motion_blur':

            sample_fn = partial(sample_fn, operator = operator, op = 'motion_blur', mask = None)
            
            kernel = operator.get_kernel().type(torch.float64).reshape(61,61)
            kernel1 = kernel[30,:] / torch.sum(kernel[30,:])
            kernel1 = torch.tensor([0.0] * 30 + [1.0] + [0.0] * 30)
            
            conv2 = Blurkernel(blur_type='gaussian',
                               kernel_size=61,
                               std=0.5,
                               device=device).to(device)
            kernel2 = conv2.get_kernel().view(1,1,61,61).type(torch.float64).reshape(61,61)
            kernel2 = kernel2[30,:] / torch.sum(kernel2[30,:])
            
            task_Svd = Deblurring2D(kernel1=kernel1, kernel2=kernel2, channels=3, img_dim=256, device=device)

            y = task_Svd.forward(ref_img)
            y_n = noiser(y)
        
        else: 
            sample_fn = partial(sample_fn, operator = operator, op = 'super_resolution', mask = None)
            y = operator.forward(ref_img)
            y_n = noiser(y)
         
        # Sampling
        # If you wish to record the intermediate steps, turn record = True below.
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        sample = sample_fn(x_start=x_start, measurement=y_n, record=False, save_root=out_path).requires_grad_()
        
        # Compute metrics
        # Ensure tensors are detached and correctly formatted for metric functions
        ref_img_detached = ref_img.detach()
        sample_detached = sample.detach()

        # PSNR calculation (expects numpy arrays, range [0, 1])
        ref_img_np = clear_color(ref_img_detached.clone()) # Use clone if clear_color modifies input
        sample_np = clear_color(sample_detached.clone())
        current_psnr = peak_signal_noise_ratio(ref_img_np, sample_np, data_range=1.0)
        
        # LPIPS calculation (expects PyTorch tensors, range [-1, 1], BCHW)
        # ref_img and sample are already in [-1, 1] and BCHW
        current_lpips = loss_fn_lpips(sample_detached, ref_img_detached).item()
        
        psnr_all.append(current_psnr)
        lpips_all.append(current_lpips)
        
        # Log metrics to CSV
        with open(metrics_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([i, current_psnr, current_lpips])
            
        # Calculate running statistics
        avg_psnr = np.mean(psnr_all)
        avg_lpips = np.mean(lpips_all)
        
        n_samples = len(psnr_all)
        if n_samples > 1:
            std_psnr = np.std(psnr_all, ddof=0) # population std for consistency if not specified
            std_lpips = np.std(lpips_all, ddof=0)
            ci_psnr = 1.96 * (std_psnr / np.sqrt(n_samples))
            ci_lpips = 1.96 * (std_lpips / np.sqrt(n_samples))
        else:
            std_psnr = 0
            std_lpips = 0
            ci_psnr = 0
            ci_lpips = 0
            
        # Update tqdm description
        pbar.set_description(
            f"Img: {i} | Avg PSNR: {avg_psnr:.2f} ± {ci_psnr:.2f} | Avg LPIPS: {avg_lpips:.4f} ± {ci_lpips:.4f}"
        )
        pbar.set_postfix_str(f"Current PSNR: {current_psnr:.2f}, Current LPIPS: {current_lpips:.4f}")

        # for _ in range(batch_size):
        #     plt.imsave(os.path.join(out_path, 'input', fnames[_]), clear_color(y_n[_,:,:,:].unsqueeze(dim=0)))
        #     plt.imsave(os.path.join(out_path, 'label', fnames[_]), clear_color(ref_img[_,:,:,:].unsqueeze(dim=0)))
        #     plt.imsave(os.path.join(out_path, 'recon', fnames[_]), clear_color(sample[_,:,:,:].unsqueeze(dim=0)))

    # Log final summary
    if psnr_all and lpips_all: # Ensure lists are not empty
        final_avg_psnr = np.mean(psnr_all)
        final_avg_lpips = np.mean(lpips_all)
        n_total = len(psnr_all)
        if n_total > 1:
            final_std_psnr = np.std(psnr_all, ddof=0)
            final_std_lpips = np.std(lpips_all, ddof=0)
            final_ci_psnr = 1.96 * (final_std_psnr / np.sqrt(n_total))
            final_ci_lpips = 1.96 * (final_std_lpips / np.sqrt(n_total))
        else:
            final_ci_psnr = 0
            final_ci_lpips = 0
        logger.info(f"Finished processing {n_total} images.")
        logger.info(f"Final Average PSNR: {final_avg_psnr:.2f} ± {final_ci_psnr:.2f}")
        logger.info(f"Final Average LPIPS: {final_avg_lpips:.4f} ± {final_ci_lpips:.4f}")
        logger.info(f"Metrics saved to {metrics_log_path}")

if __name__ == '__main__':
    main()
