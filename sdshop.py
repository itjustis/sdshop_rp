
import json
from IPython import display as disp
import argparse, sys

models_path = "/workspace/"
output_path = "/workspace/"

import os
import subprocess, time
sys.path.extend([
    'src/taming-transformers',
    'src/clip',
    'stable-diffusion/',
    'k-diffusion',
    'pytorch3d-lite',
    'AdaBins',
    'MiDaS',
])

model_sha256 = 'fe4efff1e174c627256e44ec2991ba279b3816e364b49f9be2abc0b3ff3f8556'
model_url =  'https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt'

def run_server(hf='',nt=''):
    from IPython import display as disp
    print('setup...')
    if nf=='xxxxxx' or nt=='xxxxxx':
        print('error: no tokens provided')
    else:
        if not os.path.exists(models_path + '/sd-v1-4.ckpt'):
        
            url = model_url
            token=hf

            headers = {"Authorization": "Bearer "+token}

            # contact server for model
            print(f"Attempting to download model...this may take a while")
            ckpt_request = requests.get(model_url, headers=headers)
            request_status = ckpt_request.status_code

            # inform user of errors
            if request_status == 403:
              raise ConnectionRefusedError("You have not accepted the license for this model.")
            elif request_status == 404:
              raise ConnectionError("Could not make contact with server")
            elif request_status != 200:
              raise ConnectionError(f"Some other error has ocurred - response code: {request_status}")

            # write to model path
            if request_status == 200:
                print('model downloaded!')
                with open(os.path.join(models_path, model_checkpoint), 'wb') as model_file:
                    model_file.write(ckpt_request.content)

        if not os.path.exists('k-diffusion/k_diffusion/__init__.py'):
                os.makedirs(models_path, exist_ok=True)
                os.makedirs(output_path, exist_ok=True)
                setup_environment = True
                print_subprocess = False
                if setup_environment:

                    print("Setting up environment...")
                    start_time = time.time()
                  

                    all_process = [

                        ['git', 'clone', 'https://github.com/deforum/stable-diffusion'],
                        ['git', 'clone', 'https://github.com/shariqfarooq123/AdaBins.git'],
                        ['git', 'clone', 'https://github.com/isl-org/MiDaS.git'],
                        ['git', 'clone', 'https://github.com/MSFTserver/pytorch3d-lite.git'],

                    ]
                    for process in all_process:
                        running = subprocess.run(process,stdout=subprocess.PIPE).stdout.decode('utf-8')
                        if print_subprocess:
                            print(running)

                    print(subprocess.run(['git', 'clone', 'https://github.com/deforum/k-diffusion/'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
                    with open('k-diffusion/k_diffusion/__init__.py', 'w') as f:
                        f.write('')
                    end_time = time.time()
                    print(f"Environment set up in {end_time-start_time:.0f} seconds")
                    
        if not os.path.exists('/workspace/temp.temp'):
            print('packages setups...')
            p_i=0
            all_process = [
                        ['pip', 'install', 'torch==1.12.1+cu113', 'torchvision==0.13.1+cu113', '--extra-index-url', 'https://download.pytorch.org/whl/cu113'],
                        ['pip', 'install', 'pandas', 'scikit-image', 'opencv-python', 'accelerate', 'ftfy', 'jsonmerge', 'matplotlib', 'resize-right', 'timm', 'torchdiffeq'],
                        ['pip', 'install', 'flask_cors', 'flask_ngrok', 'pyngrok==4.1.1', 'omegaconf==2.2.3', 'einops==0.4.1', 'pytorch-lightning==1.7.4', 'torchmetrics==0.9.3', 'torchtext==0.13.1', 'transformers==4.21.2', 'kornia==0.6.7'],
                        ['pip', 'install', '-e', 'git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers'],
                        ['pip', 'install', '-e', 'git+https://github.com/openai/CLIP.git@main#egg=clip'],
                        ['apt-get', 'update'],
                        ['apt-get', 'install', '-y', 'python3-opencv']
                    ]

            for process in all_process:
                running = subprocess.run(process,stdout=subprocess.PIPE).stdout.decode('utf-8')
                print(running)
                disp.clear_output(wait=True)
                print('please wait...')
                p_i += 1
            
            with open('/workspace/temp.temp', 'w') as f:
                f.write('temp')

        run(nt)
def run(nt):
    print('starting...')
    from IPython import display as disp
    import os
    if os.path.exists(models_path + '/sd-v1-4.ckpt'):
            import gc, math, os, pathlib, subprocess, sys, time
            import cv2
            import numpy as np
            import pandas as pd
            import random
            import requests
            import torch
            import torch.nn as nn
            import torchvision.transforms as T
            import torchvision.transforms.functional as TF

            from contextlib import contextmanager, nullcontext
            from einops import rearrange, repeat
            from omegaconf import OmegaConf
            from PIL import Image
            from pytorch_lightning import seed_everything
            from skimage.exposure import match_histograms
            from torchvision.utils import make_grid
            from tqdm import tqdm, trange
            from types import SimpleNamespace
            from torch import autocast

            from helpers import DepthModel, sampler_fn
            from k_diffusion.external import CompVisDenoiser
            from ldm.util import instantiate_from_config
            from ldm.models.diffusion.ddim import DDIMSampler
            from ldm.models.diffusion.plms import PLMSSampler

            def sanitize(prompt):
                whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                tmp = ''.join(filter(whitelist.__contains__, prompt))
                return tmp.replace(' ', '_')

            def add_noise(sample: torch.Tensor, noise_amt: float) -> torch.Tensor:
                return sample + torch.randn(sample.shape, device=sample.device) * noise_amt

            def get_output_folder(output_path, batch_folder):
                out_path = os.path.join(output_path,time.strftime('%Y-%m'))
                if batch_folder != "":
                    out_path = os.path.join(out_path, batch_folder)
                os.makedirs(out_path, exist_ok=True)
                return out_path

            def load_img(path, shape, use_alpha_as_mask=False):
                # use_alpha_as_mask: Read the alpha channel of the image as the mask image
                if path.startswith('http://') or path.startswith('https://'):
                    image = Image.open(requests.get(path, stream=True).raw)
                else:
                    image = Image.open(path)

                if use_alpha_as_mask:
                    image = image.convert('RGBA')
                else:
                    image = image.convert('RGB')

                image = image.resize(shape, resample=Image.LANCZOS)

                mask_image = None
                if use_alpha_as_mask:
                  # Split alpha channel into a mask_image
                  red, green, blue, alpha = Image.Image.split(image)
                  mask_image = alpha.convert('L')
                  image = image.convert('RGB')

                image = np.array(image).astype(np.float16) / 255.0
                image = image[None].transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
                image = 2.*image - 1.

                return image, mask_image

            def load_mask_latent(mask_input, shape):
                # mask_input (str or PIL Image.Image): Path to the mask image or a PIL Image object
                # shape (list-like len(4)): shape of the image to match, usually latent_image.shape

                if isinstance(mask_input, str): # mask input is probably a file name
                    if mask_input.startswith('http://') or mask_input.startswith('https://'):
                        mask_image = Image.open(requests.get(mask_input, stream=True).raw).convert('RGBA')
                    else:
                        mask_image = Image.open(mask_input).convert('RGBA')
                elif isinstance(mask_input, Image.Image):
                    mask_image = mask_input
                else:
                    raise Exception("mask_input must be a PIL image or a file name")

                mask_w_h = (shape[-1], shape[-2])
                mask = mask_image.resize(mask_w_h, resample=Image.LANCZOS)
                mask = mask.convert("L")
                return mask

            def prepare_mask(mask_input, mask_shape, mask_brightness_adjust=1.0, mask_contrast_adjust=1.0):
                # mask_input (str or PIL Image.Image): Path to the mask image or a PIL Image object
                # shape (list-like len(4)): shape of the image to match, usually latent_image.shape
                # mask_brightness_adjust (non-negative float): amount to adjust brightness of the iamge,
                #     0 is black, 1 is no adjustment, >1 is brighter
                # mask_contrast_adjust (non-negative float): amount to adjust contrast of the image,
                #     0 is a flat grey image, 1 is no adjustment, >1 is more contrast

                mask = load_mask_latent(mask_input, mask_shape)

                # Mask brightness/contrast adjustments
                if mask_brightness_adjust != 1:
                    mask = TF.adjust_brightness(mask, mask_brightness_adjust)
                if mask_contrast_adjust != 1:
                    mask = TF.adjust_contrast(mask, mask_contrast_adjust)

                # Mask image to array
                mask = np.array(mask).astype(np.float32) / 255.0
                mask = np.tile(mask,(4,1,1))
                mask = np.expand_dims(mask,axis=0)
                mask = torch.from_numpy(mask)

                if args.invert_mask:
                    mask = ( (mask - 0.5) * -1) + 0.5

                mask = np.clip(mask,0,1)
                return mask

            def maintain_colors(prev_img, color_match_sample, mode):
                if mode == 'Match Frame 0 RGB':
                    return match_histograms(prev_img, color_match_sample, multichannel=True)
                elif mode == 'Match Frame 0 HSV':
                    prev_img_hsv = cv2.cvtColor(prev_img, cv2.COLOR_RGB2HSV)
                    color_match_hsv = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2HSV)
                    matched_hsv = match_histograms(prev_img_hsv, color_match_hsv, multichannel=True)
                    return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2RGB)
                else: # Match Frame 0 LAB
                    prev_img_lab = cv2.cvtColor(prev_img, cv2.COLOR_RGB2LAB)
                    color_match_lab = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2LAB)
                    matched_lab = match_histograms(prev_img_lab, color_match_lab, multichannel=True)
                    return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)


            def make_callback(sampler_name, dynamic_threshold=None, static_threshold=None, mask=None, init_latent=None, sigmas=None, sampler=None, masked_noise_modifier=1.0):
                # Creates the callback function to be passed into the samplers
                # The callback function is applied to the image at each step
                def dynamic_thresholding_(img, threshold):
                    # Dynamic thresholding from Imagen paper (May 2022)
                    s = np.percentile(np.abs(img.cpu()), threshold, axis=tuple(range(1,img.ndim)))
                    s = np.max(np.append(s,1.0))
                    torch.clamp_(img, -1*s, s)
                    torch.FloatTensor.div_(img, s)

                # Callback for samplers in the k-diffusion repo, called thus:
                #   callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
                def k_callback_(args_dict):
                    if dynamic_threshold is not None:
                        dynamic_thresholding_(args_dict['x'], dynamic_threshold)
                    if static_threshold is not None:
                        torch.clamp_(args_dict['x'], -1*static_threshold, static_threshold)
                    if mask is not None:
                        init_noise = init_latent + noise * args_dict['sigma']
                        is_masked = torch.logical_and(mask >= mask_schedule[args_dict['i']], mask != 0 )
                        new_img = init_noise * torch.where(is_masked,1,0) + args_dict['x'] * torch.where(is_masked,0,1)
                        args_dict['x'].copy_(new_img)

                # Function that is called on the image (img) and step (i) at each step
                def img_callback_(img, i):
                    # Thresholding functions
                    if dynamic_threshold is not None:
                        dynamic_thresholding_(img, dynamic_threshold)
                    if static_threshold is not None:
                        torch.clamp_(img, -1*static_threshold, static_threshold)
                    if mask is not None:
                        i_inv = len(sigmas) - i - 1
                        init_noise = sampler.stochastic_encode(init_latent, torch.tensor([i_inv]*batch_size).to(device), noise=noise)
                        is_masked = torch.logical_and(mask >= mask_schedule[i], mask != 0 )
                        new_img = init_noise * torch.where(is_masked,1,0) + img * torch.where(is_masked,0,1)
                        img.copy_(new_img)

                if init_latent is not None:
                    noise = torch.randn_like(init_latent, device=device) * masked_noise_modifier
                if sigmas is not None and len(sigmas) > 0:
                    mask_schedule, _ = torch.sort(sigmas/torch.max(sigmas))
                elif len(sigmas) == 0:
                    mask = None # no mask needed if no steps (usually happens because strength==1.0)
                if sampler_name in ["plms","ddim"]:
                    # Callback function formated for compvis latent diffusion samplers
                    if mask is not None:
                        assert sampler is not None, "Callback function for stable-diffusion samplers requires sampler variable"
                        batch_size = init_latent.shape[0]

                    callback = img_callback_
                else:
                    # Default callback function uses k-diffusion sampler variables
                    callback = k_callback_

                return callback

            def sample_from_cv2(sample: np.ndarray) -> torch.Tensor:
                sample = ((sample.astype(float) / 255.0) * 2) - 1
                sample = sample[None].transpose(0, 3, 1, 2).astype(np.float16)
                sample = torch.from_numpy(sample)
                return sample

            def sample_to_cv2(sample: torch.Tensor, type=np.uint8) -> np.ndarray:
                sample_f32 = rearrange(sample.squeeze().cpu().numpy(), "c h w -> h w c").astype(np.float32)
                sample_f32 = ((sample_f32 * 0.5) + 0.5).clip(0, 1)
                sample_int8 = (sample_f32 * 255)
                return sample_int8.astype(type)


            import cv2

            def imgtobytes(image):
              success, encoded_image = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
              content2 = encoded_image.tobytes()
              return content2

            def load_img(image, shape):

                image = image.resize(shape, resample=Image.LANCZOS)
                image = np.array(image).astype(np.float16) / 255.0
                image = image[None].transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
                return 2.*image - 1.

            def generate(args, return_latent=False, return_sample=False):

                seed_everything(args.seed)
                #os.makedirs(args.outdir, exist_ok=True)

                if args.sampler == 'plms':
                    sampler = PLMSSampler(model)
                else:
                    sampler = DDIMSampler(model)

                model_wrap = CompVisDenoiser(model)
                batch_size = args.n_samples
                prompt = args.prompt
                assert prompt is not None
                data = [batch_size * [prompt]]

                init_latent = None
                #if args.init_latent is not None:
                #    init_latent = args.init_latent
                #elif args.init_sample is not None:
                #    init_latent = model.get_first_stage_encoding(model.encode_first_stage(args.init_sample))

                if args.use_init and args.init_image != None and args.init_image != '':
                    init_image = load_img(args.init_image, shape=(args.W, args.H)).to(device)


                    #init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
                    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

                if not args.use_init and args.strength > 0:
                    print("\nNo init image, but strength > 0. This may give you some strange results.\n")

                sampler.make_schedule(ddim_num_steps=args.steps, ddim_eta=args.ddim_eta, verbose=False)

                t_enc = int((1.0-args.strength) * args.steps)

                start_code = None
                if args.fixed_code and init_latent == None:
                    start_code = torch.randn([args.n_samples, args.C, args.H // args.f, args.W // args.f], device=device)

                callback = make_callback(sampler=args.sampler,
                                        dynamic_threshold=args.dynamic_threshold,
                                        static_threshold=args.static_threshold)

                results = []
                precision_scope = autocast if args.precision == "autocast" else nullcontext
                with torch.no_grad():
                    with precision_scope("cuda"):
                        with model.ema_scope():
                            for n in range(args.n_samples):
                                for prompts in data:
                                    uc = None
                                    if args.scale != 1.0:
                                        uc = model.get_learned_conditioning(batch_size * [""])
                                    if isinstance(prompts, tuple):
                                        prompts = list(prompts)
                                    c = model.get_learned_conditioning(prompts)


                                    if args.sampler in ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral"]:
                                        shape = [args.C, args.H // args.f, args.W // args.f]
                                        sigmas = model_wrap.get_sigmas(args.steps)
                                        if args.use_init:
                                            sigmas = sigmas[len(sigmas)-t_enc-1:]
                                            x = init_latent + torch.randn([args.n_samples, *shape], device=device) * sigmas[0]
                                        else:
                                            x = torch.randn([args.n_samples, *shape], device=device) * sigmas[0]
                                        model_wrap_cfg = CFGDenoiser(model_wrap)
                                        extra_args = {'cond': c, 'uncond': uc, 'cond_scale': args.scale}
                                        if args.sampler=="klms":
                                            samples = sampling.sample_lms(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=False, callback=callback)
                                        elif args.sampler=="dpm2":
                                            samples = sampling.sample_dpm_2(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=False, callback=callback)
                                        elif args.sampler=="dpm2_ancestral":
                                            samples = sampling.sample_dpm_2_ancestral(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=False, callback=callback)
                                        elif args.sampler=="heun":
                                            samples = sampling.sample_heun(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=False, callback=callback)
                                        elif args.sampler=="euler":
                                            samples = sampling.sample_euler(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=False, callback=callback)
                                        elif args.sampler=="euler_ancestral":
                                            samples = sampling.sample_euler_ancestral(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=False, callback=callback)
                                    else:

                                        if init_latent != None:
                                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                                            samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=args.scale,
                                                                    unconditional_conditioning=uc,)
                                        else:
                                            if args.sampler == 'plms' or args.sampler == 'ddim':
                                                shape = [args.C, args.H // args.f, args.W // args.f]
                                                samples, _ = sampler.sample(S=args.steps,
                                                                                conditioning=c,
                                                                                batch_size=args.n_samples,
                                                                                shape=shape,
                                                                                verbose=False,
                                                                                unconditional_guidance_scale=args.scale,
                                                                                unconditional_conditioning=uc,
                                                                                eta=args.ddim_eta,
                                                                                x_T=start_code,
                                                                                img_callback=callback)

                                    if return_latent:
                                        results.append(samples.clone())

                                    x_samples = model.decode_first_stage(samples)
                                    if return_sample:
                                        results.append(x_samples.clone())

                                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                                    for x_sample in x_samples:
                                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                        image = Image.fromarray(x_sample.astype(np.uint8))
                                        results.append(image)
                return results


            #@markdown **Select and Load Model**

            model_config = "v1-inference.yaml" #@param ["custom","v1-inference.yaml"]
            model_checkpoint =  "sd-v1-4.ckpt" #@param ["custom","sd-v1-4-full-ema.ckpt","sd-v1-4.ckpt","sd-v1-3-full-ema.ckpt","sd-v1-3.ckpt","sd-v1-2-full-ema.ckpt","sd-v1-2.ckpt","sd-v1-1-full-ema.ckpt","sd-v1-1.ckpt"]
            custom_config_path = "" #@param {type:"string"}
            custom_checkpoint_path = "" #@param {type:"string"}

            load_on_run_all = True #@param {type: 'boolean'}
            half_precision = True # check
            check_sha256 = False #@param {type:"boolean"}

            model_map = {
                "sd-v1-4-full-ema.ckpt": {'sha256': '14749efc0ae8ef0329391ad4436feb781b402f4fece4883c7ad8d10556d8a36a'},
                "sd-v1-4.ckpt": {'sha256': 'fe4efff1e174c627256e44ec2991ba279b3816e364b49f9be2abc0b3ff3f8556'},
                "sd-v1-3-full-ema.ckpt": {'sha256': '54632c6e8a36eecae65e36cb0595fab314e1a1545a65209f24fde221a8d4b2ca'},
                "sd-v1-3.ckpt": {'sha256': '2cff93af4dcc07c3e03110205988ff98481e86539c51a8098d4f2236e41f7f2f'},
                "sd-v1-2-full-ema.ckpt": {'sha256': 'bc5086a904d7b9d13d2a7bccf38f089824755be7261c7399d92e555e1e9ac69a'},
                "sd-v1-2.ckpt": {'sha256': '3b87d30facd5bafca1cbed71cfb86648aad75d1c264663c0cc78c7aea8daec0d'},
                "sd-v1-1-full-ema.ckpt": {'sha256': 'efdeb5dc418a025d9a8cc0a8617e106c69044bc2925abecc8a254b2910d69829'},
                "sd-v1-1.ckpt": {'sha256': '86cd1d3ccb044d7ba8db743d717c9bac603c4043508ad2571383f954390f3cea'}
            }

            # config path
            ckpt_config_path = custom_config_path if model_config == "custom" else os.path.join(models_path, model_config)
            if os.path.exists(ckpt_config_path):
                print(f"{ckpt_config_path} exists")
            else:
                ckpt_config_path = "./stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
            print(f"Using config: {ckpt_config_path}")

            # checkpoint path or download
            ckpt_path = custom_checkpoint_path if model_checkpoint == "custom" else os.path.join(models_path, model_checkpoint)
            ckpt_valid = True
            if os.path.exists(ckpt_path):
                print(f"{ckpt_path} exists")
            else:
                print(f"Please download model checkpoint and place in {os.path.join(models_path, model_checkpoint)}")
                ckpt_valid = False

            if check_sha256 and model_checkpoint != "custom" and ckpt_valid:
                import hashlib
                print("\n...checking sha256")
                with open(ckpt_path, "rb") as f:
                    bytes = f.read()
                    hash = hashlib.sha256(bytes).hexdigest()
                    del bytes
                if model_map[model_checkpoint]["sha256"] == hash:
                    print("hash is correct\n")
                else:
                    print("hash in not correct\n")
                    ckpt_valid = False

            if ckpt_valid:
                print(f"Using ckpt: {ckpt_path}")

            def load_model_from_config(config, ckpt, verbose=False, device='cuda', half_precision=True):
                map_location = "cuda" #@param ["cpu", "cuda"]
                print(f"Loading model from {ckpt}")
                pl_sd = torch.load(ckpt, map_location=map_location)
                if "global_step" in pl_sd:
                    print(f"Global Step: {pl_sd['global_step']}")
                sd = pl_sd["state_dict"]
                model = instantiate_from_config(config.model)
                m, u = model.load_state_dict(sd, strict=False)
                if len(m) > 0 and verbose:
                    print("missing keys:")
                    print(m)
                if len(u) > 0 and verbose:
                    print("unexpected keys:")
                    print(u)

                if half_precision:
                    model = model.half().to(device)
                else:
                    model = model.to(device)
                model.eval()
                return model

            if load_on_run_all and ckpt_valid:
                local_config = OmegaConf.load(f"{ckpt_config_path}")
                model = load_model_from_config(local_config, f"{ckpt_path}", half_precision=half_precision)
                device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                model = model.to(device)


            import cv2
            from PIL import ImageOps


            def imgtobytes(image):
              success, encoded_image = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
              content2 = encoded_image.tobytes()
              return content2




            def load_img(image, shape, use_alpha_as_mask=False):
                # use_alpha_as_mask: Read the alpha channel of the image as the mask image

                image = image.resize(shape, resample=Image.LANCZOS)

                mask_image = None
                if use_alpha_as_mask:
                  # Split alpha channel into a mask_image
                  red, green, blue, alpha = Image.Image.split(image)
                  mask_image = alpha.convert('L')
                  mask_image = ImageOps.invert(mask_image)
                  image = image.convert('RGB')

                image = np.array(image).astype(np.float16) / 255.0
                image = image[None].transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
                image = 2.*image - 1.

                return image, mask_image


            def generate(args, return_latent=False, return_sample=False, return_c=False):
                seed_everything(args.seed)
                os.makedirs(args.outdir, exist_ok=True)

                sampler = PLMSSampler(model) if args.sampler == 'plms' else DDIMSampler(model)
                model_wrap = CompVisDenoiser(model)
                batch_size = args.n_samples
                prompt = args.prompt
                assert prompt is not None
                data = [batch_size * [prompt]]
                precision_scope = autocast if args.precision == "autocast" else nullcontext

                init_latent = None
                mask_image = None
                init_image = None
                if args.init_latent is not None:
                    init_latent = args.init_latent
                elif args.init_sample is not None:
                    with precision_scope("cuda"):
                        init_latent = model.get_first_stage_encoding(model.encode_first_stage(args.init_sample))
                elif args.use_init and args.init_image != None and args.init_image != '':
                    init_image, mask_image = load_img(args.init_image,
                                                      shape=(args.W, args.H),
                                                      use_alpha_as_mask=args.use_alpha_as_mask)
                    init_image = init_image.to(device)
                    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
                    with precision_scope("cuda"):
                        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

                if not args.use_init and args.strength > 0 and args.strength_0_no_init:
                    print("\nNo init image, but strength > 0. Strength has been auto set to 0, since use_init is False.")
                    print("If you want to force strength > 0 with no init, please set strength_0_no_init to False.\n")
                    args.strength = 0

                # Mask functions
                if args.use_mask:
                    assert args.mask_file is not None or mask_image is not None, "use_mask==True: An mask image is required for a mask. Please enter a mask_file or use an init image with an alpha channel"
                    assert args.use_init, "use_mask==True: use_init is required for a mask"
                    assert init_latent is not None, "use_mask==True: An latent init image is required for a mask"

                    mask = prepare_mask(args.mask_file if mask_image is None else mask_image,
                                        init_latent.shape,
                                        args.mask_contrast_adjust,
                                        args.mask_brightness_adjust)

                    if (torch.all(mask == 0) or torch.all(mask == 1)) and args.use_alpha_as_mask:
                        raise Warning("use_alpha_as_mask==True: Using the alpha channel from the init image as a mask, but the alpha channel is blank.")

                    #mask = mask.to(device)
                    #mask = repeat(mask, '1 ... -> b ...', b=batch_size)

                    init_mask = mask_image
                    latmask = init_mask.convert('RGB').resize((init_latent.shape[3], init_latent.shape[2]))
                    latmask = np.moveaxis(np.array(latmask, dtype=np.float32), 2, 0) / 255
                    latmask = latmask[0]
                    latmask = np.around(latmask)
                    latmask = np.tile(latmask[None], (4, 1, 1))

                    mask = torch.asarray(1.0 - latmask).to(device).type(model.dtype)
                    nmask = torch.asarray(latmask).to(device).type(model.dtype)

                    init_latent = init_latent * mask + torch.randn_like(init_latent, device=device) * 1.0 * nmask
                else:
                    mask = None

                t_enc = int((1.0-args.strength) * args.steps)

                # Noise schedule for the k-diffusion samplers (used for masking)
                k_sigmas = model_wrap.get_sigmas(args.steps)
                k_sigmas = k_sigmas[len(k_sigmas)-t_enc-1:]

                if args.sampler in ['plms','ddim']:
                    sampler.make_schedule(ddim_num_steps=args.steps, ddim_eta=args.ddim_eta, ddim_discretize='fill', verbose=False)

                callback = make_callback(sampler_name=args.sampler,
                                        dynamic_threshold=args.dynamic_threshold,
                                        static_threshold=args.static_threshold,
                                        mask=mask,
                                        init_latent=init_latent,
                                        sigmas=k_sigmas,
                                        sampler=sampler)

                results = []
                with torch.no_grad():
                    with precision_scope("cuda"):
                        with model.ema_scope():
                            for prompts in data:
                                uc = None
                                if args.scale != 1.0:
                                    uc = model.get_learned_conditioning(batch_size * [""])
                                if isinstance(prompts, tuple):
                                    prompts = list(prompts)
                                c = model.get_learned_conditioning(prompts)

                                if args.init_c != None:
                                    c = args.init_c

                                if args.sampler in ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral"]:
                                    samples = sampler_fn(
                                        c=c,
                                        uc=uc,
                                        args=args,
                                        model_wrap=model_wrap,
                                        init_latent=init_latent,
                                        t_enc=t_enc,
                                        device=device,
                                        cb=callback)
                                else:
                                    # args.sampler == 'plms' or args.sampler == 'ddim':
                                    if init_latent is not None and args.strength > 0:
                                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                                    else:
                                        z_enc = torch.randn([args.n_samples, args.C, args.H // args.f, args.W // args.f], device=device)
                                    if args.sampler == 'ddim':
                                        samples = sampler.decode(z_enc,
                                                                 c,
                                                                 t_enc,
                                                                 unconditional_guidance_scale=args.scale,
                                                                 unconditional_conditioning=uc,
                                                                 img_callback=callback)
                                    elif args.sampler == 'plms': # no "decode" function in plms, so use "sample"
                                        shape = [args.C, args.H // args.f, args.W // args.f]
                                        samples, _ = sampler.sample(S=args.steps,
                                                                        conditioning=c,
                                                                        batch_size=args.n_samples,
                                                                        shape=shape,
                                                                        verbose=False,
                                                                        unconditional_guidance_scale=args.scale,
                                                                        unconditional_conditioning=uc,
                                                                        eta=args.ddim_eta,
                                                                        x_T=z_enc,
                                                                        img_callback=callback)
                                    else:
                                        raise Exception(f"Sampler {args.sampler} not recognised.")

                                if return_latent:
                                    results.append(samples.clone())

                                x_samples = model.decode_first_stage(samples)
                                if return_sample:
                                    results.append(x_samples.clone())

                                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                                if return_c:
                                    results.append(c.clone())

                                for x_sample in x_samples:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    image = Image.fromarray(x_sample.astype(np.uint8))
                                    results.append(image)
                return results
            def DeforumArgs():

                #@markdown ** Settings**


                dynamic_threshold = None
                static_threshold = None


                save_samples = False
                save_settings = False
                display_samples = False


                n_batch = 1
                batch_name = "StableFun"
                filename_format = "{timestring}_{index}_{prompt}.png"
                seed_behavior = "iter"
                make_grid = False
                grid_rows = 2
                outdir = get_output_folder(output_path, batch_name)


                use_init = False
                strength = 0.0
                strength_0_no_init = True
                init_image = "https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg"

                use_mask = False
                use_alpha_as_mask = False
                mask_file = "https://www.filterforge.com/wiki/images/archive/b/b7/20080927223728%21Polygonal_gradient_thumb.jpg"
                invert_mask = False

                mask_brightness_adjust = 1.0
                mask_contrast_adjust = 1.0

                n_samples = 1
                precision = 'autocast'
                C = 4
                f = 8

                prompt = ""
                timestring = ""
                init_latent = None
                init_sample = None
                init_c = None

                return locals()
            args = SimpleNamespace(**DeforumArgs())





            import base64
            import torch
            import logging
            from flask import Flask, Response, request, send_file, abort, stream_with_context
            from flask_cors import CORS
            from PIL import Image
            from io import BytesIO


            from click import secho
            from zipfile import ZipFile

            # the following line is specific to remote environments (like google colab)

            # the following line is specific to remote environments (like google colab)
            from flask_ngrok import run_with_ngrok

            # Load the model for use (this may take a minute or two...or three)
            secho("Loading Model...", fg="yellow")

            secho("Finished!", fg="green")

            disp.clear_output(wait=True)
            # Start setting up flask
            #logging.getLogger('flask_cors').level = logging.DEBUG

            app = Flask(__name__)


            CORS(app)
            # Define a function to help us "control the randomness"

            samplers_list=["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim"]
            fail_res = Response(
                        json.dumps({"message": 'error', "code": 400, "status": "FAIL"}),
                        mimetype="application/json",
                        status=400,
                    )


            def get_name(prompt, seed):
              return f'{prompt}-{seed}'

            @app.route("/api/check", methods=["POST"])
            def check():
                try:
                  r = request
                  headers = r.headers
                  if headers["message"] == "hello":
                    return Response(response="{}", status=200)
                  else:
                    return abort(fail_res)
                except:
                  return abort(fail_res)

            #display.clear_output()
            @app.route("/api/img2img", methods=["POST"])
            def img2img():
              try:
                r = request
                headers = r.headers
                #print(r.headers)
                inpaint = headers["inpaint"]
                ######
                seed = int(headers["seed"])
                variation = int(headers['variation'])+1
                prompt = headers['prompt']

                args.seed = seed
                args.prompt = prompt
                args.strength = float(headers['strength'])
                args.steps = int(headers['steps'])

                if not  headers['sampler'] in samplers_list:
                  args.sampler = 'euler'
                else:
                  args.sampler =  headers['sampler']

                if args.sampler == 'ddim':
                  args.ddim_eta = float(headers['ddim_eta'])
                else:
                  args.ddim_eta = 0


                if inpaint=="true":
                  args.use_alpha_as_mask = True
                  args.use_mask = True
                  args.strength = 0.2
                else:
                  args.use_alpha_as_mask = False
                  args.use_mask = False
                W_in, H_in = int(headers["W"]), int(headers["H"])
                W, H = map(lambda x: x - x % 64, (W_in, H_in))

                args.W = W
                args.H = H


                data = r.data
                if variation == 1:
                  f = BytesIO()
                  f.write(base64.b64decode(data))
                  f.seek(0)
                  if inpaint=="true":
                    img = Image.open(f)
                  else:
                    img = Image.open(f).convert("RGB")

                  newsize = (W, H)
                  img = img.resize(newsize)

                  args.init_image = img
                  args.use_init=True
                print('variation # '+str(variation))


                args.scale = float(headers['scale'])

                results = generate(args, return_latent=False, return_sample=False)

                newsize = (W_in, H_in)
                img = results[0]
                img = img.resize(newsize)

                return_image = imgtobytes(np.asarray(img))
                return Response(response=return_image, status=200, mimetype="image/png")

              except Exception as e:
                print("type error: " + str(e))
                if (args.sampler=='plms'):
                  return abort(fail_res)
                else:
                  return abort(fail_res)



            @app.route("/api/txt2img", methods=["POST"])
            def txt2img():
              try:
                r = request
                headers = r.headers
                #print(r.headers)
                W_in, H_in = int(headers["W"]), int(headers["H"])
                W, H = map(lambda x: x - x % 64, (W_in, H_in))
                args.W = W
                args.H = H
                args.use_mask = False

                ######
                seed = int(headers["seed"])
                prompt = headers['prompt']
                ######
                if not  headers['sampler'] in samplers_list:
                  args.sampler = 'ddim'
                else:
                  args.sampler =  headers['sampler']

                if args.sampler == 'ddim':
                  args.ddim_eta = float(headers['ddim_eta'])
                else:
                  args.ddim_eta = 0

                print(args.sampler)

                ######
                args.use_init=False
                args.seed = seed
                args.prompt = prompt
                args.strength = 0
                args.steps = int(headers['steps'])
                args.scale = float(headers['scale'])
                #########

                seed_everything(seed)

                results = generate(args, return_latent=False, return_sample=False)

                newsize = (W_in, H_in)
                img = results[0]
                img = img.resize(newsize)

                return_image = imgtobytes(np.asarray(img))
                return Response(response=return_image, status=200, mimetype="image/png")


              except Exception as e:
                print("type error: " + str(e))
                if (args.sampler=='plms'):

                  res = Response(
                        json.dumps({"message": 'error', "code": 400, "status": "FAIL"}),
                        mimetype="application/json",
                        status=400,
                    )
                  return abort(res)
                else:

                  res = Response(
                        json.dumps({"message": 'error', "code": 400, "status": "FAIL"}),
                        mimetype="application/json",
                        status=400,
                    )
                  return abort(res)


            if len(nt)>0:
                print(subprocess.run(['ngrok', 'authtoken', nt]))
            else:
                print ('looks like no ngrok token provided')
            run_with_ngrok(app)
            disp.clear_output(wait=True)
            print('COPY & PASTE NGROK URL  ( "running on..." ) TO PHOTOSHOP PLUGIN API FIELD')
            app.run()
