from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler, EulerDiscreteScheduler, UNet2DConditionModel
from diffusers.utils import make_image_grid

import sys

#from pipeline_stable_diffusion_xl_ipex import StableDiffusionXLPipelineIpex

import argparse
import time
import torch
import datetime
from datetime import datetime, date, timedelta
from safetensors.torch import load_file

device="cpu"

import psutil

profile = 0

def get_host_memory():
    memory_allocated = round(psutil.Process().memory_info().rss / 1024**3, 3)
    print("cpu"," memory used total: ", memory_allocated, "GB")

# model_id = "/home/una/hf_models/sdxl-turbo"

base = "/home/models/HF_models/stable-diffusion-xl-base-1.0/"
repo = "/home/models/HF_models/SDXL-Lightning/"
# ckpt = "sdxl_lightning_4step_lora.safetensors" # Use the correct ckpt for your step setting!
ckpt = "sdxl_lightning_4step_unet.safetensors"

# Load model.
unet = UNet2DConditionModel.from_config(base, subfolder="unet")
unet.load_state_dict(load_file(f"{repo}{ckpt}", device="cpu"))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--bf16', default=True, action='store_true', help="FP32 - Default")
    parser.add_argument("--batch", default=1, type=int, help="batch size")
    parser.add_argument("--height", default=512, type=int, help="Height")
    parser.add_argument("--width", default=512, type=int, help="Width")
    parser.add_argument("--step", default=4, type=int, help="Revolving loop")
    parser.add_argument("--repeat", default=3, type=int, help="Repeat Infer Time")
    parser.add_argument("--prompt", default="beautiful beach with white sands, trophical island, coconat tree, a beautiful boat on the beach, moon night, 8k", type=str, help="Input Prompt")

    args = parser.parse_args()
    
    prompt = args.batch * [args.prompt]
    
    #scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    #scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    #pipe = StableDiffusionXLPipelineIpex.from_pretrained(model_id, scheduler = scheduler, use_auth_token=True, low_cpu_mem_usage=True, use_safetensors = True, safety_checker = None)
    # pipe = StableDiffusionXLPipelineIpex.from_pretrained(model_id, use_auth_token=True, low_cpu_mem_usage=True, use_safetensors = True, safety_checker = None)
    torchdtype = torch.bfloat16 if args.bf16 else torch.float32
    variant = "bf16" if args.bf16 else "fp32"

    # original pipe
    # pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torchdtype, variant=variant)
    # optimized pipe
    '''
    pipe = StableDiffusionXLPipelineIpex.from_pretrained(
            base, unet=unet, torch_dtype=torchdtype, variant=variant,
            use_auth_token=True, 
            low_cpu_mem_usage=True, 
            use_safetensors = True, 
            safety_checker = None)
    '''

    pipe = StableDiffusionXLPipeline.from_pretrained(
            base, unet=unet, torch_dtype=torchdtype, variant=variant,
            use_auth_token=True, 
            low_cpu_mem_usage=True, 
            use_safetensors = True, 
            safety_checker = None, custom_pipeline="pipeline_stable_diffusion_xl_ipex")
    
    inference_step = args.step
    
    get_host_memory()
    
    data_type = torch.bfloat16 if args.bf16 else torch.float32
    # data_type = torch.float32 if args.bf16 else torch.float32
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    
    pipe.prepare_for_ipex(data_type, prompt, height=args.height, width=args.width, guidance_scale=0.0)

    get_host_memory()
    
    generator = torch.Generator(device).manual_seed(4)
    with torch.no_grad(), torch.cpu.amp.autocast(enabled=args.bf16, dtype=torch.bfloat16):
        if profile > 0:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=2)
            ) as p:
                for i in range(8):
                    print('')
                    t1 = time.time()
                    image = pipe(prompt, num_inference_steps=inference_step, height=args.height, width=args.width, guidance_scale=0.0)
                    t2 = time.time()
                    print('SDXL-Lightning inference latency: {:.3f} sec'.format(t2-t1))
                    print('******************************')
                    print('')
                    p.step()
                p.export_chrome_trace("sdxl_lightning_ipex_trace" + str(p.step_num) + datetime.today().isoformat() + ".json")
                print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
                print("")
        else:
            sum = 0
            images = []
            for i in range(args.repeat):
                print('')
                t1 = time.time()
                image = pipe(prompt, num_inference_steps=inference_step, height=args.height, width=args.width, guidance_scale=0.0, generator = generator)
                t2 = time.time()
            
                print('SDXL-Lightning inference latency: {:.3f} sec'.format(t2-t1))
                print('******************************')
                print('')
                
                get_host_memory()
                
                if i > 0 :
                    sum += t2 - t1

                for j in range(args.batch):
                    image.images[j].save("saved_pic" + str(j) + "ite" + str(i) + ".png")

                    images.append(image.images[j])

            out = make_image_grid(images, rows = args.batch, cols = args.repeat) 
            post_fix = "bf16" if args.bf16 else "fp32"
            out.save("grid_" + post_fix  +".png")
             
            print("finish job! avg latency {0:4.4f}s".format(sum / (args.repeat - 1)))

