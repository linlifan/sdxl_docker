from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler, EulerDiscreteScheduler
from diffusers.utils import make_image_grid

import sys
from pipeline_stable_diffusion_xl_ipex import StableDiffusionXLPipelineIpex

import argparse
import time
import torch
import datetime
from datetime import datetime, date, timedelta

device="cpu"

import psutil

profile = 0

def get_host_memory():
    memory_allocated = round(psutil.Process().memory_info().rss / 1024**3, 3)
    print("cpu"," memory used total: ", memory_allocated, "GB")

model_id = "/home/una/hf_models/sdxl-turbo"

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--bf16', default=False, action='store_true', help="FP32 - Default")
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
    pipe = StableDiffusionXLPipelineIpex.from_pretrained(model_id, use_auth_token=True, low_cpu_mem_usage=True, use_safetensors = True, safety_checker = None)
    
    inference_step = args.step
     
    #print(pipe.unet)

    #get_host_memory()
    
    data_type = torch.bfloat16 if args.bf16 else torch.float32

    pipe.prepare_for_ipex(data_type, prompt, height=args.height, width=args.width, guidance_scale=0.0)

    #get_host_memory()
    
    generator = torch.Generator(device).manual_seed(4)
    with torch.no_grad(), torch.cpu.amp.autocast(enabled=args.bf16, dtype=torch.bfloat16):
        sum = 0
        images = []
        for i in range(args.repeat):
            print('')
            t1 = time.time()
            image = pipe(prompt, num_inference_steps=inference_step, height=args.height, width=args.width, guidance_scale=0.0, generator = generator)
            t2 = time.time()
          
            #print('SDXL-Turbo inference latency: {:.3f} sec'.format(t2-t1))
            #print('******************************')
            #print('')
            
            #get_host_memory()
            
            if i > 0 :
                sum += t2 - t1

            for j in range(args.batch):
                image.images[j].save("saved_pic" + str(j) + "ite" + str(i) + ".png")

                images.append(image.images[j])

        out = make_image_grid(images, rows = args.batch, cols = args.repeat) 
        post_fix = "bf16" if args.bf16 else "fp32"
        out.save("grid_" + post_fix  +".png")
             
    print("finish job! avg latency {0:4.4f}".format(sum / (args.repeat - 1)))

