from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionImageVariationPipeline,StableDiffusionInstructPix2PixPipeline
from diffusers.utils import logging
from PIL import Image, ImageOps
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
import torchvision.transforms as transforms
# from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
# strength = [0.1,0.15,0.2,0.25]
# guidance = [0.5,1,1.5,2]
class Augmentation(nn.Module):
    pipe = None 
    def __init__(self,strength=0.15,guidance=1,prompt=["simple quickdraw of the image "],diff_steps=200):
        super(Augmentation, self).__init__()
        model_id = "timbrooks/instruct-pix2pix"
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        # self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
        # self.pipe =  StableDiffusionImageVariationPipeline.from_pretrained("lambdalabs/sd-image-variations-diffusers",revision="v2.0").to("cuda:1")
        # self.pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
        # self.pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
        logging.disable_progress_bar()
        self.pipe.to('cuda')
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.safety_checker = None
        self.prompt = prompt
        self.strength =strength
        self.guidance_scale =guidance
        self.num_inference_steps= diff_steps
        self.transforms =transform = transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.ToTensor(),
                                            transforms.Lambda(lambda t: (t * 2) - 1)
                                        ])
    def forward(self, image):
        # L=[]
        # for i in range(3)
        # canvas = image.resize((512, 512), Image.BILINEAR)
        # canvas= [] 
        # for i in range(3):
        # for i in range(6):
        #      canvas.append(image)
        # canvas = torch.stack(canvas)
        # canvas.squeeze(1)
        # prompts = [self.prompt[0]+prompt]
        prompts = self.prompt
        with autocast(device_type='cuda', dtype=torch.float16):
                # with autocast(device_type='cuda', dtype=torch.float16):
                # print("here")
                out = self.pipe(prompt=prompts,image=image,image_guidance_scale=1.5, guidance_scale=7.5,num_inference_steps=10,num_images_per_prompt=4).images
                # out = self.pipe(image=canvas,guidance_scale=self.guidance_scale,num_inference_steps=self.num_inference_steps).images[0]
        # for i in range(3):
        #     out[i] = out[i].resize(image.size, Image.BILINEAR)
        # L.append(out)
        # print(len(out))
        return out
        
        
        
