import torch

from pipeline_stable_video_diffusion_inpainting import StableVideoDiffusionInpaintingPipeline
from diffusers.utils import load_image, export_to_video
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from PIL import Image
from scheduling_euler_discrete_training import EulerDiscreteSchedulerTraining
from diffusers import DiffusionPipeline

import numpy as np
def decode_latents(latents):
    
    latents = 1 / pipe.vae.config.scaling_factor * latents
    image = pipe.vae.decode(latents, return_dict=False,num_frames=1)[0]
    #image = pipe.vae.decode(latents, return_dict=False)
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    image = image.cpu().permute(0, 1, 2, 3).squeeze(0).float().numpy()
    
    return image



pipe = StableVideoDiffusionInpaintingPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid"
)

#pipe = DiffusionPipeline.from_pretrained(
#    "stabilityai/stable-diffusion-xl-base-1.0"
#)
pipe.scheduler = EulerDiscreteSchedulerTraining.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
# Load the conditioning image
image = load_image("shirt.png")
vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
mask_image= load_image("mask.png")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipe(
    image=image,
    motion_bucket_id=30,
    overlay_init_image=image,
    overlay_mask_image=mask_image,
    overlay_end=0.5,
    generator=torch.Generator().manual_seed(42),
).frames[0]

export_to_video(frames, "generated.mp4", fps=7)

with torch.no_grad():
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    
    #export_to_video(frames, "generated.mp4", fps=7)
    image = image_processor.preprocess(image)
    image = image.to(device="cuda")
    print("sigmas",pipe.scheduler.sigmas)
    latents = pipe.vae.encode(image).latent_dist.mode()
    noise = torch.randn_like(latents)
    num_inference_steps = 900
    strength = 1
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    
    t_start = max(num_inference_steps - init_timestep, 0)
    print("t_start = ",t_start)
    timesteps = pipe.scheduler.timesteps[t_start * 1 :]
    final_timestep = timesteps[:1]
    noised_latents = pipe.scheduler.add_noise(latents, noise, final_timestep)
    
    re_encoded_image = decode_latents(noised_latents)
    reshaped = np.transpose(re_encoded_image, (1, 2, 0))
    re_encoded_image_uint8 = (reshaped * 255).astype(np.uint8)

    print("decoded",re_encoded_image_uint8.shape)
    im = Image.fromarray(re_encoded_image_uint8)
    im.save("done.jpeg")
    
