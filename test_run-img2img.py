import torch

from pipeline_stable_video_diffusion_img2img import StableVideoDiffusionPipelineImg2Img
from diffusers.utils import load_image, export_to_video


pipe = StableVideoDiffusionPipelineImg2Img.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid"
)
pipe.enable_model_cpu_offload()

# Load the conditioning image
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipe(image, decode_chunk_size=8, generator=generator,end_frame=image).frames[0]

export_to_video(frames, "generated.mp4", fps=7)