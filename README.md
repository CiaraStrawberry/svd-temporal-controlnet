# Stable Video Diffusion Temporal Controlnet

## Overview
Introducing the Stable Video Diffusion Temporal Controlnet! This innovative tool merges the prowess of a controlnet style encoder with the robustness of the svd base. It's designed to enhance your video diffusion projects by providing precise temporal control.

## Setup
- **Controlnet Model:** First, obtain a svd temporal controlnet model folder from here and drop the controlnet folder into the repo folder: [temporal-controlnet-depth-svd-v1](https://huggingface.co/CiaraRowles/temporal-controlnet-depth-svd-v1)
- **Installation:** run `pip install -r requirements.txt`
- **Execution:** Run "run_inference.py".

## Demo
[Stable Video Diffusion Temporal Controlnet Repository](https://github.com/CiaraStrawberry/sdv_controlnet)

![combined_with_square_image_new_gif](https://github.com/CiaraStrawberry/sdv_controlnet/assets/13116982/055c8d3b-074e-4aeb-9ddc-70d12b5504d5)

## Notes
- **Focus on Central Object:** The system tends to extract motion features primarily from a central object and, occasionally, from the background. It's best to avoid overly complex motion or obscure objects.
- **Simplicity in Motion:** Stick to motions that svd can handle well without the controlnet. This ensures it will be able to apply the motion.

## Acknowledgements
- **Diffusers Team:** For the svd implementation.
- **Pixeli99:** For providing a practical svd training script: [SVD_Xtend](https://github.com/pixeli99/SVD_Xtend)
