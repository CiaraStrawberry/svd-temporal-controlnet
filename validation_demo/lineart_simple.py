from PIL import Image
import os
import glob
from controlnet_aux.processor import Processor

# Set the directory paths
source_dir = "D:/gitprojects/sdv_controlnet/validation_demo/rgb"  # Source directory
target_dir = "D:/gitprojects/sdv_controlnet/validation_demo/lineart"  # Target directory

# Ensure target directory exists
os.makedirs(target_dir, exist_ok=True)

def generate_canny_images(image_files, processor, target_subfolder):
    # Ensure target subfolder exists
    os.makedirs(target_subfolder, exist_ok=True)

    for image_file in image_files:
        # Load image
        img = Image.open(image_file).convert("RGB")
        img = img.resize((512, 512))

        # Process image to generate Canny edge image
        processed_image = processor(img, to_pil=True)

        # Save the processed image
        save_path = os.path.join(target_subfolder, os.path.basename(image_file))
        processed_image.save(save_path)

# Initialize ControlNet Aux processor for Canny edge detection
processor_id = 'lineart_realistic'
processor = Processor(processor_id)

# Process images in the source directory
image_files = sorted(glob.glob(os.path.join(source_dir, '*.png')))  # Assuming images are .png

if len(image_files) > 10:
    generate_canny_images(image_files, processor, target_dir)
    print(f"lineart_realistic images saved in {target_dir}")
else:
    print("Not enough images to process.")
