import os
import glob
from PIL import Image
from controlnet_aux.processor import Processor

# Set the directory paths
source_dir = "/data/webvid/webvid/preprocessed_videos2"  # Replace with your source directory
target_dir = "/data/webvid/webvid/lineart"  # Replace with your target directory

# Ensure target directory exists
os.makedirs(target_dir, exist_ok=True)
i = 0
def generate_canny_images(image_files, processor, target_subfolder):
    # Ensure target subfolder exists
    os.makedirs(target_subfolder, exist_ok=True)

    for image_file in image_files:
        # Load image
        img = Image.open(image_file).convert("RGB")
        img = img.resize((507, 290))

        # Process image to generate Canny edge image
        processed_image = processor(img, to_pil=True)

        # Save the processed image
        save_path = os.path.join(target_subfolder, os.path.basename(image_file))
        processed_image.save(save_path)

# Initialize ControlNet Aux processor for Canny edge detection
processor_id = 'lineart_realistic'
processor = Processor(processor_id)

# Process each image sequence in the source directory
for subdir in os.listdir(source_dir):
    image_subfolder = os.path.join(source_dir, subdir)
    target_subfolder = os.path.join(target_dir, subdir)

    if os.path.isdir(image_subfolder) and not os.path.isdir(target_subfolder):
        image_files = sorted(glob.glob(os.path.join(image_subfolder, '*.png')))  # Assuming images are .png

        # Check if there are exactly 14 images
        if len(image_files) > 10:
            i = i + 1
            generate_canny_images(image_files, processor, target_subfolder)
            print(f"lineart_realistic image {i} saved")
    else:
        i = i + 1
        print("skipping i")

print("Canny image generation completed.")
