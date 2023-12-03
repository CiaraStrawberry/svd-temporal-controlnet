import os, io, csv, math, random
import numpy as np
from einops import rearrange

import torch
from decord import VideoReader
import cv2

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from utils.util import zero_rank_print
#from torchvision.io import read_image
from PIL import Image
def pil_image_to_numpy(image):
    """Convert a PIL image to a NumPy array."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)

def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
    """Convert a NumPy image to a PyTorch tensor."""
    if images.ndim == 3:
        images = images[..., None]
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images.float() / 255


class WebVid10M(Dataset):
    def __init__(
            self,
            csv_path, video_folder,depth_folder,
            sample_size=256, sample_stride=4, sample_n_frames=14,
        ):
        zero_rank_print(f"loading annotations from {csv_path} ...")
        with open(csv_path, 'r') as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        random.shuffle(self.dataset)    
        self.video_folder    = video_folder
        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.depth_folder = depth_folder
        print("length",len(self.dataset))
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        print("sample size",sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    




    def center_crop(self,img):
        h, w = img.shape[-2:]  # Assuming img shape is [C, H, W] or [B, C, H, W]
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        return img[..., top:top+min_dim, left:left+min_dim]
        
    
    def get_batch(self, idx):
        def sort_frames(frame_name):
            return int(frame_name.split('_')[1].split('.')[0])
    
        while True:
            video_dict = self.dataset[idx]
            videoid, name, page_dir = video_dict['videoid'], video_dict['name'], video_dict['page_dir']
            preprocessed_dir = os.path.join(self.video_folder, videoid)
            depth_folder = os.path.join(self.depth_folder, videoid)
            if not os.path.exists(depth_folder):
                idx = random.randint(0, len(self.dataset) - 1)
                continue
    
            print(f"run idx = {videoid}")
    
            image_files = sorted(os.listdir(preprocessed_dir), key=sort_frames)
            total_frames = len(image_files)
            print(total_frames)
            if total_frames < 14:
                idx = random.randint(0, len(self.dataset) - 1)
                continue
            clip_length = min(total_frames, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx = random.randint(0, total_frames - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
    
            # Read and accumulate images in a NumPy array
            numpy_images = np.array([pil_image_to_numpy(Image.open(os.path.join(preprocessed_dir, image_files[int(i)]))) for i in batch_index])
    
            # Convert the NumPy array to a PyTorch tensor
            pixel_values = numpy_to_pt(numpy_images)
    
            # Similarly for depth frames
            #depth_folder = os.path.join(self.depth_folder, videoid)
            depth_files = sorted(os.listdir(depth_folder), key=sort_frames)[:14]
            numpy_depth_images = np.array([pil_image_to_numpy(Image.open(os.path.join(depth_folder, df))) for df in depth_files])
            depth_pixel_values = numpy_to_pt(numpy_depth_images)
    
            return pixel_values, depth_pixel_values, pixel_values[0]

        
        
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        #while True:
           # try:
        pixel_values, depth_pixel_values,first_frame = self.get_batch(idx)
           #     break
          #  except Exception as e:
          #      print(e)
          #      idx = random.randint(0, self.length - 1)

        pixel_values = self.pixel_transforms(pixel_values)
        sample = dict(pixel_values=pixel_values, depth_pixel_values=depth_pixel_values)
        return sample




if __name__ == "__main__":
    from utils.util import save_videos_grid

    dataset = WebVid10M(
        csv_path="/data/webvid/results_2M_train.csv",
        video_folder="/data/webvid/data/videos",
        sample_size=256,
        sample_stride=4, sample_n_frames=16,
        is_image=True,
    )
    import pdb
    pdb.set_trace()
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=16,)
    for idx, batch in enumerate(dataloader):
        print(batch["pixel_values"].shape, len(batch["text"]))
        # for i in range(batch["pixel_values"].shape[0]):
        #     save_videos_grid(batch["pixel_values"][i:i+1].permute(0,2,1,3,4), os.path.join(".", f"{idx}-{i}.mp4"), rescale=True)