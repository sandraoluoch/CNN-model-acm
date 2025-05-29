import os
import numpy as np
from tifffile import imread
from torch.utils.data import Dataset
from monai.data import image_reader

class ACM_Dataset(Dataset):
    """
    This is a custom dataset that loads in 3D grayscale images and prepares them for training, inference and evaluation 
    on a U-Net CNN model.
    """
    # defining the variables 
    def __init__(self, img_dir, mask_dir, transform=None):
        # getting the folder paths
        self.img_dir = img_dir 
        self.mask_dir = mask_dir 
        # setting up transform 
        self.transform = transform 

        # getting the image filenames that are inside the folders
        self.images = sorted([f for f in os.listdir(img_dir)])
        self.masks = sorted([f for f in os.listdir(mask_dir)])
    
    def __len__(self):
         return len(self.images)
    

    # getting the images from the file path   
    def __getitem__(self, idx):
            # iterate through list of image/mask filenames
            img_filename = self.images[idx]
            mask_filename = self.masks[idx]
            
            # joining folder filepath with image names
            img_filepath = os.path.join(self.img_dir, img_filename)
            mask_filepath = os.path.join(self.mask_dir, mask_filename)
        
            # storing images (raw images) and labels (segmentation masks) in a dictionary for MONAI
            sample = {"image": img_filepath, "label": mask_filepath}


            if self.transform is not None:
                 sample = self.transform(sample) 
                 
                 return sample


            
           
            

