# Custom PyTorch dataset for a semicol challenge
# Uses original images and take random crops of size x,y instead of hardcoded patches.

# DATASET FOR PATCHES LARGER THAN 256x256. (USES RANDOM CROP TO GRAB REQUESTED SIZE THEN DOWN-SAMPLES TO 256x256)

import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
# from torchvision.transforms import Resize
from torchvision.io import read_image
import random
import cv2
# from utils.stain_normalization import MacenkoStainNormalization

from utils.tiatoolbox_staintools import *



torch.manual_seed(1337)
class SemicolDataset(Dataset):
    def __init__(self, root_dir, patch_dim=1024, transform=None, normalise=True, norm_settings=(), is_train=True, load_onto_ram=False, downsize_to_256=False, n_classes=10):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.mask_paths = []

        self.is_train = is_train
        self.load_onto_ram = load_onto_ram
        self.normalise = normalise
        self.mean, self.std = norm_settings
        self.n_classes = n_classes

        self.downsize_to_256 = downsize_to_256
        # set target image
        self.target_image = "/home/cggm1/data/semicol/DATASET_TRAIN/01_MANUAL/DS_M_1/ukk_case_04/image/ukk_case_04 [d=2.16945,x=91117,y=78100,w=6508,h=6509].png"
        self.target_image = cv2.imread(self.target_image)
        self.macenko_stain_normalizer = get_normalizer("Macenko")
        self.macenko_stain_normalizer.fit(self.target_image)


        print(f"mean and std are: {self.mean}, {self.std}")
        # Add normalization to the transform pipeline
        if self.normalise:
            self.normalise_transform = T.Compose([
                T.Normalize(self.mean, self.std)
            ])

        self.patch_dim = (patch_dim, patch_dim)
        self.original_patch_dim = (3000, 3000)
        self.resize = T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC) # if we wanted to grab a larger patch but feed downsized (256x256) into model
        self.target_image = read_image('./target_image.png')
        self.stain_normalizer = MacenkoStainNormalization(self.target_image)

        # Set random seed for PyTorch
        torch.manual_seed(1337)

        # Recursively search for all .png files in the 'image' directories
        for dirpath, _, filenames in os.walk(root_dir):
            if 'image' in dirpath:
                image_paths = [os.path.join(dirpath, f) for f in filenames if f.endswith('.png')]
                self.image_paths.extend(image_paths)
                mask_paths = [os.path.join(root_dir, dirpath.replace('image', 'mask'), f.replace('.png', '-labelled.png')) for f in filenames if f.endswith('.png')]
                self.mask_paths.extend(mask_paths)

        self.image_paths.sort()
        self.mask_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def apply_augmentations(self, img, mask):
        # Apply random horizontal flip
        if random.random() < 0.5:
            img = T.functional.hflip(img)
            mask = T.functional.hflip(mask)

        # Apply random vertical flip
        if random.random() < 0.5:
            img = T.functional.vflip(img)
            mask = T.functional.vflip(mask)

        return img, mask

    def __getitem__(self, idx):
        # Load mask to memory
        mask_path = self.mask_paths[idx]
        mask = read_image(mask_path)

        if self.n_classes == 10:
            mask[mask == 10] = 0

        max_attempts = 50  # You can set this to a suitable value
        attempt = 0
        while attempt < max_attempts:
            # Get variables for random crop
            top, left, height, width = T.RandomCrop.get_params(torch.zeros(self.original_patch_dim), self.patch_dim)

            # Load mask into a crop function
            mask_crop = T.functional.crop(mask, top, left, height, width)

            # Check if mask contains only class 0
            if torch.sum(mask_crop) != 0:
                break
            attempt += 1
        else:
            # print(f"Too many attempts on mask: {mask_path}")
            # Handle the case when a suitable crop is not found within max_attempts
            # You can either return a default crop or handle this case in another way
            pass

        # Load image direct into a crop function
        img_path = self.image_paths[idx]
        image = T.functional.crop(read_image(img_path), top, left, height, width)

        # image = image.permute(1, 2, 0).numpy()  # Convert to HWC for stain normalization

        # # Apply stain normalization
        # image = self.macenko_stain_normalizer.transform(image)

        # # Convert back to torch tensor
        # image = torch.from_numpy(image).permute(2, 0, 1)

        # # Apply normalization
        if self.normalise:
            image = self.normalise_transform(image.to(torch.float32))


        # Apply the same augmentations to both image and mask
        image, mask_crop = self.apply_augmentations(image, mask_crop)

        # # Convert the Torch tensors to PIL images and save them to disk
        # image_pil = T.ToPILImage()(image)
        # image_pil.save(f'image_{idx}.png')

        # mask_pil = T.ToPILImage()(mask_crop)
        # mask_pil.save(f'mask_{idx}.png')

        # print("__getitem__: Unique values in image:", torch.unique(image))
        # print("__getitem__: Unique values in mask:", torch.unique(mask_crop))


        if self.downsize_to_256:
            # print(f"returning downsized image and mask of sizes {self.resize(image).shape} and {self.resize(mask_crop).shape}")
            return self.resize(image), mask_crop
        else:
            # print(f"__GETITEM__: returning image and mask of sizes {image.shape} and {mask_crop.shape}")
            return image, mask_crop



# label_mapping = {
#    0 : RGB(255, 255, 255), 34.571% = 0.34571, weight = 1-0.34571 = 0.65429
#    1 : RGB(255, 0, 255), 3.728% = 0.03728, weight = 1-0.03728 = 0.96272
#    1 : RGB(241, 57, 207), 2.476% = 0.02476, weight = 1-0.02476 = 0.97524
# ############ combined class 1 percentage: 6.204% = 0.06204, weight = 1-0.06204 = 0.93796
#    2 : RGB(10, 124, 213), 12.616% = 0.12616, weight = 1-0.12616 = 0.87384
#    3 : RGB(35, 238, 171), 5.362% = 0.05362, weight = 1-0.05362 = 0.94638
#    4 : RGB(0, 255, 255), 13.450% = 0.13450, weight = 1-0.13450 = 0.86550
#    4 : rgb(216, 155, 134), 19.810% = 0.19810, weight = 1-0.19810 = 0.80190
# ############ combined class 4 percentage: 33.260% = 0.33260, weight = 1-0.33260 = 0.66740
#    5 : rgb(255, 102, 102), 10.488% = 0.10488, weight = 1-0.10488 = 0.89512
#    5 : rgb(176, 157, 127), 7.596% = 0.07596, weight = 1-0.07596 = 0.92404
# ############ combined class 5 percentage: 18.084% = 0.18084, weight = 1-0.18084 = 0.81916
#    6 : rgb(10, 254, 6), 1.625% = 0.01625, weight = 1-0.01625 = 0.98375
#    7 : rgb(111, 252, 2), 2.925% = 0.02925, weight = 1-0.02925 = 0.97075
#    8 : rgb(200, 61, 228), 2.921% = 0.02921, weight = 1-0.02921 = 0.97079
#    9 : rgb(183, 67, 21), 1.871% = 0.01871, weight = 1-0.01871 = 0.98129
#   10 : rgb(71, 50, 10), 40.452% = 0.40452, weight = 1-0.40452 = 0.59548
#    0 : rgb(0, 0, 0), 0.000% = 0.00000
# }

# #! CLASS CODE 0
# non_annotated_pixels = '#ffffff'.lstrip('#')
# non_annotated_pixels = torch.tensor(tuple(int(non_annotated_pixels[i:i+2], 16) for i in (0, 2, 4))) # rgb(255, 255, 255)

# #! CLASS CODE 1
# TUMOR_UKK = '#ff00ff'.lstrip('#')
# TUMOR_UKK = torch.tensor(tuple(int(TUMOR_UKK[i:i+2], 16) for i in (0, 2, 4))) # rgb(255, 0, 255) 
# TUMOR_LMU = '#f139cf'.lstrip('#')
# TUMOR_LMU = torch.tensor(tuple(int(TUMOR_LMU[i:i+2], 16) for i in (0, 2, 4))) # rgb(241, 57, 207) 

# #! CLASS CODE 2
# MUC = '#0a7cd5'.lstrip('#')
# MUC = torch.tensor(tuple(int(MUC[i:i+2], 16) for i in (0, 2, 4))) # rgb(10, 124, 213) 

# #! CLASS CODE 3
# TU_STROMA = '#23eeab'.lstrip('#')
# TU_STROMA = torch.tensor(tuple(int(TU_STROMA[i:i+2], 16) for i in (0, 2, 4))) # rgb(35, 238, 171) 

# #! CLASS CODE 4
# SUBMUC_OR_VESSEL_OR_ADVENT_UKK = '#00ffff'.lstrip('#')
# SUBMUC_OR_VESSEL_OR_ADVENT_UKK = torch.tensor(tuple(int(SUBMUC_OR_VESSEL_OR_ADVENT_UKK[i:i+2], 16) for i in (0, 2, 4))) # rgb(0, 255, 255) 
# SUBMUC_OR_VESSEL_OR_ADVENT_LMU = '#d89b86'.lstrip('#')
# SUBMUC_OR_VESSEL_OR_ADVENT_LMU = torch.tensor(tuple(int(SUBMUC_OR_VESSEL_OR_ADVENT_LMU[i:i+2], 16) for i in (0, 2, 4))) # rgb(216, 155, 134)

# #! CLASS CODE 5
# MUSC_PROP_MUSC_MUC_UKK = '#ff6666'.lstrip('#')
# MUSC_PROP_MUSC_MUC_UKK = torch.tensor(tuple(int(MUSC_PROP_MUSC_MUC_UKK[i:i+2], 16) for i in (0, 2, 4))) # rgb(255, 102, 102) 
# MUSC_PROP_MUSC_MUC_LMU = '#b09d7f'.lstrip('#')
# MUSC_PROP_MUSC_MUC_LMU = torch.tensor(tuple(int(MUSC_PROP_MUSC_MUC_LMU[i:i+2], 16) for i in (0, 2, 4))) # rgb(176, 157, 127) 

# #! CLASS CODE 6
# LYMPH_TIS = '#0afe06'.lstrip('#')
# LYMPH_TIS = torch.tensor(tuple(int(LYMPH_TIS[i:i+2], 16) for i in (0, 2, 4))) # rgb(10, 254, 6) 

# #! CLASS CODE 7
# ULCUS_OR_NECROSIS = '#6ffc02'.lstrip('#')
# ULCUS_OR_NECROSIS = torch.tensor(tuple(int(ULCUS_OR_NECROSIS[i:i+2], 16) for i in (0, 2, 4))) # rgb(111, 252, 2) 

# #! CLASS CODE 8
# MUCIN = '#c83de4'.lstrip('#')
# MUCIN = torch.tensor(tuple(int(MUCIN[i:i+2], 16) for i in (0, 2, 4))) # rgb(200, 61, 228)

# #! CLASS CODE 9
# BLOOD = '#b74315'.lstrip('#')
# BLOOD = torch.tensor(tuple(int(BLOOD[i:i+2], 16) for i in (0, 2, 4))) # rgb(183, 67, 21) 

# #! CLASS CODE 10
# BACK = '#47320a'.lstrip('#')
# BACK = torch.tensor(tuple(int(BACK[i:i+2], 16) for i in (0, 2, 4))) # rgb(71, 50, 10) 

# #! CLASS CODE 0
# black_pixels = '#000000'.lstrip('#')
# black_pixels = torch.tensor(tuple(int(black_pixels[i:i+2], 16) for i in (0, 2, 4))) # rgb(0, 0, 0) # add to class 0 to ignore.

# def rgb_to_label(label):
#     label = torch.flip(label,[-1])
#     """
#     Supply our label masks as tensor in RGB format. 
#     Replace pixels with specific numeric (class) values.
#     """
#     label_seg = torch.zeros(label.shape,dtype=torch.uint8)
#     label_seg [torch.all(label == non_annotated_pixels          ,axis=-1)] = 0 # Non-annotated pixels (should be ignored)
    
#     label_seg [torch.all(label == TUMOR_UKK                     ,axis=-1)] = 1 # Tumor tissue (epithelial), areas with clear high-grade intraepithelial neoplasia/adenoma might be included
#     label_seg [torch.all(label == TUMOR_LMU                     ,axis=-1)] = 1 # Tumor tissue (epithelial), areas with clear high-grade intraepithelial neoplasia/adenoma might be included

#     label_seg [torch.all(label == MUC                           ,axis=-1)] = 2 # Benign mucosa (colonic and ileal)
#     label_seg [torch.all(label == TU_STROMA                     ,axis=-1)] = 3 # Tumoral stroma

#     label_seg [torch.all(label == SUBMUC_OR_VESSEL_OR_ADVENT_LMU,axis=-1)] = 4 # Submucosal tissue, including large vessels | Blood vessels with muscular wall | Adventitial tissue / pericolic fat tissue, including large vessels
#     label_seg [torch.all(label == SUBMUC_OR_VESSEL_OR_ADVENT_UKK,axis=-1)] = 4 # Submucosal tissue, including large vessels | Blood vessels with muscular wall | Adventitial tissue / pericolic fat tissue, including large vessels

#     label_seg [torch.all(label == MUSC_PROP_MUSC_MUC_UKK        ,axis=-1)] = 5 # Muscularis propria | Muscularis mucosae
#     label_seg [torch.all(label == MUSC_PROP_MUSC_MUC_LMU        ,axis=-1)] = 5 # Muscularis propria | Muscularis mucosae

#     label_seg [torch.all(label == LYMPH_TIS                     ,axis=-1)] = 6 # Any forms of lymphatic tissue: lymphatic aggregates, lymph node tissue
#     label_seg [torch.all(label == ULCUS_OR_NECROSIS             ,axis=-1)] = 7 # Ulceration (surface) | Necrotic debris
#     label_seg [torch.all(label == MUCIN                         ,axis=-1)] = 8 # Acellular mucin lakes
#     label_seg [torch.all(label == BLOOD                         ,axis=-1)] = 9 # Bleeding areas – only erythrocytes without any stromal or other tissue
#     label_seg [torch.all(label == BACK                          ,axis=-1)] = 10# Slide backgroun

#     label_seg [torch.all(label == black_pixels                  ,axis=-1)] = 0 # Black. set to class 0 so that it will get ignored.
    
#     label_seg = label_seg[:,:,0]  #reduce from rgb to just one channel. i.e. [12,12,12] -> [12])

#     return label_seg