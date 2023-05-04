# Custom PyTorch dataset for a HDF5 file with the following structure:
# semicol.h5/ ['ground_truths', 'images', 'metadata']. Each image has a corresponding ground truth.

import os
import torch
torch.manual_seed(1337)
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.io import read_image
import random
import time

class SemicolDataset(Dataset):
  def __init__(self, file_path, transform=None, normalise=True, norm_settings=(), is_train=True, load_onto_ram=False):
      self.file_path = file_path
      self.transform = transform
      self.is_train = is_train
      self.load_onto_ram = load_onto_ram
      self.normalise = normalise
      self.mean, self.std = norm_settings

      # Add normalization to the transform pipeline
      if self.normalise:
        self.normalise_transform = T.Compose([
            T.Normalize(self.mean, self.std)
        ])

      with h5py.File(self.file_path, 'r') as f:
          all_white = f['all_white'][:]
          contain_black = f['contain_black'][:]
          self.indices = np.where(np.logical_and(all_white == False, contain_black == False))[0].tolist()
          del all_white, contain_black

          if self.load_onto_ram: 
            # Time the loading of Macenko images
            start_time_macenko = time.time()
            print('Loading the required data from the dataset into memory... (This initial loading will take several minutes)')
            self.macenko = f['Macenko'][self.indices]
            end_time_macenko = time.time()
            print('Done. Time taken to load Macenko images: {:.2f} seconds'.format(end_time_macenko - start_time_macenko))

            # Time the loading of ground truths
            start_time_ground_truths = time.time()
            print('Now loading the required ground truths... (This initial loading will take several minutes)')
            self.ground_truths = f['ground_truths'][self.indices]
            end_time_ground_truths = time.time()
            print('Done. Time taken to load ground truths: {:.2f} seconds'.format(end_time_ground_truths - start_time_ground_truths))

  def __len__(self):
    return len(self.indices)

  def __getitem__(self, index):
      if self.load_onto_ram:
        image = torch.from_numpy(self.macenko[index])
        ground_truth = torch.from_numpy(self.ground_truths[index])
      else:
        with h5py.File(self.file_path, 'r') as f:
          image = torch.from_numpy(f['Macenko'][self.indices[index]])
          ground_truth = torch.from_numpy(f['ground_truths'][self.indices[index]])
      
      #get image
      image = image.type(torch.FloatTensor)
      image = torch.permute(image, [2, 0, 1])

      # apply normalisation transform
      if self.normalise:
        image = self.normalise_transform(image)

      if self.transform:
        ground_truth = self.rgb_to_label(ground_truth) # Convert RGB to label

      ground_truth = ground_truth.type(torch.LongTensor)

      # Apply the same augmentations to both image and mask
      if self.is_train:
        if random.random() < 0.5: # Apply random horizontal flip
            image = T.functional.hflip(image)
            ground_truth = T.functional.hflip(ground_truth)
        if random.random() < 0.5: # Apply random vertical flip
            image = T.functional.vflip(image)
            ground_truth = T.functional.vflip(ground_truth)
      
      # print("__getitem__: Unique values in image:", torch.unique(image))
      # print("__getitem__: Unique values in mask:", torch.unique(ground_truth))

      return image, ground_truth

  def rgb_to_label(self, label):
      label = torch.flip(label,[-1])
      """
      Supply our label masks as tensor in RGB format. 
      Replace pixels with specific numeric (class) values.
      """
      label_seg = torch.zeros(label.shape,dtype=torch.uint8)
      label_seg [torch.all(label == non_annotated_pixels          ,axis=-1)] = 0 # Non-annotated pixels (should be ignored)
      label_seg [torch.all(label == TUMOR_UKK                     ,axis=-1)] = 1 # Tumor tissue (epithelial), areas with clear high-grade intraepithelial neoplasia/adenoma might be included
      label_seg [torch.all(label == TUMOR_LMU                     ,axis=-1)] = 1 # Tumor tissue (epithelial), areas with clear high-grade intraepithelial neoplasia/adenoma might be included
      label_seg [torch.all(label == MUC                           ,axis=-1)] = 2 # Benign mucosa (colonic and ileal)
      label_seg [torch.all(label == TU_STROMA                     ,axis=-1)] = 3 # Tumoral stroma
      label_seg [torch.all(label == SUBMUC_OR_VESSEL_OR_ADVENT_LMU,axis=-1)] = 4 # Submucosal tissue, including large vessels | Blood vessels with muscular wall | Adventitial tissue / pericolic fat tissue, including large vessels
      label_seg [torch.all(label == SUBMUC_OR_VESSEL_OR_ADVENT_UKK,axis=-1)] = 4 # Submucosal tissue, including large vessels | Blood vessels with muscular wall | Adventitial tissue / pericolic fat tissue, including large vessels
      label_seg [torch.all(label == MUSC_PROP_MUSC_MUC_UKK        ,axis=-1)] = 5 # Muscularis propria | Muscularis mucosae
      label_seg [torch.all(label == MUSC_PROP_MUSC_MUC_LMU        ,axis=-1)] = 5 # Muscularis propria | Muscularis mucosae
      label_seg [torch.all(label == LYMPH_TIS                     ,axis=-1)] = 6 # Any forms of lymphatic tissue: lymphatic aggregates, lymph node tissue
      label_seg [torch.all(label == ULCUS_OR_NECROSIS             ,axis=-1)] = 7 # Ulceration (surface) | Necrotic debris
      label_seg [torch.all(label == MUCIN                         ,axis=-1)] = 8 # Acellular mucin lakes
      label_seg [torch.all(label == BLOOD                         ,axis=-1)] = 9 # Bleeding areas – only erythrocytes without any stromal or other tissue
      # label_seg [torch.all(label == BACK                          ,axis=-1)] = 10# Slide background
      label_seg [torch.all(label == BACK                          ,axis=-1)] = 0 # Slide background
      label_seg [torch.all(label == black_pixels                  ,axis=-1)] = 0 # Black. set to class 0 so that it will get ignored.
      label_seg = label_seg[:,:,0]  #reduce from rgb to just one channel. i.e. [12,12,12] -> [12])
      return label_seg

#! CLASS CODE 0
non_annotated_pixels = '#ffffff'.lstrip('#')
non_annotated_pixels = torch.tensor(tuple(int(non_annotated_pixels[i:i+2], 16) for i in (0, 2, 4))) # rgb(255, 255, 255)

#! CLASS CODE 1
TUMOR_UKK = '#ff00ff'.lstrip('#')
TUMOR_UKK = torch.tensor(tuple(int(TUMOR_UKK[i:i+2], 16) for i in (0, 2, 4))) # rgb(255, 0, 255) 
TUMOR_LMU = '#f139cf'.lstrip('#')
TUMOR_LMU = torch.tensor(tuple(int(TUMOR_LMU[i:i+2], 16) for i in (0, 2, 4))) # rgb(241, 57, 207) 

#! CLASS CODE 2
MUC = '#0a7cd5'.lstrip('#')
MUC = torch.tensor(tuple(int(MUC[i:i+2], 16) for i in (0, 2, 4))) # rgb(10, 124, 213) 

#! CLASS CODE 3
TU_STROMA = '#23eeab'.lstrip('#')
TU_STROMA = torch.tensor(tuple(int(TU_STROMA[i:i+2], 16) for i in (0, 2, 4))) # rgb(35, 238, 171) 

#! CLASS CODE 4
SUBMUC_OR_VESSEL_OR_ADVENT_UKK = '#00ffff'.lstrip('#')
SUBMUC_OR_VESSEL_OR_ADVENT_UKK = torch.tensor(tuple(int(SUBMUC_OR_VESSEL_OR_ADVENT_UKK[i:i+2], 16) for i in (0, 2, 4))) # rgb(0, 255, 255) 
SUBMUC_OR_VESSEL_OR_ADVENT_LMU = '#d89b86'.lstrip('#')
SUBMUC_OR_VESSEL_OR_ADVENT_LMU = torch.tensor(tuple(int(SUBMUC_OR_VESSEL_OR_ADVENT_LMU[i:i+2], 16) for i in (0, 2, 4))) # rgb(216, 155, 134)

#! CLASS CODE 5
MUSC_PROP_MUSC_MUC_UKK = '#ff6666'.lstrip('#')
MUSC_PROP_MUSC_MUC_UKK = torch.tensor(tuple(int(MUSC_PROP_MUSC_MUC_UKK[i:i+2], 16) for i in (0, 2, 4))) # rgb(255, 102, 102) 
MUSC_PROP_MUSC_MUC_LMU = '#b09d7f'.lstrip('#')
MUSC_PROP_MUSC_MUC_LMU = torch.tensor(tuple(int(MUSC_PROP_MUSC_MUC_LMU[i:i+2], 16) for i in (0, 2, 4))) # rgb(176, 157, 127) 

#! CLASS CODE 6
LYMPH_TIS = '#0afe06'.lstrip('#')
LYMPH_TIS = torch.tensor(tuple(int(LYMPH_TIS[i:i+2], 16) for i in (0, 2, 4))) # rgb(10, 254, 6) 

#! CLASS CODE 7
ULCUS_OR_NECROSIS = '#6ffc02'.lstrip('#')
ULCUS_OR_NECROSIS = torch.tensor(tuple(int(ULCUS_OR_NECROSIS[i:i+2], 16) for i in (0, 2, 4))) # rgb(111, 252, 2) 

#! CLASS CODE 8
MUCIN = '#c83de4'.lstrip('#')
MUCIN = torch.tensor(tuple(int(MUCIN[i:i+2], 16) for i in (0, 2, 4))) # rgb(200, 61, 228)

#! CLASS CODE 9
BLOOD = '#b74315'.lstrip('#')
BLOOD = torch.tensor(tuple(int(BLOOD[i:i+2], 16) for i in (0, 2, 4))) # rgb(183, 67, 21) 

#! CLASS CODE 10
BACK = '#47320a'.lstrip('#')
BACK = torch.tensor(tuple(int(BACK[i:i+2], 16) for i in (0, 2, 4))) # rgb(71, 50, 10) 

#! CLASS CODE 0
black_pixels = '#000000'.lstrip('#')
black_pixels = torch.tensor(tuple(int(black_pixels[i:i+2], 16) for i in (0, 2, 4))) # rgb(0, 0, 0) # add to class 0 to ignore.

# region <label_mapping with ratios/weights>
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
# endregion