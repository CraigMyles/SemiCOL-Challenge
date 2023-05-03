import argparse
from utils.tiatoolbox_staintools import *
import matplotlib.pyplot as plt
import cv2
import h5py
import numpy as np
from tqdm import tqdm

# Adding command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--h5-location', default='./notebooks/semicol.h5', help='Path to the h5 file')
parser.add_argument('--normalisation-type', choices=['Macenko', 'Vahadane'], default='Macenko', help='Stain normalization type')
parser.add_argument('--target-image', default='/data2/semicol/DATASET_TRAIN/01_MANUAL/DS_M_1/ukk_case_04/image/ukk_case_04 [d=2.16945,x=91117,y=78100,w=6508,h=6509].png', help='Path to the target image')
args = parser.parse_args()

print("RUNNING FOR ", args.normalisation_type)

f = h5py.File(args.h5_location, 'a')
print(f.keys())

## CHECK FOR IMAGES WITH ALL WHITE PIXELS

if 'all_white' not in f.keys():
    print("Creating list of all grount truth images which only contain white pixels - so that they can be excluded from use in training")
    f.create_dataset('all_white', (len(f['ground_truths']), 1), dtype='bool')

    count_z = 0
    count_y = 0
    query = np.array([[255, 255, 255]])
    for x in tqdm(range(len(f['ground_truths']))):
        unique_list = np.unique(f['ground_truths'][x].reshape(-1, f['ground_truths'][x].shape[-1]), axis=0, return_counts=False)
        if np.array_equal(unique_list, query):
            f['all_white'][x] = 1
            count_z += 1
        else:
            f['all_white'][x] = 0
            count_y += 1
            
    print("White count: ", count_z)
    print("Non-white count: ", count_y)


## CHECK FOR IMAGES CONTAINING BLACK PIXELS

if 'contain_black' not in f.keys():
    print("creating contain_black")
    f.create_dataset('contain_black', (len(f['ground_truths']), 1), dtype='bool')

    for x in tqdm(range(len(f['images']))):
        unique_list = np.unique(f['images'][x].reshape(-1, f['images'][x].shape[-1]), axis=0, return_counts=False)
        if np.any(query == unique_list):
            f['contain_black'][x] = 1
            count_z += 1
        else:
            f['contain_black'][x] = 0
            count_y += 1

    print("contain 0,0,0 count: ", count_z)
    print("normal patch count: ", count_y)

images = f['images']
image_count = len(images)
target_image = cv2.imread(args.target_image)

# Check if the selected normalization type is Macenko
if args.normalisation_type == 'Macenko':
    if 'Macenko' not in f.keys():
        f.create_dataset('Macenko', shape=images.shape, dtype=images.dtype)
        print("created Macenko set as does not exist already")
    else:
        print("Macenko set already exists")

# Check if the selected normalization type is Vahadane
if args.normalisation_type == 'Vahadane':
    if 'Vahadane' not in f.keys():
        f.create_dataset('Vahadane', shape=images.shape, dtype=images.dtype)
        print("created Vahadane set as does not exist already")
    else:
        print("Vahadane set already exists")


all_white = f['all_white'][:]
contain_black = f['contain_black'][:]
# Filter out indices where either 'all_white' or 'contains_black' is True
indices = np.where(np.logical_and(all_white == False, contain_black == False))[0].tolist()

# Initialize stain normalizers

if args.normalisation_type == 'Macenko':
    stain_normalizer = get_normalizer("Macenko")
if args.normalisation_type == 'Vahadane':
    stain_normalizer = get_normalizer("Vahadane")

stain_normalizer.fit(target_image)

print(f"total useful rows to evaluate: {len(indices)}")


for idx in tqdm(indices):
    sample = f['images'][idx]

    f[str(args.normalisation_type)][idx] = stain_normalizer.transform(sample)
