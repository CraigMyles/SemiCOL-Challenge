import argparse
import os
import glob
import h5py
import tifffile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from unet import UNet
from tqdm import tqdm
import pandas as pd
from utils.tiatoolbox_staintools import *
import torchvision.transforms as T
import warnings

# # set target image
# target_image = "/home/cggm1/data/semicol/DATASET_TRAIN/01_MANUAL/DS_M_1/ukk_case_04/image/ukk_case_04 [d=2.16945,x=91117,y=78100,w=6508,h=6509].png"
# target_image = cv2.imread(target_image)

# #initailize stain normalizers
# macenko_stain_normalizer = get_normalizer("Macenko")
# macenko_stain_normalizer.fit(target_image)


class OMETIFFDataset(Dataset):
    def __init__(self, tiff_file, coords, macenko_stain_normalizer, patch_size=(256, 256)):
        self.tiff_file = tiff_file
        self.coords = coords
        self.patch_size = patch_size
        self.macenko_stain_normalizer = macenko_stain_normalizer  # Pass the normalizer to the dataset

        mean = [198.6229/255, 148.452/255, 196.1044/255]
        std = [35.2638/255, 55.9342/255, 41.948/255]
        self.normalize = T.Normalize(mean=mean, std=std)

        print(f"Opening {tiff_file}...")
        self.tif = tifffile.TiffFile(tiff_file)
        self.series = self.tif.series[0]
        self.page = self.series.levels[0]
        self.full_image = self.page.asarray()[:,:,:3] # load the full image into memory
        print(f"Full image loaded into memory.")

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        region = self.read_region_tifffile(self.coords[idx], self.patch_size)

        # Apply Macenko stain normalization
        # print("Applying stain normalization... on shape {}".format(region.shape))
        region = self.macenko_stain_normalizer.transform(region.transpose(1, 2, 0))
        # print("Stain normalization complete.")
        region = region.transpose(2, 0, 1)
        # print("region shape after stain normalization and transpose: {}".format(region.shape))

        region = region.astype(np.float32)  # Convert the data type to float32

        # Apply normalization
        region = self.normalize(torch.from_numpy(region))
        


        return region

    def read_region_tifffile(self, location, size):
        x, y = location
        width, height = size
        region = self.full_image[y:y + height, x:x + width, :]  # Read the region from the full_image
        region = np.transpose(region, (2, 0, 1))  # Transpose the region to have the shape (C, H, W)

        #macenko normalise now?

        return region



def main(args):
    coords_path = args.coords
    input_data_path = args.input_data

    #################################################################
    mean = [198.6229/255, 148.452/255, 196.1044/255]
    std = [35.2638/255, 55.9342/255, 41.948/255]
    normalize = T.Normalize(mean=mean, std=std)

    # set target image
    target_image = "./target_image.png"
    target_image = cv2.imread(target_image)

    #initailize stain normalizers
    macenko_stain_normalizer = get_normalizer("Macenko")
    macenko_stain_normalizer.fit(target_image)




    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # IS THIS CORRECT??????????????????????????????????????????????????
    #################################################################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Setting up model...")
    model = UNet(n_channels=3, n_classes=args.num_classes, bilinear=False)

    model = model.to(memory_format=torch.channels_last)

    print("Loading model weights...")
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    
    # Load existing DataFrame if it exists
    csv_filename = os.path.join("class_counts", "all_class_counts.csv")
    if os.path.exists(csv_filename):
        all_class_counts_df = pd.read_csv(csv_filename, index_col=0)
    else:
        all_class_counts_df = pd.DataFrame(columns=["filename"] + [f"class_{i}" for i in range(args.num_classes)])
        os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
        all_class_counts_df.to_csv(csv_filename)


    # Iterate through h5 files
    h5_files = glob.glob(os.path.join(coords_path, "*.h5"))
    for h5_file in tqdm(h5_files):
        # Initialize class_counts_dict for the current file
        class_counts_dict = {i: 0 for i in range(args.num_classes)}
        # Locate corresponding .ome.tiff file
        base_name = os.path.splitext(os.path.basename(h5_file))[0]
        tiff_file = os.path.join(input_data_path, base_name + ".ome.tiff")

        # Load coordinates from h5 file
        with h5py.File(h5_file, "r") as h5:
            coords = h5["coords"][:]

        # Create dataset and dataloader
        dataset = OMETIFFDataset(tiff_file, coords, macenko_stain_normalizer)
        dataloader = DataLoader(dataset, batch_size=128, num_workers=8, pin_memory=False)

        # Check if the row already exists for the current case
        if base_name in all_class_counts_df["filename"].values:
            print(f"Skipping {base_name} as it already exists in the CSV file.")
            continue

        # Pass patches to your model
        for patches in tqdm(dataloader):
            patches = patches.to(device=device)

            with torch.no_grad():

                mask_pred = model(patches)

                mask_pred_no_class0 = mask_pred.clone()
                class_0_indices = (mask_pred.argmax(dim=1) == 0)  # Find pixels assigned class 0

                # Set class 0 values to -inf for pixels assigned class 0
                mask_pred_no_class0[:, 0, :, :][class_0_indices] = float('-inf')

                # Get the highest class after excluding class 0 for those pixels
                second_best_class = mask_pred_no_class0.argmax(dim=1)

                # Replace class 0 with the second highest class in the best_class tensor
                best_class = mask_pred.argmax(dim=1)
                best_class[class_0_indices] = second_best_class[class_0_indices]

                # Update class counts
                class_counts = torch.bincount(best_class.view(-1)).tolist()
                for i, count in enumerate(class_counts):
                    class_counts_dict[i] += count
        
        # Convert the class_counts_dict to a list of counts and prepend the base_name
        class_counts_list = [base_name] + [class_counts_dict[i] for i in range(args.num_classes)]

        # Append the class counts list to the DataFrame
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=DeprecationWarning)
            warnings.simplefilter(action='ignore', category=FutureWarning) # append will be removed in pandas 2.0.0
            all_class_counts_df = all_class_counts_df.append(pd.Series(class_counts_list, index=all_class_counts_df.columns), ignore_index=True)

        # Save the aggregated DataFrame to a CSV file
        os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
        # Save the aggregated DataFrame to a CSV file
        with open(csv_filename, 'a') as f:
            all_class_counts_df[-1:].to_csv(f, header=False)

        # Cleanup: close the tiff file
        dataset.tif.close()




            # Process outputs as needed, e.g., save results, etc.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment .ome.tiff files in a folder structure.")
    parser.add_argument("--coords", default="/home/cggm1/semicol_dev/semicol_patch_locs", help="Path to coordinate input folder (.h5s)")
    parser.add_argument("--input_data", default="/home/cggm1/data/semicol/DATASET_VAL/DATASET_VAL/02_BX", help="Path to input folder (.ome.tiffs)")
    parser.add_argument("--model_path", default="/home/cggm1/data/semicol/checkpoints/unet/rebalanced/256/tqti7a48/checkpoint_epoch30.pth", help="Path to model file (.pth)")
    parser.add_argument("--num_classes", type=int, default=11, help="Number of classes.")
    args = parser.parse_args()
    main(args)