import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from unet3plus.UNet_3Plus import UNet_3Plus, UNet_3Plus_DeepSup
import math
from unet import UNet
from vit.Vision_Transformer import SimpleViTSegmentation
from vit.crossformer import CrossFormerSegmentation
from torchvision.models.segmentation import deeplabv3_resnet50
import cv2
from utils.tiatoolbox_staintools import *

import torchvision.transforms as T
from torchvision.io import read_image

from PIL import Image

mean = [198.6229/255, 148.452/255, 196.1044/255]
std = [35.2638/255, 55.9342/255, 41.948/255] 

MODEL_INPUT_SIZE = 256
ORG_IMAGE_SIZE = 3000



# set target image
target_image = "./target_image.png"
target_image = cv2.imread(target_image)

#initailize stain normalizers
macenko_stain_normalizer = get_normalizer("Macenko")
macenko_stain_normalizer.fit(target_image)

def gaussian_weighted_distance_map(patch_size, sigma):
    y, x = np.mgrid[0:patch_size, 0:patch_size]

    # Compute the Euclidean distance to the nearest patch boundary
    distance = np.minimum(np.minimum(x, patch_size - x - 1), np.minimum(y, patch_size - y - 1))

    # Apply Gaussian function to the distance map
    window = np.exp(-(distance ** 2) / (2 * sigma ** 2))

    # Normalize the window function
    window /= np.sum(window)

    return torch.tensor(window, dtype=torch.float32)

def stitch_patches_with_adaptive_overlap(patches, image_shape, patch_size, stride, device, sigma=20, num_classes=11):
    image_shape = (image_shape[2], image_shape[0], image_shape[1])
    count_matrix = torch.zeros((num_classes, image_shape[1], image_shape[2])).to(device)
    stitched_image = torch.zeros((num_classes, image_shape[1], image_shape[2])).to(device)
    window = gaussian_weighted_distance_map(patch_size, sigma).to(device)

    num_vertical_patches = int(np.ceil((image_shape[1] - patch_size) / stride)) + 1
    num_horizontal_patches = int(np.ceil((image_shape[2] - patch_size) / stride)) + 1

    idx = 0
    for i in range(num_vertical_patches):
        for j in range(num_horizontal_patches):
            if idx < len(patches):
                patch_height = min(patch_size, image_shape[1] - i * stride)
                patch_width = min(patch_size, image_shape[2] - j * stride)
                patch = patches[idx].to(device)[:, :patch_height, :patch_width] * window[:patch_height, :patch_width].unsqueeze(0)


                stitched_image[:, i * stride:i * stride + patch_height, j * stride:j * stride + patch_width] += patch
                count_matrix[:, i * stride:i * stride + patch_height, j * stride:j * stride + patch_width] += window[:patch_height, :patch_width].unsqueeze(0)
                idx += 1

    stitched_image /= count_matrix
    return stitched_image

def extract_patches(image, patch_size, stride):

    image = image.transpose(2, 0, 1)
    patches = []
    num_vertical_patches = int(np.ceil((image.shape[1] - patch_size) / stride)) + 1
    num_horizontal_patches = int(np.ceil((image.shape[2] - patch_size) / stride)) + 1

    #print('num_vertical_patches: ', num_vertical_patches)
    #print('num_horizontal_patches: ', num_horizontal_patches)

    # print(
    #     f"Inside extract_patches: \n image.shape: {image.shape} \n patch_size: {patch_size} \n stride: {stride} \n num_vertical_patches: {num_vertical_patches} \n num_horizontal_patches: {num_horizontal_patches}"
    # )
    
    for i in range(num_vertical_patches):
        for j in range(num_horizontal_patches):
            y_start = i * stride
            y_end = y_start + patch_size
            x_start = j * stride
            x_end = x_start + patch_size

            patch = image[:, y_start:y_end, x_start:x_end]
            patches.append(patch)
    return patches


def hanning_window(patch_size):
    window_1d = np.hanning(patch_size)
    window_2d = np.outer(window_1d, window_1d)
    return torch.tensor(window_2d, dtype=torch.float32)

def stitch_patches_with_overlap(patches, image_shape, patch_size, stride, device, num_classes=11):
    # image_shape = image_shape.permute(2, 0, 1)
    image_shape = (image_shape[2], image_shape[0], image_shape[1])

    count_matrix = torch.zeros((num_classes, image_shape[1], image_shape[2])).to(device)
    stitched_image = torch.zeros((num_classes, image_shape[1], image_shape[2])).to(device)
    window = hanning_window(patch_size).to(device)


    num_vertical_patches = int(np.ceil((image_shape[1] - patch_size) / stride)) + 1
    num_horizontal_patches = int(np.ceil((image_shape[2] - patch_size) / stride)) + 1
    print('num_vertical_patches: ', num_vertical_patches)
    print('num_horizontal_patches: ', num_horizontal_patches)

    idx = 0
    for i in range(num_vertical_patches):
        for j in range(num_horizontal_patches):
            if idx < len(patches):
                patch_height = min(patch_size, image_shape[1] - i * stride)
                patch_width = min(patch_size, image_shape[2] - j * stride)

                

                # print(f"Window shape {window[:patch_height, :patch_width].unsqueeze(0).shape}")
                # print(f"Patch height {patch_height}, Patch width {patch_width}")
                # print(f"Patch shape {patches[idx].to(device)[:11, :patch_height, :patch_width].shape}")

                # For some reason, segformer does not work properly and causes an issue with the patch shape
                # So we have to use the following line instead


                # patch = patches[idx].to(device)[:, :patch_height, :patch_width] * window[:patch_height, :patch_width].unsqueeze(0)
                patch = patches[idx].to(device)[:num_classes, :patch_height, :patch_width] * window[:patch_height, :patch_width].unsqueeze(0)

                # print(f"Patch shape {patch.shape}")
                # print(f"Window shape {window[:patch_height, :patch_width].unsqueeze(0).shape}")
                # print(f"Stitched image shape {stitched_image[:, i * stride:i * stride + patch_height, j * stride:j * stride + patch_width].shape}")
                # print(f"Count matrix shape {count_matrix[:, i * stride:i * stride + patch_height, j * stride:j * stride + patch_width].shape}")
                


                stitched_image[:, i * stride:i * stride + patch_height, j * stride:j * stride + patch_width] += patch
                count_matrix[:, i * stride:i * stride + patch_height, j * stride:j * stride + patch_width] += window[:patch_height, :patch_width].unsqueeze(0)
                idx += 1

    stitched_image /= count_matrix
    return stitched_image

class PatchesDataset(Dataset):
    def __init__(self, patches):
        self.patches = patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx]

def segment_png(png_file_path, architecture, model, device, batch_size=128, overlap=128, num_classes=11, normalise=True):
    mean = [198.6229/255, 148.452/255, 196.1044/255]
    std = [35.2638/255, 55.9342/255, 41.948/255]
    normalize = T.Normalize(mean=mean, std=std)

    image = cv2.imread(png_file_path)

    orig_height, orig_width = (ORG_IMAGE_SIZE,ORG_IMAGE_SIZE)

    image = macenko_stain_normalizer.transform(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # IS THIS CORRECT??????????????????????????????????????????????????

    overlap = overlap
    patch_size = MODEL_INPUT_SIZE
    stride = patch_size - overlap
    print(
        f"Patch size: {patch_size}, Stride: {stride}, Overlap: {overlap}"
    )

    # pad_height = (orig_height // MODEL_INPUT_SIZE + 1) * MODEL_INPUT_SIZE
    # pad_width = (orig_width // MODEL_INPUT_SIZE + 1) * MODEL_INPUT_SIZE
    # pad_top = (pad_height - orig_height) // 2
    # pad_bottom = pad_height - orig_height - pad_top
    # pad_left = (pad_width - orig_width) // 2
    # pad_right = pad_width - orig_width - pad_left

    # Calculate the padding dimensions
    padded_height = ((orig_height - patch_size) // stride + 1) * stride + patch_size
    padded_width = ((orig_width - patch_size) // stride + 1) * stride + patch_size
    pad_top = (padded_height - orig_height) // 2
    pad_bottom = padded_height - orig_height - pad_top
    pad_left = (padded_width - orig_width) // 2
    pad_right = padded_width - orig_width - pad_left

    # Pad the image to make it divisible into patches of size 256x256
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    image = F.pad(image, padding, mode='reflect')

    image = image.numpy().transpose(1, 2, 0)
    image = np.clip(image, 0, 255).astype(np.uint8)
    image_pil = Image.fromarray(image)


    print(
        f"Image shape: {image.shape}, Patch size: {patch_size}, Stride: {stride}, Overlap: {overlap}"
    )

    patches = extract_patches(image, patch_size, stride)

    patches_dataset = PatchesDataset(patches)
    dataloader = DataLoader(patches_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    predicted_patches = []
    for batch_patches in tqdm(dataloader, desc="Processing patches for: " + png_file_path.split("/")[-1] + ""):
        batch = torch.stack(tuple(batch_patches)).to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

        if normalise:
            batch = normalize(batch)

        with torch.no_grad():
            with autocast():
                # print(f"About to make pred on batch shape {batch.shape}")

                mask_pred = model(batch)

                try:
                    if 'out' in mask_pred:
                        mask_pred = mask_pred['out']
                except:
                    pass

                if architecture == 'unet_3plus_deepsup_cgm' or architecture == 'unet_3plus_deepsup':
                    # combine the 5 outputs and average them to get the final output
                    mask_pred = torch.mean(torch.stack(mask_pred), dim=0)

                if mask_pred.shape[2] != MODEL_INPUT_SIZE:
                    mask_pred = F.interpolate(mask_pred, size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), mode='nearest')

                mask_pred_no_class0 = mask_pred.clone()
                class_0_indices = (mask_pred.argmax(dim=1) == 0)  # Find pixels assigned class 0

                # Set class 0 values to -inf for pixels assigned class 0
                mask_pred_no_class0[:, 0, :, :][class_0_indices] = float('-inf')

                # Get the highest class after excluding class 0 for those pixels
                second_best_class = mask_pred_no_class0.argmax(dim=1)

                # Replace class 0 with the second highest class in the best_class tensor
                best_class = mask_pred.argmax(dim=1)
                best_class[class_0_indices] = second_best_class[class_0_indices]

                mask_pred_one_hot = F.one_hot(best_class, num_classes).permute(0, 3, 1, 2).float()
                # print(f"Mask pred shape {mask_pred_one_hot.shape}")
                predicted_patches.extend(mask_pred_one_hot.cpu())

        # Clear GPU cache
        torch.cuda.empty_cache()

    # for each predicted_patches, print their shape
    # for i in range(len(predicted_patches)):
    #     print(f"Predicted patch {i} shape {predicted_patches[i].shape}")


    if args.mode == 'overlap':
        stitched_image = stitch_patches_with_overlap(predicted_patches, image.shape, patch_size, stride, device, num_classes)
    elif args.mode == 'adaptive':
        stitched_image = stitch_patches_with_adaptive_overlap(predicted_patches, image.shape, patch_size, stride, device, num_classes)
    else:
        raise ValueError('Invalid stitching mode')

    # Crop the stitched image to remove the padding
    cropped_image = stitched_image[:, pad_top:pad_top+orig_height, pad_left:pad_left+orig_width]

    output_image = cropped_image.argmax(dim=0).float().cpu().numpy()
    output_image = Image.fromarray(output_image.astype(np.uint8))

    palette = [255, 255, 255, 255, 0, 255, 10, 124, 213, 35, 238, 171, 0, 255, 255, 255, 102, 102, 10, 254, 6, 111, 252, 2, 200, 61, 228, 183, 67, 21, 71, 50, 10]
    output_image.putpalette(palette)
    output_image = output_image.convert('P')
    #print('out segment_png. PIL image size: ', output_image.size)
    return output_image

def process_files(input_folder, output_folder, architecture, model_path, mode, batch_size, overlap, num_classes, normalise):
    # Create the output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    NUM_CLASSES = num_classes

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #'unet','vit', 'crossformer', 'deeplabv3', 'deeplabv3_from_scratch'
    if args.architecture == 'unet':
      model = UNet(n_channels=3, n_classes=NUM_CLASSES, bilinear=False)

    if args.architecture == 'unet3plus':
      model = UNet_3Plus(
        in_channels=3,
        n_classes=NUM_CLASSES,
        feature_scale=4,
        is_deconv=True,
        is_batchnorm=True)
    
    if args.architecture == 'unet_3plus_deepsup':
      model = UNet_3Plus_DeepSup(
        in_channels=3,
        n_classes=NUM_CLASSES,
        feature_scale=4,
        is_deconv=True,
        is_batchnorm=True)

    if architecture == 'vit':
      model = SimpleViTSegmentation(
          channels=3,
          image_size=args.patch_dim,
          patch_size=32,
          num_classes=NUM_CLASSES,  # Change this to the number of segmentation classes you have
          dim=1024, # 512, 768, 1024, 2048?
          depth=6,
          heads=16,
          mlp_dim=2048
      )
    
    if architecture == 'crossformer':
      model = CrossFormerSegmentation(
          num_classes=NUM_CLASSES,             # number of segmentation classes
          dim=(64, 128, 256, 512),             # dimension at each stage
          depth=(2, 2, 8, 2),                  # depth of transformer at each stage
          global_window_size=(8, 4, 2, 1),     # global window sizes at each stage
          local_window_size=8,                 # local window size (held constant at 7 for all stages in the paper)
      )

    if architecture == 'deeplabv3':
      model = deeplabv3_resnet50(num_classes=NUM_CLASSES, pretrained_backbone=True)

    if architecture == 'deeplabv3_from_scratch':
      model = deeplabv3_resnet50(num_classes=NUM_CLASSES, pretrained_backbone=False)


    model = model.to(memory_format=torch.channels_last)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # Iterate through the input folder structure
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".png"):
                input_file_path = os.path.join(root, file)

                # Create mirrored output folder structure
                relative_path = os.path.relpath(root, input_folder)
                split_path = os.path.split(relative_path)
                if split_path[1] == "image":
                    modified_relative_path = split_path[0]
                else:
                    modified_relative_path = relative_path
                output_subfolder = os.path.join(output_folder, modified_relative_path)
                Path(output_subfolder).mkdir(parents=True, exist_ok=True)

                # Check if the segmented image already exists
                output_file_path = os.path.join(output_subfolder, file)
                if os.path.exists(output_file_path):
                    #print(f"Segmented image already exists: {output_file_path}")
                    continue # to next file

                # Apply segmentation pipeline
                segmented_png = segment_png(input_file_path, architecture, model, device, batch_size=batch_size, overlap=overlap, num_classes=NUM_CLASSES)

                # Save segmented output
                segmented_png.save(output_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment .PNG files in a folder structure.")
    parser.add_argument("--input", default="/data2/semicol/DATASET_VAL/01_MANUAL", help="Path to input folder")
    parser.add_argument("--output", default="/data2/semicol/DATASET_VAL_OUTPUT/01_MANUAL", help="Path to output folder")
    parser.add_argument('--architecture', '-a', metavar='ARCH', type=str, help='Architecture. i.e U-Net, Vision Transformer, CrossFormer, DeepLab v3', default='unet', choices=['unet','vit', 'crossformer', 'deeplabv3', 'deeplabv3_from_scratch', 'unet3plus', 'unet_3plus_deepsup'])
    # parser.add_argument("--model_path", default="/home/cggm1/Documents/GitHub/semicol_dev/checkpoints/256x256/xq0d3jf5_checkpoint_epoch18.pth", help="Path to trained model")
    parser.add_argument("--model_path", default="/app/SemiCOL-Challenge/trained_models/g4jx4n9k/checkpoint_epoch37.pth", help="Path to trained model")
    # mode, stitch_patches_with_overlap or stitch_patches_with_adaptive_overlap, default is stitch_patches_with_overlap
    parser.add_argument("--mode", default="overlap", help="Mode to stitch patches. Choose between 'overlap' and 'adaptive_overlap'.")
    parser.add_argument("--overlap", type=int, default=128, help="Overlap between patches.")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes.")
    parser.add_argument("--batch_size", type=int, default=128, help="Number of classes.")
    parser.add_argument('--normalise', default=False, type=lambda s: s.lower() == 'true', help='Normalise images (mean & std.), [true or false]')

    args = parser.parse_args()
    print("COMMENCING SEGMENTATION FOR SEMICOL CHALLENGE TASK 1")
    process_files(args.input, args.output, args.architecture, args.model_path, args.mode, args.batch_size, args.overlap, args.num_classes, args.normalise)