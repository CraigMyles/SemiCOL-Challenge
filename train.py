#Based on https://github.com/milesial/Pytorch-UNet
import argparse
import logging
import os
import random
import sys
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import traceback

from torchmetrics.classification import JaccardIndex
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

torch.manual_seed(1337)
random.seed(1337)
numpy.random.seed(1337)

import matplotlib.pyplot as plt

from PIL import Image
import string

import wandb
from evaluate_crop import evaluate
from unet import UNet
from unet3plus.UNet_3Plus import UNet_3Plus, UNet_3Plus_DeepSup, UNet_3Plus_DeepSup_CGM
from vit.Vision_Transformer import SimpleViTSegmentation
from vit.crossformer import CrossFormerSegmentation
from utils.dice_score import dice_loss
from torchvision.models.segmentation import deeplabv3_resnet50

from utils.semicol_dataset import SemicolDataset as SemicolDataset
from utils.semicol_dataset_crop import SemicolDataset as SemicolDatasetCrop

label_mapping = {
    0 : (255, 255, 255), # shouldnt really be a class
    1 : (255, 0, 255),
    2 : (10, 124, 213),
    3 : (35, 238, 171),
    4 : (0, 255, 255),
    5 : (255, 102, 102),
    6 : (10, 254, 6),
    7 : (111, 252, 2),
    8 : (200, 61, 228),
    9 : (183, 67, 21),
    10 : (71, 50, 10)
    # 11 : (0, 0, 0) # shouldnt really be a class
}

class_names = [
    "unannotated",
    "TUMOR",
    "MUC",
    "TU_STROMA",
    "SUBMUC_OR_VESSEL_OR_ADVENT",
    "MUSC_PROP_OR_MUSC_MUC",
    "LYMPH_TIS",
    "ULCUS_OR_NECROSIS",
    "MUCIN",
    "BLOOD",
    "BACK"
]

# Everything gets 0.1, while tumour get 0.2 (class 0 gets 0.0 since we want to ignore it)
challenge_weights = [ 
    0.0,
    0.2,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0
]

rebalanced_weights = [ # 1 - (percentages of frequency i.e. class 1 frequency was 0.06204)
    0, # just set this to 0.
    0.93796*2, # 1
    0.87384, # 2
    0.94638, # 3
    0.66740, # 4
    0.81916, # 5
    0.98375, # 6
    0.97075, # 7 # May want to *2 or *1.5 since this is helpful in detecting tumour.
    0.97079, # 8
    0.98129, # 9
    0.59548/10 # 10 not used in evaluation metric.
]

rebalanced_weights_extreme = [ # 1 - (percentages of frequency i.e. class 1 frequency was 0.06204)
    0, # just set this to 0.
    0.93796*4, # 1
    0.87384*1, # 2
    0.94638*2, # 3
    0.66740*0.8, # 4
    0.81916*1, # 5
    0.98375*1, # 6
    0.97075*2.5, # 7 # May want to *2 or *1.5 since this is helpful in detecting tumour.
    0.97079*1, # 8
    0.98129*1, # 9
    0.59548/10 # 10 not used in evaluation metric.
]

normalised_weights = [ #Inverse frequency with tumour doubled (then re-normalised...)
    0.0000, 0.0681, 0.0334, 0.0788, 0.0127, 0.0234, 0.2589, 0.1443, 0.1443, 0.2257, 0.0104
]

# def denormalize(tensor, mean, std):
#     for t, m, s in zip(tensor, mean, std):
#         t.mul_(s).add_(m)
#     return tensor


# def denormalize(tensor, mean, std):
#     mean = torch.tensor(mean).view(1, -1, 1, 1)
#     std = torch.tensor(std).view(1, -1, 1, 1)
#     return tensor * std + mean


def label2rgb(label):
    # Map each pixel value to its corresponding RGB color
    mapped_pixels = numpy.zeros((args.patch_dim, args.patch_dim, 3))
    for i in range(args.patch_dim):
        for j in range(args.patch_dim):
            pixel_value = int(label[i][j])
            mapped_pixels[i][j] = label_mapping[pixel_value]
    return mapped_pixels.astype(numpy.uint8)

def train_model(
        data_path: str,
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 1,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        class_names: list = class_names,
        weights: str = 'challenge',
        index_to_ignore: int = -100,
        patch_dim: int = 1024,
        loaded_model: str = None,
        architecture: str = 'unet',
        load_onto_ram: bool = False,
        optim_type: str = 'rmsprop',
        normalise: bool = True,
        class_reduce: bool = False,
):

    # h5 mean and std
    semicol_mean = [198.6229/255, 148.452/255, 196.1044/255] # Values pre-determined through normalisation_calc.py 
    semicol_std = [35.2638/255, 55.9342/255, 41.948/255] # Values pre-determined through normalisation_calc.py

    deeplabv3_mean = [0.485, 0.456, 0.406] # from https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
    deeplabv3_std = [0.229, 0.224, 0.225] # from https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/

    #3000x3000 dataset mean and std
    orig_mean = [0.86917771, 0.76583144, 0.85458537]
    orig_std = [0.11682992, 0.1986467, 0.12049045]

    # if using pre-trained deeplabv3, we need to normalise the input images using their mean and std.
    # otherwise we use the mean and std of the semicol dataset. (even for the untrained deeplabv3)

    if (architecture == 'deeplabv3'):
        print('Using deeplabv3 pretrained mean and std...')
        norm_settings = (deeplabv3_mean, deeplabv3_std)
    elif patch_dim > 256:
        print('Using original dataset mean and std...')
        norm_settings = (orig_mean, orig_std)
    else:
        print('Using h5 semicol dataset mean and std...')
        norm_settings = (semicol_mean, semicol_std)

    # 1. Create dataset
    if patch_dim == 256:
        try:
            dataset = SemicolDataset(data_path, 
                                            transform=True, 
                                            normalise=normalise, 
                                            norm_settings=norm_settings, 
                                            is_train=True, 
                                            load_onto_ram=load_onto_ram)
        except (AssertionError, RuntimeError, IndexError):
            print('Error: Dataset not found.')
    else:
        path_to_dataset_root = '/home/cggm1/data/semicol/DATASET_TRAIN/01_MANUAL/' # Change this to 01_MANUAL location for image sizes > 256x256
        print('RANDOM CROP DATASET LOADING...')
        dataset = SemicolDatasetCrop(path_to_dataset_root, 
                                            patch_dim=patch_dim, 
                                            transform=False, 
                                            normalise=normalise, 
                                            norm_settings=norm_settings, 
                                            is_train=True, 
                                            load_onto_ram=load_onto_ram, 
                                            downsize_to_256=False, 
                                            n_classes=n_classes)
        print('RANDOM CROP DATASET LOADED.')


    # dir checkpoint = dir checkpoint / architecture / weights / patch_dim
    dir_checkpoint = Path('/home/cggm1/data/semicol/checkpoints/')
    dir_checkpoint = dir_checkpoint / architecture / weights / str(patch_dim)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    val_set.dataset.is_train = False

    # 3. Create data loaders
    num_workers = 16*torch.cuda.device_count() if torch.cuda.is_available() else os.cpu_count()
    if load_onto_ram: num_workers = 0

    pin_memory = load_onto_ram
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    wandb_projects = {'unet': 'U-Net', 'unet3plus': 'U-Net', 'unet_3plus_deepsup': 'U-Net', 'unet_3plus_deepsup_cgm': 'U-Net', 'vit': 'ViT', 'crossformer': 'ViT', 'crossformer_larger': 'ViT', 'deeplabv3': 'ViT', 'deeplabv3_from_scratch': 'ViT'}
    project_name = wandb_projects[args.architecture]
    # if args.load_model, run experiment with same tags as previous run.
    if loaded_model:
        try:
            wandb_id = loaded_model.split('/')[-2]
            print(f'Resuming from run id: {wandb_id}')
            experiment = wandb.init(project=project_name, resume='must', id=str(wandb_id), anonymous='never', tags=[f"{args.architecture}", f"{patch_dim}x{patch_dim}", "Macenko", "Augmentation"])
        except:
            print('Error: Could not resume run, generating new run.')
            experiment = wandb.init(project=project_name, resume='allow', anonymous='never', tags=[f"{args.architecture}", f"{patch_dim}x{patch_dim}", "Macenko", "Augmentation"])
    else:
        experiment = wandb.init(project=project_name, resume='allow', anonymous='never', tags=[f"{args.architecture}", f"{patch_dim}x{patch_dim}", "Macenko", "Augmentation"])

    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp, weights=weights, patch_dim=patch_dim, optim_type=optim_type), allow_val_change=True
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
        weights:         {weights}
        ignore_index:    {index_to_ignore}
        patch_dim:       {patch_dim}
        num_workers:     {num_workers}
        pin_memory:      {pin_memory}
        architecture:    {args.architecture}
        wandb group:     {project_name}
        optim_type:      {optim_type}
        normalise:       {normalise}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    if optim_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=learning_rate, weight_decay=weight_decay)
    elif optim_type.lower() == 'rmsprop':
      optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif optim_type.lower() == 'sgd':
      optimizer = optim.SGD(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    else:
      raise ValueError(f'Invalid optim_type: {optim_type}, must be one of (adam, rmsprop, sgd).')

    

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    weights_map = {'challenge': challenge_weights, 'normalised': normalised_weights, 'rebalanced': rebalanced_weights, 'rebalanced_extreme': rebalanced_weights_extreme}
    weights = weights_map[args.weights]

    if n_classes == 10 and len(weights)==11:
        weights = weights[:-1] # remove background from class as it's merged with unannotated so it doesnt get predicted.
        class_names = class_names[:-1] # same as above.
        print(f"n_classes: {n_classes}")
        print(f"Weights len: {len(weights)}, \n weights: {weights}")

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device), ignore_index=index_to_ignore) 
    global_step = 0

    # 4.5 Define jaccard metrics
    jaccard_metric = JaccardIndex(task='multiclass', num_classes=n_classes, ignore_index=None).to(device)
    jaccard_metric_ignore_0 = JaccardIndex(task='multiclass', num_classes=n_classes, ignore_index=0).to(device)

    # 5. Begin training
    if isinstance(model, torch.nn.DataParallel): model = model.module
    model.train()
    for epoch in range(1, epochs + 1):
        
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:

                images, true_masks = batch[0], batch[1]

                if class_reduce:
                    if (true_masks == 0).all(): # if the unique values in true_masks are only 0, continune
                        continue
                    elif torch.equal(torch.unique(true_masks).sort()[0], torch.tensor([0, 4])): # if [0,4], 10% chance of continue (skip).
                        if torch.rand(1) < 0.1:  # 10% probability
                            continue
                    # else:
                    # print(f"torch.unique(true_masks): {torch.unique(true_masks)}")
                    # print(f"continuing with batch")
                    # pass

                # print(f"images shape: {images.shape}")
                # print(f"images[0] shape: {images[0].shape}")

                # print(
                #     f"IN TRAINING LOOP:\n GOT images shape: {images.shape}, true_masks shape: {true_masks.shape}"
                # )

                # # (Check that the images are loaded correctly)
                # print(f"images shape: {images.shape}")
                # print(f"true_masks shape: {true_masks.shape}")


                assert images.shape[1] == 3, \
                    f'Network has been defined with {3} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.squeeze(1).to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    # print(f"About to run model on images: {images.shape}")
                    masks_pred = model(images)
                    
                    # print(f"Ran model on image. The output shape is masks_pred: {masks_pred.shape}")

                    try:
                      if 'out' in masks_pred:
                          masks_pred = masks_pred['out']
                    except:
                      pass

                    if args.architecture == 'unet_3plus_deepsup_cgm' or args.architecture == 'unet_3plus_deepsup':
                        # combine the 5 outputs and average them to get the final output
                        masks_pred = torch.mean(torch.stack(masks_pred), dim=0)

                    if masks_pred.shape[2] != patch_dim:
                        masks_pred = F.interpolate(masks_pred, size=(args.patch_dim, args.patch_dim), mode='nearest')
                        # print(f"Resized masks_pred to {masks_pred.shape} because it didnt match patch_dim: {patch_dim}")
                    softmax_masks_pred = F.softmax(masks_pred, dim=1)
                    one_hot_true_masks = F.one_hot(true_masks, num_classes=n_classes).permute(0, 3, 1, 2).float()

                    loss = criterion(masks_pred, true_masks)
                    # print("Cross-entropy loss:", loss.item())

                    loss += dice_loss(
                        softmax_masks_pred,
                        one_hot_true_masks, multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if value is not None and not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if value.grad is not None and not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())


                        val_score, mask_pred, mask_true = evaluate(model, val_loader, device, amp, patch_dim, n_classes, class_reduce, args.architecture)
                        scheduler.step(val_score)
                        
                        # probs = softmax_masks_pred.detach().cpu().numpy()
                        cm = wandb.plot.confusion_matrix(probs=None,y_true=true_masks.float().cpu().numpy().flatten(), preds=masks_pred.argmax(dim=1).float().cpu().numpy().flatten(),class_names=class_names)
                        
                        jaccard_index = jaccard_metric(preds=masks_pred.argmax(dim=1).flatten(), target=true_masks.flatten())

                        jaccard_index_ignore_0 = jaccard_metric_ignore_0(preds=masks_pred.argmax(dim=1).flatten(), target=true_masks.flatten())

                        logging.info('Validation Dice score: {}'.format(val_score))
                        # logging.info('Jaccard index: {}'.format(jaccard_index))
                        logging.info('Jaccard index ignore_index=0: {}'.format(jaccard_index_ignore_0))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'jaccard index': jaccard_index,
                                'jaccard index ignore unannotated': jaccard_index_ignore_0,
                                'confusion matrix': cm,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(label2rgb(true_masks[0].float().cpu().numpy())),
                                    'pred': wandb.Image(label2rgb(masks_pred.argmax(dim=1)[0].float().cpu().numpy())),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except Exception:
                            logging.info('Could not log to wandb. Skipping.')
                            traceback.print_exc()
                            pass
                        
        if save_checkpoint:
            Path(dir_checkpoint).joinpath(wandb.run.id).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            # state_dict['mask_values'] = dataset.mask_values # ! I uncommented this but I don't know if it's required.
            torch.save(state_dict, str(dir_checkpoint / wandb.run.id / 'checkpoint_epoch{}.pth'.format(epoch)))
            if loaded_model:
                # add random characters to the end of the file name to avoid overwriting the original model
                path = str(dir_checkpoint / wandb.run.id / 'checkpoint_epoch{}_{}.pth'.format(epoch, ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))))
                torch.save(state_dict, path)
            else:
                path = str(dir_checkpoint / wandb.run.id / 'checkpoint_epoch{}.pth'.format(epoch))
                torch.save(state_dict, path)
            logging.info(f'Checkpoint {epoch} saved! at path: {path}')

    #Could potentially uncomment this if wandb is hanging after experiments.
    # wandb.finish()

def get_args():
    parser = argparse.ArgumentParser(description='Train the ViT on images and target masks')
    parser.add_argument('--data_path', '-d', metavar='DATA', type=str, help='Path to the dataset (h5 file)', default='data.h5')
    parser.add_argument('--architecture', '-a', metavar='ARCH', type=str, help='Architecture. i.e U-Net, Vision Transformer, CrossFormer, DeepLab v3', default='unet', choices=['unet', 'unet3plus', 'unet_3plus_deepsup', 'unet_3plus_deepsup_cgm', 'vit', 'crossformer', 'crossformer_larger', 'deeplabv3', 'deeplabv3_from_scratch'])
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', '-l', metavar='LR', type=float, default=1e-5, help='Learning rate', dest='lr')
    parser.add_argument('--optim_type', '-o', type=str, default='rmsprop', help='Optimiser to use', choices=['adam','sgd', 'rmsprop'])
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0, help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=11, help='Number of classes')
    parser.add_argument('--weights', type=str, default='challenge', choices=['challenge','normalised','rebalanced','rebalanced_extreme'], help='weights for loss function. Check choices for options.')
    parser.add_argument('--index_to_ignore', type=int, default=-100, help='Index to ignore in loss function')
    parser.add_argument('--patch_dim', type=int, default=1024, help='Patch dimension, 256, 512, 1024')
    parser.add_argument('--load_onto_ram', action='store_true', default=False, help='Load all images onto RAM')
    # parser.add_argument('--normalise', action='store_true', default=False, help='Normalise images, boolean')
    parser.add_argument('--normalise', default=False, type=lambda s: s.lower() == 'true', help='Normalise images (mean & std.), [true or false]')
    parser.add_argument('--class_reduce', default=False, type=lambda s: s.lower() == 'true', help='Reduce the number of ground truths which are fully class [0] and reduce number of masks with just [0,4]')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    if args.architecture == 'unet':
      model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    if args.architecture == 'unet3plus':
      model = UNet_3Plus(
        in_channels=3,
        n_classes=args.classes,
        feature_scale=4,
        is_deconv=True,
        is_batchnorm=True)

    if args.architecture == 'unet_3plus_deepsup':
      model = UNet_3Plus_DeepSup(
        in_channels=3,
        n_classes=args.classes,
        feature_scale=4,
        is_deconv=True,
        is_batchnorm=True)

    if args.architecture == 'unet_3plus_deepsup_cgm':
      model = UNet_3Plus_DeepSup_CGM(
        in_channels=3,
        n_classes=args.classes,
        feature_scale=4,
        is_deconv=True,
        is_batchnorm=True)

    if args.architecture == 'vit':
      model = SimpleViTSegmentation(
          channels=3,
          image_size=args.patch_dim,
          patch_size=32,
          num_classes=args.classes,  # Change this to the number of segmentation classes you have
          dim=1024, # 512, 768, 1024, 2048?
          depth=6,
          heads=16,
          mlp_dim=2048
      )
    
    if args.architecture == 'crossformer':
      model = CrossFormerSegmentation(
          num_classes=args.classes,            # number of segmentation classes
          dim=(64, 128, 256, 512),             # dimension at each stage
          depth=(2, 2, 8, 2),                  # depth of transformer at each stage
          global_window_size=(8, 4, 2, 1),     # global window sizes at each stage
          local_window_size=8,                 # local window size (held constant at 7 for all stages in the paper)
      )

    if args.architecture == 'crossformer_larger':
      model = CrossFormerSegmentation(
        dim = (64, 128, 256, 512),
        depth = (2, 2, 8, 2),
        global_window_size = (8, 4, 2, 1),
        local_window_size = 32,
        cross_embed_kernel_sizes = ((4, 8, 16, 32), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides = (4, 2, 2, 2),
        num_classes = args.classes,
        attn_dropout = 0.,
        ff_dropout = 0.,
        channels = 3
      )

    if args.architecture == 'deeplabv3':
      model = deeplabv3_resnet50(num_classes=args.classes, pretrained_backbone=True)

    if args.architecture == 'deeplabv3_from_scratch':
      model = deeplabv3_resnet50(num_classes=args.classes, pretrained_backbone=False)

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    if torch.cuda.device_count() > 1:
        logging.info(f'Script has access to {torch.cuda.device_count()} GPUs!')
        model = torch.nn.DataParallel(model)

    model.to(device=device)
    try:
        if isinstance(model, torch.nn.DataParallel):
            n_channels = 3
            n_classes = args.classes
            if args.architecture == 'unet':
              bilinear = model.module.bilinear  
        else:
            n_channels = 3
            n_classes = args.classes
            if args.architecture == 'unet':
              bilinear = model.bilinear

        logging.info(f'Network:\n'
                     f'\t{n_channels} input channels\n'
                     f'\t{n_classes} output channels (classes)\n'
                     f'\t{torch.cuda.device_count()} gpu(s)\n'
                     )

        train_model(
            data_path=args.data_path,
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            weights=args.weights,
            index_to_ignore=args.index_to_ignore,
            patch_dim=args.patch_dim,
            loaded_model=args.load,
            architecture=args.architecture,
            load_onto_ram=args.load_onto_ram,
            optim_type=args.optim_type,
            normalise=args.normalise,
            class_reduce=args.class_reduce
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.module.use_checkpointing() if isinstance(model, torch.nn.DataParallel) else model.use_checkpointing()
        train_model(
            data_path=args.data_path,
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            weights=args.weights,
            index_to_ignore=args.index_to_ignore,
            patch_dim=args.patch_dim,
            loaded_model=args.load,
            architecture=args.architecture,
            load_onto_ram=args.load_onto_ram,
            optim_type=args.optim_type,
            normalise=args.normalise,
            class_reduce=args.class_reduce
        )

