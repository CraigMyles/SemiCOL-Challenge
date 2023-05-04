import torch
import torch.nn.functional as F
from tqdm import tqdm

# from torchmetrics.classification import MulticlassConfusionMatrix

from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, patch_dim, n_classes=11, class_reduce=False, architecture='unet'):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch[0], batch[1]

            # if mask_true.sum() == 0:
            #     print('Skipping empty mask [eval]')
            #     continue

            if class_reduce:
                if (mask_true == 0).all(): # if the unique values in true_masks are only 0, continune
                    continue
                elif torch.equal(torch.unique(mask_true).sort()[0], torch.tensor([0, 4])): # if [0,4], 10% chance of continue (skip).
                    if torch.rand(1) < 0.1:  # 10% probability
                        continue
                else:
                    pass

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.squeeze(1).to(device=device, dtype=torch.long)



            # predict the mask
            mask_pred = net(image)

            try:
                if 'out' in mask_pred:
                    mask_pred = mask_pred['out']
            except:
                pass

            if architecture == 'unet_3plus_deepsup_cgm' or architecture == 'unet_3plus_deepsup':
                # combine the 5 outputs and average them to get the final output
                mask_pred = torch.mean(torch.stack(mask_pred), dim=0)

            # if mask_pred shape [1] is not equal to patch_dim, then resize it
            if mask_pred.shape[2] != patch_dim:
                mask_pred = F.interpolate(mask_pred, size=(patch_dim, patch_dim), mode='nearest')

            assert mask_true.min() >= 0 and mask_true.max() < n_classes, 'True mask indices should be in [0, n_classes['
            # convert to one-hot format
            mask_true = F.one_hot(mask_true, n_classes).permute(0, 3, 1, 2).float()
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), n_classes).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

            # cm = MulticlassConfusionMatrix(num_classes=n_classes)
            # confusion_matrix = cm(mask_pred, mask_true)

    net.train()
    return dice_score / max(num_val_batches, 1), mask_pred, mask_true
