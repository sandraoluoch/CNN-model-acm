"""
This script contains import functions for training, validation and evaluation of a CNN UNet model:
1) save_checkpoint: creates and saves the checkpoint
2) load_checkpoint: loads a previously saved checkpoint
3) get_loaders: preps training and validation datasets for their corresponding dataloaders
4) accuracy_loss_dice: calculates accuracy, loss, and dice score
5) save_predictions_as_imgs: resize the predictions to their original size and save them as .tif files
"""
import os
import tqdm
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
import torch.nn as nn
from acm_dataset import ACM_Dataset
from tifffile import TiffWriter
import torch.nn.functional as F
import matplotlib.pyplot as plt
from monai.inferers import SlidingWindowInferer


# create and save checkpoint
def save_checkpoint(state, filename = "acm_mini_checkpoint.pth.tar"):
    print("=> save checkpoint")
    torch.save(state, filename)

# load checkpoint
def load_checkpoint(checkpoint, model):
    print("=> loading checkpoint")
    model.load_state_dict(checkpoint["model"])

# prepare dataloader
def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        train_transform,
        val_transform, 
        batch_size,
        num_workers=4,
        pin_memory=True
        
):
    # training dataset
    train_ds = ACM_Dataset(
        img_dir = train_dir,
        mask_dir = train_maskdir,
        transform = train_transform

    )

    # training dataloader
    train_dataloader = DataLoader(
        train_ds,
        batch_size,
        num_workers = num_workers,
        pin_memory = pin_memory,
        shuffle = True

    )

    # validation dataset
    val_ds = ACM_Dataset(
        img_dir = val_dir,
        mask_dir = val_maskdir,
        transform = val_transform

    )

    # validation dataloader
    val_dataloader = DataLoader(
        val_ds,
        batch_size,
        num_workers = num_workers,
        pin_memory = pin_memory,
        shuffle = False

    )  
    return train_dataloader, val_dataloader 

# accuracy, loss and dice score function
def accuracy_loss_dice(loader, model, loss_fn=None, device="cuda"):
    # for accuracy
    num_correct = 0
    num_pixels = 0
    #for dice score
    dice_score = 0
    dice_numerator = 0
    dice_denominator = 0
    # for loss
    total_loss = 0.0 
    num_batches = 0
    
    inferer = SlidingWindowInferer(roi_size=(16, 128, 128), sw_batch_size=4, overlap=0.25, mode="gaussian")
    
    with torch.no_grad():
        model.eval() # put model in eval mode

        for batch in loader: 
            x = batch["image"].to(device)
            y = batch["label"].to(device).float()
            
            raw_preds = torch.sigmoid(inferer(inputs=x, network=model))
            y = (y > 0.5).float()

            # for loss
            if loss_fn is not None:
                loss = loss_fn(raw_preds, y)
                total_loss += loss.item()
                num_batches += 1
            
            # binarize for metrics
            preds = (raw_preds > 0.3).float()

            # for accuracy 
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

            # for dice score
            dice_numerator += 2 * (preds * y).sum()
            dice_denominator += preds.sum() + y.sum()
        

    accuracy = num_correct/num_pixels   
    dice_score = (dice_numerator)/(dice_denominator + 1e-8) # 1e-8 to prevent dividing by zero
    avg_loss = total_loss / num_batches if num_batches > 0 else None

    print(f"got accuracy of {(accuracy)*100:2f}%")
    print(f"got dice score of {dice_score:4f}")
    model.train() # put model back in training mode

    return accuracy, avg_loss, dice_score

# save predictions as images 
def save_predictions_as_imgs(model, loader, folder="saved_images/", device="cuda"):
   model.eval() # put model in eval mode

   # Original size you want to upscale to
   original_size = (30, 1848, 1248)  # [D, H, W]

   inferer = SlidingWindowInferer(
       roi_size=[16,128,128],
       sw_batch_size=4,
       mode="gaussian"
   )
   
   for idx, batch in enumerate(loader):
        x = batch["image"].to(device)
  

        with torch.no_grad():
            preds = inferer(inputs=x, network=model)
            preds = (preds > 0.3).float()
            preds = preds.detach().cpu()
            preds_filename = os.path.join(folder, f"preds_{idx}.tif")


            # Remove batch & channel dim if needed: convert [1, C, D, H, W] to [D, H, W]
            preds_np = preds.squeeze().squeeze().cpu().numpy()

            # convert [D, H, W] to [D, W, H] 
            preds_np = preds_np.transpose(0, 2, 1)  
            
            # print(f"Saving prediction {idx}, shape: {preds_np.shape}")  # Should be [30, 1248, 1848]
    
            with TiffWriter(preds_filename) as tif:
                tif.write((preds_np * 255).astype("uint8"))


            model.train()

        

        

