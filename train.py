"""
This script trains a UNet-based CNN model on 3D EMT stem cell images. 
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from monai.losses import DiceCELoss
from tqdm import tqdm
import torch.optim as optim
from acm_model import UNET
from utils import (save_checkpoint, 
                   accuracy_loss_dice,
                   get_loaders,
                   save_predictions_as_imgs
                   )
from monai.data import ITKReader, image_reader
from monai.transforms import (LoadImaged, Compose, LoadImaged, 
                              ToTensord, LambdaD, Zoomd, RandomizableTransform,
                              RandSpatialCropd, RandHistogramShiftd)
import mlflow
                 
# Hyperparameters

LEARNING_RATE = 1e-4
NUM_EPOCHS = 40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MEMORY = False
BATCH_SIZE = 1
TRAIN_IMGDIR = "/allen/aics/users/sandra.oluoch/acm_mini/train_imgs/ch1/"
TRAIN_MASKDIR = "/allen/aics/users/sandra.oluoch/acm_mini/clean_train_masks/"
VAL_IMGDIR = "/allen/aics/users/sandra.oluoch/acm_mini/val_imgs/ch1/"
VAL_MASKDIR = "/allen/aics/users/sandra.oluoch/acm_mini/clean_val_masks/"

# parameters for MLFlow
params = {
    "random_state": 888,
    "learning_rate": LEARNING_RATE,
    "epochs": NUM_EPOCHS,
    "optimizer": "Adam",
    "resize_shape": "(30, 474, 320)",
    "loss": "DiceCELoss(sigmoid=True)"
}

# set up tracking url for ML Flow
mlflow.set_tracking_uri("http://127.0.0.1:8080")
acm_experiment = mlflow.set_experiment("acm_mini_sandi")
run_name = "acm_mini_sandi_v1"
# artifact_path = "acm_mini_model"

# this function converts input into pytorch tensor and then reorders the dimensions to add a channel dimension. 3D to 4D
def reorder_dhw_to_cdhw(x):
    
    x = torch.as_tensor(x)
    # print("Before reorder, tensor is shape =", x.shape)

    if x.ndim == 3:
        x = x.permute(2, 0, 1)         # Convert [H, W, D] to [D, H, W]
        x = x.unsqueeze(0)             # Add a channel so x is now [1, D, H, W]
    elif x.ndim == 4:
        if x.shape[0] != 1:
            raise ValueError(f"Expected 1st dim to be channel=1, but got shape {x.shape}")
    else:
        raise ValueError(f"Unexpected shape {x.shape}")

    # print("After reorder, shape =", x.shape)
    return x

# the train function   
def train_fn(loader, model, loss_fn, optimizer, scaler):
    loop = tqdm(loader)
    epoch_loss = 0.0
    num_batches = 0

    for batch in loop:  
        data = batch["image"].to(DEVICE)
        targets = batch["label"].float().to(DEVICE)

        # print("Patch shape:", data.shape)


        # forward
        with torch.amp.autocast(device_type=DEVICE):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backpropagation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        num_batches += 1

        loop.set_postfix(loss = loss.item())

    return epoch_loss/num_batches 

# function that applies loads training and validation sets and performs necessary transforms 
def main():
    train_transform = Compose([
        LoadImaged(keys=["image", "label"], reader=ITKReader(), image_only=False),
        LambdaD(keys=["image", "label"], func=reorder_dhw_to_cdhw), 
        RandSpatialCropd(keys=["image", "label"], roi_size=[16, 128, 128], random_center=True, random_size=False),
        LambdaD(keys=["label"], func=lambda x: (x > 0).float()), 
        Zoomd(keys=["image", "label"], zoom=0.25, keep_size=True),
        RandHistogramShiftd(keys=["image"], prob=0.1, num_control_points=[90,500]),
        ToTensord(keys=["image", "label"]) 

    ])

    val_transform = Compose([ 
        LoadImaged(keys=["image", "label"], reader=ITKReader(), image_only=False),
        LambdaD(keys=["image", "label"], func=reorder_dhw_to_cdhw),   
        LambdaD(keys=["label"], func=lambda x: (x > 0).float()),
        ToTensord(keys=["image", "label"]), 

    ])

   
    # STEP ONE: INSTANTIATE MODEL
    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
        
    # STEP TWO: INSTANTIATE LOSS FUNCTION
    # loss_fn = DiceCELoss(sigmoid=True)
    loss_fn = DiceCELoss(sigmoid=True, include_background=False, to_onehot_y=False)

    # STEP THREE: INSTANTIATE OPTIMIZER
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    # STEP FOUR: SET UP TRAINING AND VALIDATION DATALOADERS 
    train_loader, val_loader = get_loaders(
        TRAIN_IMGDIR,
        TRAIN_MASKDIR,
        VAL_IMGDIR,
        VAL_MASKDIR,
        train_transform,
        val_transform,
        BATCH_SIZE,
        NUM_WORKERS,
        PIN_MEMORY

    )

    # STEP FIVE: START TRAINING
    scaler = torch.cuda.amp.GradScaler()
    best_dice = 0
    patience = 5
    epochs_no_improve = 0 

    # Force CUDA context init BEFORE mlflow
    if DEVICE == "cuda":
        print("Warming up CUDA...")
        torch.cuda.current_device()
        print("Using device:", torch.cuda.get_device_name(0))

    with mlflow.start_run(run_name=run_name) as run:

        mlflow.log_params(params)


        for epoch in range(NUM_EPOCHS):
    
            # define checkpoint
            checkpoint ={
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }

            # run the training function and calculate accuracy and loss
            train_accuracy, _, _ = accuracy_loss_dice(train_loader, model, device=DEVICE)
            train_loss = train_fn(train_loader, model, loss_fn, optimizer, scaler)
            
            # validation accuracy and loss
            val_accuracy, val_loss, dice_score = accuracy_loss_dice(val_loader, model, loss_fn, device=DEVICE)

            # metrics for MLFlow
            mlflow.log_metric("epoch", epoch)
            mlflow.log_metric("training_loss", train_loss)
            mlflow.log_metric("training_accuracy", train_accuracy)
            mlflow.log_metric("validation_loss", val_loss)
            mlflow.log_metric("validation_accuracy", val_accuracy)
            mlflow.log_metric("dice_score", dice_score)

            print(f"Epoch {epoch + 1} done.")
            torch.cuda.empty_cache()
            
            # early stopping
            if dice_score > best_dice:
                best_dice = dice_score
                epochs_no_improve = 0
                save_checkpoint(checkpoint)
                mlflow.pytorch.log_model(model, artifact_path="best_model")
            else:
                epochs_no_improve +=1
            if epochs_no_improve >= patience:
                print("Epochs no longer improving. Early stopping triggered.")
                break 

            # save predicted images
            save_predictions_as_imgs(model, val_loader, folder='saved_images/', device=DEVICE)

            # save train/validation metrics as an image
            # mlflow.log_artifact('train-validation-loss.png')

    return run 

if __name__ == "__main__":
    main() 
