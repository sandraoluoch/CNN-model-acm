from tifffile import imread, imwrite
import numpy as np
import os

input_dir = "/allen/aics/users/sandra.oluoch/acm_mini/val_masks"
output_dir = "/allen/aics/users/sandra.oluoch/acm_mini/clean_val_masks"

# os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if fname.endswith(".tif"):
        mask = imread(os.path.join(input_dir, fname))
        bin_mask = np.isin(mask, [128, 255]).astype(np.uint8)  # adjust threshold as needed
        imwrite(os.path.join(output_dir, fname), bin_mask)
print("Before...")
print("Unique:", np.unique(mask))
print("Min:", mask.min(), "Max:", mask.max(), "Dtype:", mask.dtype)

# mask = imread("/allen/aics/users/sandra.oluoch/acm_mini/clean_train_masks/3500006256_20240223_20X_Timelapse_scene_28_T0_C=2.tif")
# print("After...")
# print("Unique:", np.unique(mask))
# print("Min:", mask.min(), "Max:", mask.max(), "Dtype:", mask.dtype)