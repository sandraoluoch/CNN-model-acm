This is a repository for an experimental version of the deep learning model (Convolutional Neural Network) used in the All Cells Mask Pipeline for the Allen Institute for Science. The images generated from this model are used for analysis on the following paper:
https://www.biorxiv.org/content/10.1101/2024.08.16.608353v1.full 

<b> Paper Overview: </b>
- Goal is to understand EMT better by observing cell colony changes. Currently not understood very well
- 449 60-hour-long timelapse movies were split into 3D z-stacks and binary segmentations were generated from these image stacks 
- Cell colonies exist and move into different formations. Found that the EMT changes occurred earlier in the 2D colonies due to environment compared to 3D colonies. 
- MONAI (Medical Open Network for AI) is a PyTorch-based, open-source framework for deep learning in healthcare imaging. Provides standardized workflows for networks, losses, evaluation metrics, and more. 
- Feature extraction - the binary all-cells-mask was used to extract the colony’s area at each Z-slice, and total intensity was measured by the corresponding fluorescent channel
- For data validation the ground truth and predictions were overlaid on their raw image counterparts and spot checking 15% of data
- After training this model was used to predict binary colony masks for all 3D images

<b> My contribution: </b>
1) Data preprocessing and ground truth generation (created colony masks) 
2) Qualitative data validation of predictions vs ground truths
3) Helped build and train early versions of the deep learning model (CNN U-Net)
4) Ran inference on model and made predictions on large image datasets

<b> Image Dataset Examples </b>

2D Colony

![alt text](image-1.png)
Raw Brightfield Image (right), CytoGFP image (middle), Ground Truth Segmentation (left)

3D Colony

![alt text](image.png)
Raw Brightfield Image (right), CytoGFP image (middle), Ground Truth Segmentation (left)

