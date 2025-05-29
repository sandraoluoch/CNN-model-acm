import os
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    This class performs convolution twice with batch normalization, dropout and activation function. Made 
    to be used in the UNET class below. 
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            # convolution 1
            nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=True), 
            nn.InstanceNorm3d(out_channels), 
            nn.ReLU(), 
            nn.Dropout3d(p=0.2),

            # convolution 2
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=True), 
            nn.InstanceNorm3d(out_channels), 
            nn.ReLU() 
        )

    def forward(self, x):
        return self.conv(x)
    
class UNET(nn.Module):
    """
    This class is a convolutional neural network model based on the UNET architecture from 
    this paper: https://arxiv.org/abs/1505.04597. The U-Net model is a powerful tool for image segmentation.
    """
    def __init__(self, in_channels, out_channels, features=[32, 64, 128, 256]):
        super().__init__()

        self.down = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2) 

        # down the U (encoder section, downsampling). Goal is extract more complex features 
        for feature in features:
            self.down.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # bottleneck step (doubling the features). Goal is to capture high-level abstract features 
        self.bottleneck = (DoubleConv(features[-1], features[-1]*2))

        # up the U (decoder section, upsampling). Goal is to reconstruct output with same resolution as input 
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose3d(feature*2, feature, kernel_size=2, stride=2)) # upsampling step
            self.ups.append(DoubleConv(feature*2, feature)) # double convolution step

        # final convolution (last step) 
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # down the U
        for down in self.down:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # the bottleneck 
        x = self.bottleneck(x)
        reversed_skips = skip_connections[::-1] 

        # up the U
        for idx in range(0, len(self.ups), 2): 
            x = self.ups[idx](x) 
            skip_connection = reversed_skips[idx//2] 

            if x.shape != skip_connection.shape: # x.shape and skip_connections.shape should be: [B, C, D, H, W]
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="trilinear", align_corners=False) 

            concat_skip = torch.cat([x, skip_connection], dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)
    




                             

        
