import torch
import torch.nn as nn

# Loss function
def dice_bce_loss(inputs, targets, smooth = 1e-5):
    # remove if your model inherently handles sigmoid
    number_of_pixels = inputs.shape[0] * (512 * 512 * 3)
    # sigmoid = nn.Sigmoid()
    # inputs = sigmoid(inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    dice_loss = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    dice_loss = 1 - dice_loss
    # Pixel wise log loss is calculated not number of images
    # I checked reduce by mean is correct measure.
    BCE = nn.functional.binary_cross_entropy(inputs, targets, reduce='mean')
    final = dice_loss + BCE
    return final, number_of_pixels