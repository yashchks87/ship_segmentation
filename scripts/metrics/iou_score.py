import torch

# IOU metric
# SMOOTH = 1e-5
def iou_score(inputs, targets, thres = 0.5, smooth=1e-5):
    # sigmoid = nn.Sigmoid()
    # inputs = sigmoid(inputs)
    if thres != None:
        inputs = (inputs > thres).float()
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = torch.sum(inputs * targets)
    unioun = torch.sum(inputs + targets) - intersection
    # TP = torch.sum(torch.logical_and(inputs == 1, targets == 1))
    # FP = torch.sum(torch.logical_and(inputs == 1, targets == 0))
    # FN = torch.sum(torch.logical_and(inputs == 0, targets == 1))
    iou = (intersection + smooth) / (unioun + smooth)
    return iou