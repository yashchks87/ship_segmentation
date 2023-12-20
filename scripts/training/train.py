import torch
import torch.nn as nn
import wandb
import torch.optim as optim
from tqdm import tqdm
import sys
sys.path.append('../')
from training.loss import dice_bce_loss
from metrics.iou_score import iou_score

def train_model(model, train_set, val_set, epochs, device):
    # wandb.init(project = 'ship-segmentation-pytorch-wb')
    model = nn.DataParallel(model)
    model = model.to(device)
    datadict = {
        'train': train_set,
        'val' : val_set
    }
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.01)
    for epoch in range(epochs):
        train_loss, train_iou = 0.0, 0.0
        val_loss, val_iou = 0.0, 0.0
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss, running_iou = 0.0, 0.0
            with tqdm(datadict[phase], unit='batch') as tepoch:
                for img, label in tepoch:
                    img = img.to(device)
                    label = label.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(img)
                        loss, _ = dice_bce_loss(outputs, label)
                        iou = iou_score(outputs, label)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item()
                    running_iou += iou.item()
                    tepoch.set_postfix(loss = loss.item(), iou = iou.item())
            if phase == 'train':
                train_loss = running_loss / len(datadict['train'])
                train_iou = running_iou / len(datadict['train'])
                print(f'Train Loss: {train_loss}')
                print(f'Train IOU: {train_iou}')
            else:
                val_loss = running_loss / len(datadict['val'])
                val_iou = running_iou / len(datadict['val'])
                print(f'Val Loss: {val_loss}')
                print(f'Val IOU: {val_iou}')
        # wandb.log({
        #     'train_loss' : train_loss,
        #     'val_loss' : val_loss,
        #     'train_iou' : train_iou,
        #     'val_iou' : val_iou
        # })