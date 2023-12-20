import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import pandas as pd
class GetData(Dataset):
    def __init__(self, csv_file: pd.DataFrame, img_size: int, mask_size: int):
        self.img_paths = csv_file['fixed_inputs'].values.tolist()
        self.mask_paths = csv_file['mask_paths'].values.tolist()
        self.img_size = img_size,
        self.mask_size = mask_size
    
    def __len__(self) -> int:
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img = torchvision.io.read_file(self.img_paths[index])
        img = torchvision.io.decode_jpeg(img)
        mask = torchvision.io.read_file(self.mask_paths[index])
        mask = torchvision.io.decode_image(mask)
        img = torchvision.transforms.functional.resize(img, (256, 256))
        mask = torchvision.transforms.functional.resize(mask, (68, 68))
        img = img / 255
        mask = mask / 255
        return img, mask
    

def generate_loader(csv_file, img_size = 256, mask_size = 68, num_workers = 22, shuffle = True, batch_size = 482):
    dataset = GetData(csv_file, img_size, mask_size)
    dataloader = DataLoader(dataset, num_workers=num_workers, shuffle=shuffle, batch_size=batch_size)
    return dataset, dataloader