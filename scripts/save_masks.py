"""
Author: Yash Choksi
Date: 12/28/2022
"""

import os
import PIL
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from  PIL import Image

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.int16)
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)

def save_masks(csv_file_path, col_1 = 'ImageId', col_2 = 'EncodedPixels', path_begin = '../../masks/'):
    """
        It takes csv file as input and generate masks and save them to given path.
        Args:
        csv_file: Actual csv file(NOT A PATH),
        col_1: Column name for fetching image names; default value as ImageId,
        col_2: Pixel col name; default value is EncodedPixels,
        path_begin: It will store masks at this path.
        * Make sure column names are ImageId and EncodePixels.
    """
    csv_file = pd.read_csv(csv_file_path)
    csv_file = csv_file.groupby('ImageId')['EncodedPixels'].apply(list).reset_index()
    image_ids, pixels = csv_file['ImageId'].values.tolist(), csv_file['EncodedPixels'].values.tolist()
    if os.path.exists(path_begin) == False:
        os.makedirs(path_begin)
    for x in tqdm(range(len(image_ids))):
        file_name = path_begin + image_ids[x].split('.')[0]
        if type(pixels[x][0]) != str:
            Image.fromarray(np.array(np.zeros(shape=(768, 768)).astype(np.uint8))).save(f'{file_name}.png')
        else:
            img_0 = masks_as_image(pixels[x])
            img = Image.fromarray((img_0 * 255).reshape(768, 768).astype(np.uint8))
            img.save(f'{file_name}.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'Save masks script.',
                    description = 'Script to save masks for ground truth labels.')
    parser.add_argument(
        '-c', '--csv_file_path', required = True
    )
    parser.add_argument(
        '-c1', '--col_1', required = False, default = 'ImageId'
    )
    parser.add_argument(
        '-c2', '--col_2', required = False, default = 'EncodedPixels'
    )
    parser.add_argument(
        '-p', '--path_begin', required = True
    )
    args = parser.parse_args()
    # print(args.csv_file)
    save_masks(args.csv_file_path, args.col_1, args.col_2, args.path_begin)