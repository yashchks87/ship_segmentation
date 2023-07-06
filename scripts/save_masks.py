"""
Author: Yash Choksi
Date: 12/28/2022
Updated: 07/05/2023
"""

import pandas as pd
import pickle
import PIL
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import multiprocessing as mp
import argparse

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

def save_masks(data):
    try:
        image_ids, pixels, path_begin = data[0], data[1], data[2]
        file_name = path_begin + image_ids.split('.')[0]
        if type(pixels[0]) != str:
            Image.fromarray(np.array(np.zeros(shape=(768, 768)).astype(np.uint8))).save(f'{file_name}.png')
        else:
            img = masks_as_image(pixels)
            img = Image.fromarray((img * 255).reshape(768, 768).astype(np.uint8))
            img.save(f'{file_name}.png')
        return 'DONE'
    except:
        return data[0]

def save_masks_helper(csv_file_path, col_1 = 'ImageId', col_2 = 'EncodedPixels', path_begin = '../../masks/', pool_size = 15):
    csv_file = pd.read_csv(csv_file_path)
    csv_file = csv_file.groupby('ImageId')['EncodedPixels'].apply(list).reset_index()
    image_ids, pixels = csv_file['ImageId'].values.tolist(), csv_file['EncodedPixels'].values.tolist()
    if os.path.exists(path_begin) == False:
        os.makedirs(path_begin)
    with mp.Pool(pool_size) as p:
        results = p.map(save_masks, [(image_ids[x], pixels[x], path_begin) for x in range(len(image_ids))])
    for x in results:
        if x != 'DONE':
            print(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'Save masks script.',
                    description = 'Script to save masks for ground truth labels.')
    parser.add_argument(
        '-c', '--csv_file_path', required = True
    )
    parser.add_argument(
        '-p', '--path_begin', required = True
    )
    parser.add_argument(
        '-ps', '--pool_size', required = True
    )
    args = parser.parse_args()
    save_masks_helper(csv_file_path=args.csv_file_path, path_begin=args.path_begin, pool_size=int(args.pool_size))


# python save_masks.py -c ../../files/train_ship_segmentations_v2.csv -p ../../files/masks_v1/train/ -ps 25