import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm


def find_weights(**kwargs):
    # print(kwargs['csv_file'])
    paths = kwargs['csv_file']['mask_paths'].values.tolist()
    img_array = []
    # print(paths)
    for path in tqdm(paths):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels = 1)
        img = tf.image.resize(img, size = kwargs['shape'])
        img = img / 255.0
        img = img.numpy()
        img = np.where(img < 1, 0, 1)
        img_array.append(img)
    arr = np.array(img_array)
    arr = arr.flatten()
    print('Calculating weights using sklearn.....')
    weight = compute_class_weight(class_weight = 'balanced', classes = [0, 1], y = arr)
    weight_dict = {
        0 : weight[0],
        1 : weight[1],
    }
    return weight_dict
    
    

if __name__ == '__main__':
    find_weights()