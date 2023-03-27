import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob
import sys
from dotenv import load_dotenv
from tqdm import tqdm
tqdm.pandas()
import math
from sklearn.model_selection import train_test_split
import PIL
import tensorflow.keras.backend as K
sys.path.append('../scripts/helper_functions_cv/tensorflow_helpers/')
from save_weights_every_epoch import CallbackForSavingModelWeights
from multiprocessing import Pool
from sklearn.utils import compute_class_weight
from sklearn.metrics import precision_score, recall_score, accuracy_score
import tensorflow_datasets as tfds
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot
import multiprocessing as mp
sys.path.append('../scripts/')
from find_bad_ones import find_bad_ones
import pickle
import json


load_dotenv('../config_files/dev.env')
parser = argparse.ArgumentParser()
parser.add_argument("--file_location")
args = parser.parse_args()
params_path = args.file_location
assert os.path.exists(params_path) == True, "File does not exists"
with open(params_path) as f:
    data = f.read()

dict_params = dict(json.loads(data))

allowed_gpus = dict_params['allowed_gpus']
gpus = tf.config.list_physical_devices("GPU")
final_gpu_list = [gpus[x] for x in allowed_gpus]
tf.config.set_visible_devices(final_gpu_list, "GPU")

strategy = tf.distribute.MirroredStrategy()
AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync


with open(dict_params['input_file_location'], 'rb') as handle:
    updated_train_csv = pickle.load(handle)

def split_datasets(data, test_size = 0.01):
    train, test = train_test_split(data, test_size = test_size, random_state = 42) 
    train, val = train_test_split(train, test_size = test_size, random_state = 42)
    return train, val, test

train, val, test = split_datasets(updated_train_csv)

train_labels = train['class_labels'].values.tolist()
computed = compute_class_weight(class_weight='balanced', classes=[0, 1], y=train_labels)
class_weight_dict = {
    0: computed[0],
    1: computed[1]
}

print(class_weight_dict)

def read_train_imgs(img, label, shape):
    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, channels = 3)
    img = tf.image.resize(img, size = shape)
    img = img / 255
    return img, label

def get_data(data, shape = (256, 256), shuffle = True, repeat = True, batch = True, batch_size = 32):
    imgs, labels = data['fixed_paths'].values.tolist(), data['class_labels'].values.tolist()
    shapes = [shape for x in range(len(imgs))]
    tensor = tf.data.Dataset.from_tensor_slices((imgs, labels, shapes))
    tensor = tensor.cache()
    if repeat:
        tensor = tensor.repeat()
    if shuffle:
        tensor = tensor.shuffle(8048 * 1)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        tensor = tensor.with_options(opt)
    tensor = tensor.map(read_train_imgs)
    if batch:
        tensor = tensor.batch(batch_size * REPLICAS)
    tensor = tensor.prefetch(AUTO)
    return tensor

def create_model(model_name, shape):
    with strategy.scope():
        input_layer = tf.keras.Input(shape = shape)
        construct = getattr(keras.applications, model_name)
        mid_layer = construct(include_top = False, 
                            weights = None, 
                            pooling = 'avg')(input_layer)
        last_layer = keras.layers.Dense(1, activation = 'sigmoid')(mid_layer)
        model = keras.Model(input_layer, last_layer)
    return model
def compile_new_model(model):
    with strategy.scope():
        loss = keras.losses.BinaryCrossentropy(label_smoothing=0.05)
        optimizer = keras.optimizers.SGD()
        prec = keras.metrics.Precision(name = 'prec')
        rec = keras.metrics.Recall(name = 'rec')
        model.compile(
            loss = loss,
            optimizer = optimizer,
            metrics = [prec, rec]
        )
    return model

K.clear_session()
log_dir = f"{os.environ['tb_path']}classification/{dict_params['l_w_folder']}/"
if os.path.exists(log_dir) == False:
    os.makedirs(log_dir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir)
weights_path = f'/home/ubuntu/ml-data-training/weights/weights/{dict_params["l_w_folder"]}/'
weights_save = CallbackForSavingModelWeights(weights_path)
batch_size = dict_params['batch_size']
train_dataset = get_data(train, shape=(32, 32), batch_size = batch_size)
val_dataset = get_data(val, shape=(32, 32), repeat = False, shuffle = False, batch_size=batch_size)
model = create_model(dict_params['model_name'], (32, 32, 3))
model = compile_new_model(model)
model.load_weights(f'/home/ubuntu/ml-data-training/weights/weights/res50_class_weight_ls/30.h5')
model_hist = model.fit(
    train_dataset,
    validation_data = val_dataset,
    verbose = 1,
    epochs = dict_params['epochs'],
    steps_per_epoch = len(train) // (batch_size * REPLICAS),
    callbacks = [
        tensorboard_callback,
        weights_save
    ],
    class_weight = class_weight_dict
)


print('Training finished.')