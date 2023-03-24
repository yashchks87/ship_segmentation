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
import os
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


load_dotenv('../config_files/dev.env')

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

strategy = tf.distribute.get_strategy()
AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync

with open('./data.pickle', 'rb') as handle:
    updated_train_csv = pickle.load(handle)

def split_datasets(data, test_size = 0.01):
    train, test = train_test_split(data, test_size = test_size, random_state = 42) 
    train, val = train_test_split(train, test_size = test_size, random_state = 42)
    return train, val, test

train, val, test = split_datasets(updated_train_csv)


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
        loss = keras.losses.BinaryCrossentropy()
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
log_dir = f"{os.environ['tb_path']}classification/xcep_baseline/"
if os.path.exists(log_dir) == False:
    os.makedirs(log_dir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir)
weights_path = '/home/ubuntu/ml-data-training/weights/weights/xcep_baseline/'
weights_save = CallbackForSavingModelWeights(weights_path)
batch_size = 512
train_dataset = get_data(train)
val_dataset = get_data(val, repeat = False, shuffle = False)
model = create_model('Xception', (256, 256, 3))
model = compile_new_model(model)
model_hist = model.fit(
    train_dataset,
    validation_data = val_dataset,
    verbose = 1,
    epochs = 100,
    steps_per_epoch = len(train) // (batch_size * REPLICAS),
    callbacks = [
        tensorboard_callback,
        weights_save
    ]
)

print('Training finished.')