import tensorflow as tf
import multiprocessing as mp

def helper(data):
    try:
        img = tf.io.read_file(data)
        img = tf.image.decode_jpeg(img, channels = 3)
        return 'DONE'
    except:
        return data

def find_bad_ones(data_list):
    with mp.Pool(20) as p:
        issues = p.map(helper, data_list)
    bad_ones = []
    for x in issues:
        if x != 'DONE':
            bad_ones.append(x)
    return bad_ones