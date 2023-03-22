import PIL
import multiprocessing as mp
import numpy as np

def helper(data):
    try:
        img = PIL.Image.open(data)
        img = np.array(img)
        return 'DONE'
    except:
        return data

def find_bad_ones_torch(data_list):
    with mp.Pool(20) as p:
        issues = p.map(helper, data_list)
    bad_ones = []
    for x in issues:
        if x != 'DONE':
            bad_ones.append(x)
    return bad_ones