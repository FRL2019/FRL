import time
import pickle
import numpy as np
import tensorflow as tf

# helper methods to print nice table (taken from CGT code)
def fmt_item(x, l):
    if isinstance(x, np.ndarray):
        assert x.ndim==0
        x = x.item()
    if isinstance(x, float): rep = "%g"%x
    else: rep = str(x)
    return " "*(l - len(rep)) + rep

def fmt_row(width, row):
    out = " | ".join(fmt_item(x, width) for x in row)
    return out

def flipkernel(kern):
    return kern[(slice(None, None, -1),) * 2 + (slice(None), slice(None))]

def conv2d_flipkernel(x, k, name=None):
    return tf.nn.conv2d(x, flipkernel(k), name=name,
                        strides=(1, 1, 1, 1), padding='SAME')

def save_pkl(obj, path):
    with open(path, 'w') as f:
        pickle.dump(obj, f)


def load_pkl(path):
    with open(path) as f:
        obj = pickle.load(f)
        return obj


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def timeit(f):
    def timed(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()

        print("   [-] %s : %2.5f sec" % (f.__name__, end_time - start_time))
        return result
    return timed