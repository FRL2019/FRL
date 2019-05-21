#coding:utf-8
import os
import time
import ipdb
import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg') # do not require GUI


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def plot_results(results, domain, filename):
    print('\nSave results to %s' % filename)
    fontsize = 20
    if isinstance(results, list):
        plt.figure()
        plt.plot(range(len(results)), results, label='loss')
        plt.title('domain: %s' % domain)
        plt.xlabel('episodes', fontsize=fontsize)
        plt.legend(loc='best', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)  
        plt.yticks(fontsize=fontsize) 
        plt.savefig(filename, format='pdf')
        print('Success\n')

    else:
        plt.figure(figsize=(16, 20)) # , dpi=300
        plt.subplot(311)
        x = range(len(results['rec']))
        plt.plot(x, results['rec'], label='rec')
        plt.plot(x, results['pre'], label='pre')
        plt.plot(x, results['f1'], label='f1')
        plt.title('domain: %s' % domain, fontsize=fontsize)
        plt.xlabel('episodes', fontsize=fontsize)
        plt.legend(loc='best', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)  
        plt.yticks(fontsize=fontsize) 

        plt.subplot(312)
        plt.plot(range(len(results['rw'])), results['rw'], label='reward')
        plt.xlabel('episodes', fontsize=fontsize)
        plt.legend(loc='best', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)  
        plt.yticks(fontsize=fontsize) 

        if 'loss' in results:
            plt.subplot(313)
            plt.plot(range(len(results['loss'])), results['loss'], label='loss')
            plt.xlabel('episodes', fontsize=fontsize)
            plt.legend(loc='best', fontsize=fontsize)
            plt.xticks(fontsize=fontsize)  
            plt.yticks(fontsize=fontsize) 
        
        plt.subplots_adjust(wspace=0.5,hspace=0.5)
        plt.savefig(filename, format='pdf')
        print('Success\n')


def ten_fold_split_ind(num_data, fname, k, random=True):
    """
    TODO:   Split data for k-fold-cross-validation
    params: num_data: len(data); 
            fname:    save indices to file; 
            k:        k fold; 
            random:   split randomly or sequentially
    return: indices of k-fold data
    """
    print('Getting tenfold indices ...')
    if os.path.exists(fname):
        print('Loading tenfold indices from %s\n' % fname)
        return load_pkl(fname)

    n = num_data/k
    indices = []

    if random:
        tmp_idxs = np.arange(num_data)
        np.random.shuffle(tmp_idxs)
        for i in range(k):
            if i == k - 1:
                indices.append(tmp_idxs[i*n: ])
            else:
                indices.append(tmp_idxs[i*n: (i+1)*n])
    else:
        for i in xrange(k):
            indices.append(range(i*n, (i+1)*n))

    save_pkl(indices, fname)
    return indices


def index2data(indices, data):
    """
    TODO:   Split data into k-fold according to indices
    params: indices: indices of k-fold data
            data:    data to be split; can be a dict or a list
    return: k-fold data, including training data and validation data
    """
    print('Spliting data according to indices ...')
    folds = {'train': [], 'valid': []}
    if type(data) == dict:
        keys = data.keys()
        print('data.keys: {}'.format(keys))
        num_data = len(data[keys[0]])
        for i in range(len(indices)):
            valid_data = {}
            train_data = {}
            for k in keys:
                valid_data[k] = []
                train_data[k] = []
            for idx in range(num_data):
                for k in keys:
                    if idx in indices[i]:
                        valid_data[k].append(data[k][idx])
                    else:
                        train_data[k].append(data[k][idx])
            folds['train'].append(train_data)
            folds['valid'].append(valid_data)
    else:
        num_data = len(data)
        for i in range(len(indices)):
            valid_data = []
            train_data = []
            for idx in range(num_data):
                if idx in indices[i]:
                    valid_data.append(data[idx])
                else:
                    train_data.append(data[idx])
            folds['train'].append(train_data)
            folds['valid'].append(valid_data)

    return folds


def timeit(f):
    def timed(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()

        print("   [-] %s : %2.5f sec" % (f.__name__, end_time - start_time))
        return result
    return timed


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pkl(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
        return obj

@timeit
def save_npy(obj, path):
    np.save(path, obj)
    print("  [*] save %s" % path)

@timeit
def load_npy(path):
    obj = np.load(path)
    print("  [*] load %s" % path)
    return obj


if __name__ == '__main__':
    plot_results(range(1000), 'test_loss', 'figures/test_loss')
    results = {'rec': np.arange(100)*0.01, 'pre': np.arange(100)*0.01, 'f1': np.arange(100)*0.01, 
                'loss': np.arange(100)*700, 'reward': np.arange(100)*0.5}
    plot_results(results, 'test_all', 'figures/test_all')