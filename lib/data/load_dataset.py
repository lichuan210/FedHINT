import os
import numpy as np
import pandas as pd
import pickle

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_st_dataset(dataset):
    if 'PEMS03' in dataset:
        data = np.load('./datasets/PEMS03/PEMS03.npz')
        data = data['data']
        adj_matrix = load_pickle('./datasets/PEMS03/adj_PEMS03.pkl')
    elif 'PEMS04FLOW' in dataset:
        data = np.load('./datasets/PEMS04/PEMS04.npz')
        data = data['data']
        data = data[:,:,0]
        adj_matrix = load_pickle('./datasets/PEMS04/adj_PEMS04.pkl')
    elif 'PEMS07' in dataset:
        data = np.load('./datasets/PEMS07/PEMS07.npz')
        data = data['data']
        adj_matrix = load_pickle('./datasets/PEMS07/adj_PEMS07.pkl')
    elif 'PEMS08FLOW' in dataset:
        data = np.load('./datasets/PEMS08/PEMS08.npz')
        data = data['data']
        data = data[:,:,0]
        adj_matrix = load_pickle('./datasets/PEMS08/adj_PEMS08.pkl')
    else:
        raise ValueError

    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    return data, adj_matrix



