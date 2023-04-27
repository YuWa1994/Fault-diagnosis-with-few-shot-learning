# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 14:32:04 2021

@author: yuwa
"""

import os
import numpy as np
import pandas as pd
import zipfile
import math
import torch
from torch.autograd import Variable
import scipy.io as sio
    

def mapminmax_normalize(xtrain, xtest, dim=0):
    xtrain,x_max, x_min = mapminmax(xtrain, dim)
    xtest = mapminmax_ps(xtest, x_max, x_min)
    return xtrain, xtest, x_max, x_min


def mapminmax(x, dim=0):
    x_max = np.max(x, axis=dim)
    x_min = np.min(x,axis=dim)
    x_out = (x - x_min) / (x_max-x_min)
    return x_out, x_max, x_min


def mapminmax_ps(x, x_max, x_min):
    x_out = (x - x_min) / (x_max-x_min)
    return x_out


def zscore(x, dim=0):
    mu = np.mean(x, axis=dim)
    sigma = np.std(x, axis=dim)
    x_out = (x - mu) / sigma
    return x_out, mu, sigma

def normalize(x, mu, sigma):
    return (x - mu) / sigma

def zscore_normalize(xtrain, xtest, dim=0):
    xtrain, mu, sigma = zscore(xtrain, dim)
    xtest = normalize(xtest, mu, sigma)
    return xtrain, xtest, mu, sigma
    
def slide_window(stream, win, num=None, stride=None, start=0):
    '''
    stream : 
        A 1*n vector.
    win : 
        The length of window.
    '''
    if np.size(stream,0) != 1: stream = stream.reshape(1,-1)
    if stride is None: stride = win
    if num is None: num = math.floor((np.size(stream)-win)/stride + 1)
    
    index = list(range(start, np.size(stream)-stride, stride))
    if num>np.size(index): num = np.size(index)
    
    data = np.zeros((num, win))
    for i in range(num):
        data[i,:] = stream[:,index[i]:index[i]+win]
    
    return data
    
def shuffle_data(x, dim=0):
    # np.random.seed()
    indices = np.random.permutation(np.size(x,dim))
    x = x[indices]  
    return x, indices

def sort_data(x):
    x_unique = np.unique(x)
    
    sorted_x = []
    indices = []
    
    for n, i in enumerate(x_unique):
        indices.extend(np.where(x==n)[0])
        sorted_x.extend(x[np.where(x==n)])
    return np.array(indices), np.array(sorted_x)


def split_train_test(x, test_ratio=0.5, dim=0):
    # np.random.seed(random_state)
    
    shuffled_indices = np.random.permutation(np.size(x,dim))
    test_set_size = math.floor(int(len(x)) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    x_train = x[train_indices]
    x_test = x[test_indices]     
    return x_train, x_test, train_indices, test_indices

def split_train_val_test(x, test_ratio=0.2, val_ratio=0.1, dim=0):
    # np.random.seed(random_state)
    
    shuffled_indices = np.random.permutation(np.size(x,dim))
    test_set_size = math.floor(int(len(x)) * test_ratio)
    val_set_size = math.floor(int(len(x)) * val_ratio)
    train_set_size = int(len(x)) - test_set_size - val_set_size
       
    train_indices = shuffled_indices[:train_set_size ]
    test_indices = shuffled_indices[train_set_size :train_set_size +test_set_size]
    val_indices = shuffled_indices[train_set_size +test_set_size:]
    
    x_train = x[train_indices]
    x_tset = x[test_indices]  
    x_val = x[val_indices]
    return x_train, x_tset, x_val, train_indices, test_indices, val_indices

def get_one_hot(Y, dtype=float, dim=None):
    '''
    Convert a vector to one-hot matrix
    '''
    #Y = pd.get_dummies(Y)
    #Y = Y.to_numpy().astype(dtype)

    if dim is None: dim = int(np.max(Y)+1)
    Y = list(Y)
    Y = [int(x) for x in Y]
    Y = np.eye(dim)[Y]
    return Y
    
def creat_label(class_num, one_hot=False, dtype=float):
    '''
    class_num: a vector contains the number of each class
    If parameter one_hot=True, return a one-hot matrix
    '''
    label = []
    for index, num in enumerate(class_num):
        c = (index+1) * np.ones([num, 1])
        label = np.append(label, c)
    if one_hot:
        label = get_one_hot(label, dtype)
    return label    


def get_ZipFile(datapath, savepath=None, columns=1, skiprow=None, dataname=None, 
                data_selector=None,
                func=lambda n: n[6:10]+'.'+n[3:5]+'.'+n[0:2]+' '+n[11:-4]):
    '''
    From a folder with zipped files read cvs data
    '''
    # skiprow = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    # columns = (0,2,4,6)
    allfiles = []
    for (_, _, filenames) in os.walk(datapath): allfiles.extend(filenames) 
    if data_selector is None: data_selector=np.arange(len(allfiles), dtype=int)
        
    for i, name in enumerate(allfiles):
        if i in data_selector:
            this_zip = zipfile.ZipFile(datapath+'/'+name, mode='r')
            names = this_zip.namelist()           
            file = this_zip.open(names[0])      
    
            data = np.array(pd.read_csv(file, skiprows=skiprow))
            data = data[:,columns].astype(float)
            dataname = func(name)
    
            sio.savemat(f'{savepath}/{dataname}.mat', {'data':data}) 
            file.close()
            this_zip.close()  
    return
    
def unzip_file(filepath, columns=(1), skiprow=None,):
    this_zip = zipfile.ZipFile(filepath, mode='r')
    names = this_zip.namelist()           
    file = this_zip.open(names[0])      
    
    data = np.array(pd.read_csv(file, skiprows=skiprow))
    data = data[:,columns].astype(float)

    file.close()
    this_zip.close()      
    return data


def get_AllFileName_from_folder(folder, 
                                func=lambda n:n[:-4]):
    allfiles = []
    for (_, _, name) in os.walk(folder): 
        for i, n in enumerate(name):
            filename = func(n)
            allfiles.append(filename) 
    return allfiles


def rename_files_in_folder(path, func=lambda n: n[6:10]+'.'+n[3:5]+'.'+n[0:2]+' '+n[11:]):
    files = [i for i in os.listdir(path)]
    for i, n in enumerate(files):
        if n[:4] != '2022':
            old = path +'/'+n
            new = path +'/' + func(n)
            os.rename(old, new)


def setup_seed(seed=0):
    np.random.seed(seed)    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def snr_signal(X, snr=8, ratio=1, dim=1):
    X2 = X
    if ratio<1: 
        indices = np.random.permutation(X.shape[0])  
        indices = indices[:int(X.shape[0]*ratio)]
        X = X[indices]
        
    noise = np.random.randn(X.shape[0], X.shape[1])
    noise = noise - np.mean(noise, axis=dim).reshape(-1,1)
    signal_power = np.sum(X**2, axis=dim).reshape(-1,1)/X.shape[1]
    noise_power = signal_power / 10**(snr/10)
    noise_std = np.sqrt(noise_power)/np.std(noise,axis=dim).reshape(-1,1)
    noise = noise_std * noise
    
    if ratio<1:
        X2[indices] = X + noise
        output = X2
    else: output = X + noise 
    
    return output
     
def get_cell2mat(data):
    X = []
    for i, ind in enumerate(data):
        X.extend(np.concatenate(data[i]))
    return np.array(X)    
     
def empty_file(path):
    for i in os.listdir(path):
       path_file = os.path.join(path,i)
       if os.path.isfile(path_file):
          os.remove(path_file)
       else:
         for f in os.listdir(path_file):
             path_file2 =os.path.join(path_file,f)
             if os.path.isfile(path_file2):
                 os.remove(path_file2)     
     
def _corrupt(X):
    dummy = np.random.uniform(0,1,X.shape)
    dummy = dummy-np.random.uniform(0,0.3,X.shape[0]).reshape(-1,1,1)
    indices_zero = np.where(dummy<0)    
    X[indices_zero] = 0
    return X          
     
def _print_parameter(self, model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
    return   

def _convert_to_torch(X):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.from_numpy(X).to(device)
    X = Variable(X).float()
    return X

def save_checkpoint(model, save_path, epoch=None):
    torch.save({
        'model_state_dict': model.state_dict(),
        #'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)
    
    
def load_checkpoint(model, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, epoch    


def load_my_state_dict(model, pre_train, 
                       drop_dict=[]):
    ### transfer the pre-trained weight
    
    pretrain_dict = pre_train.state_dict()
    model_dict = model.state_dict()
    
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k not in drop_dict}
    
    for i, name in enumerate(drop_dict):
        pretrain_dict[name] = model_dict[name]
    
    model_dict.update(pretrain_dict) 
    model.load_state_dict(pretrain_dict)
    return model



def Mode(L, method='max'):
    # find the most frequent element in a list
    all_count = []
    for i, n in enumerate(np.unique(L)): all_count.append(L.count(n))
    if method == 'max': 
        count = np.max(all_count)
        count_idx = np.unique(L)[np.where(all_count == count)[0]][0]
    return count_idx, count
    

def get_data(datapath, is_normalization=False, test_ratio=0.5):
    data = sio.loadmat(datapath)
    try:
        trainX = data['trainX'].astype(np.float32) 
        trainY = data['trainY']-1        
        testX = data['testX'].astype(np.float32) 
        testY = data['testY']-1                
    except:
        X = data['X'].astype(np.float32)  
        Y = data['Y']
        trainX, testX, train_indices, test_indices = split_train_test(X, 
                                      test_ratio=test_ratio, dim=0)
        trainY, testY = Y[train_indices], Y[test_indices]

    if is_normalization:
        trainX, mu, sigma = zscore(trainX, dim=0)
        X = normalize(X, mu, sigma)
    return trainX, trainY, testX, testY