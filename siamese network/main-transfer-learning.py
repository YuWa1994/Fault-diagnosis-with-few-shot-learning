

import numpy as np
import scipy.io as sio 
from YW_packages import YW_utils as utils
import os
import Siamese_Network as SN
import random


def get_data(datapath, is_normalization=False, is_fft=False):
    data = sio.loadmat(datapath)
    
    try:
        trainX = data['trainX'].astype(np.float32) 
        testX = data['testX'].astype(np.float32) 
        trainY = data['trainY']
        testY = data['testY']   
        if trainY.shape[0]>1:
            trainY = np.argmax(trainY, axis=1)
            testY = np.argmax(testY, axis=1)
        else:
            trainY = trainY-1
            testY = testY-1
    except:
        X = data['X'].astype(np.float32) 
        Y = np.argmax(data['Y'], axis=1)
        trainX, testX, train_indices, test_indices = utils.split_train_test(X, test_ratio=0.5)
        trainY, testY = Y[train_indices], Y[test_indices]
        
    if is_fft:
        trainX = np.absolute(np.fft.fft(trainX, axis=1)).astype(np.float32) 
        testX = np.absolute(np.fft.fft(testX, axis=1)).astype(np.float32) 

    if is_normalization:
        trainX, mu, sigma = utils.zscore(trainX, dim=0)
        testX = utils.normalize(testX, mu, sigma)
    return trainX, trainY, testX, testY


def get_sample_data(X, Y, num=10, classes=None):
    if classes is None: classes = len(np.unique(Y))
    
    Xs, Ys = [], [],
    for i in range(classes):
        idx = np.where(Y==i)
        idx = random.sample(list(idx[0]), num)
        
        Xs.extend(X[idx])    
        Ys.extend(Y[idx])    
    return np.array(Xs), np.array(Ys)

# ---------------------------------------------
#  transfer learning with siamese network
# ---------------------------------------------
datasets = ['dataA', 'dataB', 'dataC', 'dataD', 'dataE', 'dataF']
datapath = 'D:/YW/Project/Few-shot & zero-shot learning/datasets/'

utils.setup_seed(seed=0)
source_data = datasets[0]
target_data = datasets[3]

save_result = f'./results/{target_data}'

# source data
trainX, trainY, testX, testY = get_data(datapath+source_data, is_fft=True)
trainX, trainY = get_sample_data(trainX, trainY, num=5, classes=10) # randomly select a small number of training data


### Define Nets
embedding = SN.embedding(in_size=trainX.shape[1])

### Train source task
model = SN.Siamese_net(trainX, trainY, n_epoch=1000, is_normalization=True)
model = model.train(embedding)
     
### train target task
trainX, trainY, testX, testY = get_data(datapath+target_data, is_fft=True)
trainX, trainY = get_sample_data(trainX, trainY, num=5, classes=10) # randomly select a small number of training data    
model = model.transfer(trainX, trainY, n_epoch=100)

### test
out = model.test(testX)    
result = {}
result['prediction'] = out['prediction']
result['dist'] = out['dist']
result['loss'] = model.loss
result['true_label'] = testY

    
if not os.path.exists(save_result): os.makedirs(save_result)
sio.savemat(f'{save_result}/result-{model.n_support}shot.mat', result)  
    

