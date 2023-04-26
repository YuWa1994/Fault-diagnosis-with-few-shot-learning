

import numpy as np
import scipy.io as sio 
from YW_packages import YW_utils as utils
import os
import Prototypical_Network as PN
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
#  few-shot learning with prototypical network
# ---------------------------------------------
datasets = ['dataA', 'dataB', 'dataC', 'dataD', 'dataE', 'dataF']
datapath = 'D:/YW/Project/Few-shot & zero-shot learning/datasets/'
for i in [0]:
    utils.setup_seed(seed=0)
    data = datasets[i]
    save_result = f'./results/{data}'
    
    #utils.setup_seed(seed=0)
    trainX, trainY, testX, testY = get_data(datapath+data, is_fft=True)
    trainX, trainY = get_sample_data(trainX, trainY, num=10) # randomly select a small number of training data

    
    ### Define Nets
    encoder = PN.encoder()
    
    ### Train Nets
    model = PN.Protonet(trainX, trainY, n_epoch=200, is_normalization=True)
    model = model.train(encoder)
         
    ### test
    out = model.test(testX)
    
    result = {}
    result['distance'] = out['distance']
    result['prediction'] = out['prediction']
    result['embedding'] = out['embedding']
    result['loss'] = model.loss
    result['prototypes'] = model.prototypes.detach().cpu().numpy()
    result['true_label'] = testY
    
        
    if not os.path.exists(save_result): os.makedirs(save_result)
    sio.savemat(f'{save_result}/result-{model.n_support}shot.mat', result)   
    

