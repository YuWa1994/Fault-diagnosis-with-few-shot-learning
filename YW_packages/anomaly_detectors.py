# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 16:41:58 2022

@author: yuwa
"""
import numpy as np
import sklearn
from sklearn.neighbors import NearestNeighbors
import scipy.io as sio
import torch.nn as nn
#import CompactBilinearPooling
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix
from sklearn.covariance import EllipticEnvelope
from YW_packages import YW_utils as utils


def cal_dist(p, q, method='Euclidean', axis=1):
    # p and q must have same dimensions
    if method == 'Euclidean':
        dist =np.mean(np.sqrt(np.square(p-q)), axis)
    return dist

def Mode(L, method='max'):
    # find the most frequent element in a list
    all_count = []
    for i, n in enumerate(np.unique(L)): all_count.append(L.count(n))
    if method == 'max': 
        count = np.max(all_count)
        count_idx = np.unique(L)[np.where(all_count == count)[0]][0]
    return count_idx, count


class knnens():
    ''' KNNENS: A k-Nearest Neighbor Ensemble-Based Method for Incremental
    Learning Under Data Stream With Emerging New Classes
    
    Parameters
    ----------
    X : 
        Array-like of shape (n_samples, n_features),
    label : 
        Array-like of shape (n_samples, 1),    
    num_bags : 
        Defines the number of ensembled base models
    num_bootstraps : 
        Number of samples to train each base model
    buffer_size : 
        Number of samples in buffer
    sub_size : Scalor in (0, 1]
        Defines the dimension of subspace, if 1, all festures are employed to train the model.
        
    Returns
    -------        
    Prediction
    '''

    def __init__(self, X, label, K=10, num_bags=25, num_bootstraps=40,threshold=1, 
                 buffer_size=100, sub_size=1):
        super().__init__()
        self.X = X
        self.label = label
        self.K = K
        self.num_bags = num_bags 
        self.num_bootstraps = num_bootstraps
        self.threshold = threshold 
        self.buffer_size = buffer_size
        self.sub_size = sub_size
        self.buffer = []
        
    def fit_knnens(self):
        X = self.X
        label = self.label
        
        self.n_features = X.shape[1]
        self.class_model = {}
        self.class_model['radius'] = []
        self.class_model['data'] = []
        self.class_model['label'] = []
        self.class_model['subspace_idx'] = []
        self.classes = np.unique(label)
        self.n_classes = len(self.classes)
        for i in range(self.num_bags):
            tmp_label = []
            tmp_radius = []
            tmp_data = []
            subspace_idx = np.random.permutation(self.n_features)[:
                                  int(self.sub_size*self.n_features)] 
            
            for cidx, cj in enumerate(self.classes):
                subX = X[np.where(label==cj)[0],:]
                shuffle_idx = np.random.permutation(subX.shape[0])[:self.num_bootstraps]
                bag_data = subX[shuffle_idx,:][:,subspace_idx]
                knn = NearestNeighbors(n_neighbors=self.K+1).fit(bag_data)
                nn_dist, nn_idx = knn.kneighbors(bag_data)
                radius = np.mean(nn_dist[:,1:], 1)  
                
                tmp_label.extend(np.ones(self.num_bootstraps)*cj)       
                tmp_data.extend(bag_data)
                tmp_radius.extend(radius)
                
            self.class_model['radius'].append(np.array(tmp_radius))
            self.class_model['data'].append(np.array(tmp_data))
            self.class_model['label'].append(np.array(tmp_label))
            self.class_model['subspace_idx'].append(np.array(subspace_idx))
        return self
        
    def predict_knnens(self, X):
        n_instances = X.shape[0]
        pred = []
        for i in range(n_instances):
            pred.append(self.calculate_score(X[i,:]))
        return pred

    def calculate_score(self, X):
        preds = []
        for i in range(self.num_bags):
            tmp_label = self.class_model['label'][i]
            radius = self.class_model['radius'][i]   
            save_data = self.class_model['data'][i]
            subspace_idx = self.class_model['subspace_idx'][i]  
            
            nn_dist = cal_dist(
                np.tile(np.reshape(X[subspace_idx],(1,-1)), (save_data.shape[0], 1)), 
                                save_data)
            
            min_idx = np.where(nn_dist == np.min(nn_dist))[0][0]  
            tmp_pred = tmp_label[min_idx]
            
            score = nn_dist / radius 
            n_s = len(np.where(score<self.threshold)[0])  
            
            if n_s > 0:
                preds.append(tmp_pred)
            else:
                preds.append('new')

        final_pred,_ = Mode(preds) 
        
        if final_pred == 'new':
            final_pred = np.max(self.classes)+1
            self.buffer.append(X)
            if np.array(self.buffer).shape[0] == self.buffer_size:
                self.add_new_class()
                self.buffer = []
            
        return int(final_pred)

    def add_new_class(self):
        X_new = np.array(self.buffer)
        cj = int(np.max(self.classes)+1)
        self.classes = np.insert(self.classes, cj, cj)
        self.n_classes = len(self.classes)
        
        for i in range(self.num_bags):
            subspace_idx = self.class_model['subspace_idx'][i]
            shuffle_idx = np.random.permutation(X_new.shape[0])[:self.num_bootstraps]
            
            bag_data = X_new[shuffle_idx,:][:,subspace_idx]
            knn = NearestNeighbors(n_neighbors=self.K+1).fit(bag_data)
            nn_dist, nn_idx = knn.kneighbors(bag_data)
            radius = np.mean(nn_dist[:,1:], 1)  
            
            self.class_model['radius'][i] = np.concatenate((self.class_model['radius'][i], radius))
            self.class_model['data'][i] = np.concatenate((self.class_model['data'][i], bag_data))
            self.class_model['label'][i] = np.concatenate((self.class_model['label'][i], 
                                                           np.ones(self.num_bootstraps)*cj))    
        return



class SENCForest():
    ''' SENCForest: Classification Under Streaming Emerging New Classes: 
        A Solution UsingCompletely-Random Trees
    '''
    def __init__(self, X, label, num_Tree=10, num_Sub=25, num_Dim=40,
                 height_limit=200,
                 ):
        super().__init__()
        self.num_Tree = num_Tree
        self.num_Sub = num_Sub
        self.num_Dim = num_Dim
        self.c = 2 * (np.log(num_Sub - 1) + 0.5772156649) - 2 * (num_Sub - 1) / num_Sub
        
        self.classes = np.unique(label)
        self.height_limit = height_limit
        
        self.class_idx = []
        for i in self.classes:
            self.class_idx.append(np.where(label==i))
            
        
            
    def fit(self):
        idx_Sub = []
        pathline = []
        pathline3 = []
        self.Trees = []
        self.id = 0
        for i in range(self.num_Tree):
            for j in range(len(self.classes)):
                tmp_idx = self.class_idx[j]
                shuffle_idx = np.random.permutation(tmp_idx.shape[0])
                if tmp_idx.shape[0]<self.num_Sub:
                    print('number of instances is too small.')
                    break
                else:
                    idx_Sub.extend(shuffle_idx[:self.num_Sub])
                    
            self.Trees.append(self.SENCTree(idx_Sub))
        
    def SENCTree(self, idx_Sub):
        Tree = {}
        Tree['Height'] = 0
        num = len(idx_Sub)
        
        if Tree['Height'] >= self.height_limit or num<=10:
            if num>1:
                Tree['NodeStatus'] = 0
                Tree['SplitAttribute'] = []
                Tree['SplitPoint'] = []
                Tree['LeftChild'] = []
                Tree['RightChild'] = []
                Tree['Size'] = num
                Tree['idx'] = idx_Sub
                Tree['x'] = self.x[idx_Sub,:]
                Tree['id'] = self.id
                self.id = self.id+1
                
                           
class _NN_for_LC_INC(nn.Module):
    def __init__(self, classes):
        super(_NN_for_LC_INC, self).__init__()
        self.classes = classes
        self._embedding_module = nn.Sequential(
            nn.Linear(in_features=1024,out_features=512), 
            nn.LeakyReLU(0.2),  
            nn.Linear(in_features=512,out_features=256),
            nn.LeakyReLU(0.2),  
            nn.Linear(in_features=256,out_features=128), 
            nn.LeakyReLU(0.2),           
            )    
        
        self.CBP_layer = CompactBilinearPooling(128, 128, 2048)
        
        self._comparator = nn.Sequential(
            nn.Linear(in_features=2048,out_features=1024), 
            nn.LeakyReLU(0.2),  
            nn.Linear(in_features=1024,out_features=self.classes), 
            nn.Softmax(dim=1)           
            )  

    def forward(self, X, Centers):  
        X_out = self._embedding_module(X)
        Centers_out = self._embedding_module(Centers)
        out = self.CBP_layer(X_out, Centers_out)
        out = self._comparator(out)
        return out         
    
class LC_INC():
    ''' Learning to Classify With Incremental New Class 
    '''
    def __init__(self, X, label,buffer_size=100,tradeoff_parameter=0.3
                 ):
        super().__init__()
        self.X = X
        self.label = label
        self.buffer_size = buffer_size
        self.buffer = []
        self.classes = len(np.unique(label))
        
        self.tradeoff_parameter = tradeoff_parameter
        
        self.model = _NN_for_LC_INC(self.classes)
        
            
    def fit(self):
        idx_Sub = []
        pathline = []
        pathline3 = []
        self.Trees = []
        self.id = 0
        for i in range(self.num_Tree):
            for j in range(len(self.classes)):
                tmp_idx = self.class_idx[j]
                shuffle_idx = np.random.permutation(tmp_idx.shape[0])
                if tmp_idx.shape[0]<self.num_Sub:
                    print('number of instances is too small.')
                    break
                else:
                    idx_Sub.extend(shuffle_idx[:self.num_Sub])
                    
            self.Trees.append(self.SENCTree(idx_Sub))

        
    def class_center(self, X, Y):
        self.class_center = []
        for i in range(self.known_classes):
            c = X[np.where(Y==i)]
            self.class_center.append(np.mean(c, axis=1))
    
    def get_score(self, y): 
        scores = (min(self.tradeoff_parameter*self.chunck_size, 1)*y +
             max(self.tradeoff_parameter*self.chunck_size, 1)*self.Entropy_measure(y))
        return scores
    
    def Entropy_measure(p, axis=1, eps=0.0000001):
        return -1*np.sum(p*np.log2(p+eps), axis)    


class traditional_novelty_detector():
    ''' Including LOF, iForest, OCSVM
    
    Parameters
    ----------
    X : 
        Array-like of shape (n_samples, n_features),
    label : 
        Array-like of shape (n_samples, 1),    

    buffer_size : 
        Number of samples in buffer

    Returns
    -------        
    Prediction
    '''

    def __init__(self, X, label, model, 
                 is_update=True, 
                 is_normalization=True,
                 buffer_size=150):
        super().__init__()
        self.X = X
        self.label = label
        self.model = model
        self.is_update = is_update 
        self.is_normalization = is_normalization
        self.buffer_size = buffer_size
        self.buffer = []
        self.n_classes = len(np.unique(self.label)) 
        
    def fit_model_N(self, trainX=None):
        if trainX is None: trainX = self.X
        if self.is_normalization:
            trainX, self.zscore_mu, self.zscore_sigma = utils.zscore(trainX, dim=0)
            
        if self.model == 'LOF':
            model_N = LocalOutlierFactor(n_neighbors=5, novelty=True, contamination=0.01).fit(trainX) 
            
        elif self.model == 'iForest':
            model_N = IsolationForest(n_estimators=200, 
                                    max_features=3,
                                    bootstrap=False, contamination=0.01).fit(trainX)            
        elif self.model == 'OCSVM':
            model_N = OneClassSVM(nu=0.01, kernel="rbf", gamma=0.01).fit(trainX)            
        return model_N

    def fit_model_C(self, trainX=None):
        if trainX is None: trainX = self.X
        if self.is_normalization:
            trainX, self.zscore_mu, self.zscore_sigma = utils.zscore(trainX, dim=0)
        trainY = np.ravel(self.label)

        if self.n_classes > 1:
            model_C = sklearn.ensemble.RandomForestClassifier(n_estimators=200) 
            model_C = model_C.fit(trainX, trainY)             
        else:
            model_C = None    
        return model_C

        
    def model_predict(self, X, model_N, model_C):
        n_instances = X.shape[0] 
        X = np.array_split(X, n_instances)
        pred = []
        self.update_time = []
        for i in range(n_instances):
            data = X[i]
            if self.is_normalization: 
                data = utils.normalize(data, self.zscore_mu, self.zscore_sigma)             
            
            prediction = int(model_N.predict(data))    

            if prediction == -1:
                prediction = self.n_classes+1
                if self.is_update:
                    self.buffer.extend(data)
                    if np.array(self.buffer).shape[0]>=self.buffer_size:
                        print('--------Update model--------')
                        model_N, model_C = self.update_model(self.buffer)      
                        self.update_time.append(i)
            if prediction == 1 and self.n_classes>1:
                prediction = int(model_C.predict(data))
                
            pred.append(prediction)
        return pred

    def update_model(self, buffer):
        self.X = np.append(self.X, self.buffer, axis=0)  
        self.label = np.append(self.label, 
                      np.ones((self.buffer_size, 1))*self.n_classes, 
                      axis=0)       
        self.n_classes = self.n_classes + 1
        self.buffer = []
        model_N = self.fit_model_N(self.X)
        model_C = self.fit_model_C(self.X)
        return model_N, model_C



if __name__ == "__main__":
    data = sio.loadmat('./synthetic_data.mat')
    trainX = data['tr_X']
    trainY = data['tr_y']
    #testY = data['te_y']
    newX =  np.concatenate((data['te_X'],
                            5*np.random.normal(0, 1, (200, 2)), 
                            10*np.random.normal(0, 1, (200, 2))))
    
    model = knnens(trainX, trainY-1).fit_knnens()  
    out = model.predict_knnens(newX)


    