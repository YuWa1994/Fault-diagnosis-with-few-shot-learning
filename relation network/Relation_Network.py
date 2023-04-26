import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable
from YW_packages import YW_utils as utils
import random


class Relation_net(nn.Module):
    def __init__(self, trainX, trainY, 
                 n_epoch=100,
                 is_normalization=True,
                 ):
        super().__init__()
        self.X = trainX
        self.num = trainX.shape[0]
        
        if is_normalization:
            trainX, self.zscore_mu, self.zscore_sigma = utils.zscore(trainX, dim=0)
        
        self.n_class = len(np.unique(trainY))
        self.trainX = trainX  
        self.trainY = trainY  

        self.n_epoch = n_epoch
        self.is_normalization = is_normalization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_support_query(self, X=None, Y=None, num_support=1):
        if Y is None: Y = self.trainY
        if X is None: X = self.trainX
        
        Xs, Xq, Ys, Yq = [], [], [], []
        for i in range(len(np.unique(Y))):
            idx = np.where(Y==i)[0]
            idx = idx[np.random.permutation(len(idx))]
            
            Xs.extend(X[idx[:num_support], :])   # support data
            Xq.extend(X[idx[num_support:], :])   # query data
            Ys.extend(Y[idx[:num_support]])      # support label
            Yq.extend(Y[idx[num_support:]])      # query label
        
        Xs = Variable(torch.from_numpy(np.array(Xs)).to(self.device))
        Xq = Variable(torch.from_numpy(np.array(Xq)).to(self.device))

        Ys = torch.from_numpy(np.array(Ys))
        Yq = torch.from_numpy(np.array(Yq))
        self.n_support = num_support
        self.n_query = len(idx)-num_support
        return Xs, Xq, Ys, Yq


    def get_pairs(self, Ps, Pq):
        _Ps = Ps.repeat(Pq.shape[0],1,1)
        _Pq = Pq.repeat(Ps.shape[0],1,1)
        _Pq = torch.transpose(_Pq,0,1)
        relation_pairs = torch.cat((_Ps,_Pq),2).view(-1, Pq.shape[1]*2)
        return relation_pairs

    
    def train(self, embedding_module, relation_module, 
              optim_EM=None, optim_RM=None, loss_func=None):   
        self.embedding_module = embedding_module.to(self.device)
        self.relation_module = relation_module.to(self.device)
              
        self.loss_func = torch.nn.MSELoss() if loss_func is None else loss_func
        self.optim_EM = torch.optim.RMSprop(self.embedding_module.parameters(), lr=0.001) if optim_EM is None else optim_EM
        self.optim_RM = torch.optim.RMSprop(self.relation_module.parameters(), lr=0.001) if optim_RM is None else optim_RM
        
        self.loss = []
        for epoch in range(self.n_epoch):  
            Xs, Xq, Ys, Yq = self.get_support_query(num_support=1)
            
            Zs = self.embedding_module(Xs)
            Zq = self.embedding_module(Xq)
            #Zs = Xs
            #Zq = Xq
            
            relation_pairs = self.get_pairs(Zs, Zq)
            #label_pairs = self.get_pairs(Ys.view(-1,1), Yq.view(-1,1))
            relations = self.relation_module(relation_pairs).view(-1,self.n_class)
            
            label = F.one_hot(torch.from_numpy(np.array(Yq)).long(), 
               num_classes=self.n_class).to(self.device)   
            
            loss = self.loss_func(relations, label.type_as(relations))
                       
            self.optim_EM.zero_grad()
            self.optim_RM.zero_grad()
            loss.backward()
            self.optim_EM.step()  
            self.optim_RM.step() 
            
            self.loss.append(loss.detach().cpu().numpy())
            
            _, predict = relations.max(1)
            _, real = label.max(1)
            acc = torch.eq(predict, real).float().mean()
            print("[Epoch %d/%d] [Loss : %f] [acc : %f]" % (epoch, self.n_epoch, loss.data, acc.data))    
        return self


    def test(self, X):
        self.embedding_module.eval()
        self.relation_module.eval()
        num = X.shape[0]

        prediction = []
        relation = []
        for i in range(num):
            testX = X[i]
            if self.is_normalization: 
                testX = utils.normalize(testX, self.zscore_mu, self.zscore_sigma)  
                
            testX = Variable(torch.from_numpy(testX).to(self.device))
            Zq = self.embedding_module(testX.view(1,-1))
            
            relations = torch.zeros(1,10).to(self.device)
            for i in range(1):
                Xs, _, Ys, _ = self.get_support_query(num_support=1)
                Zs = self.embedding_module(Xs)
                relation_pairs = self.get_pairs(Zs, Zq)
                relations += self.relation_module(relation_pairs).view(-1,self.n_class)
            
            _, predict = relations.max(1)
            prediction.extend(predict.detach().cpu().numpy())
            relation.extend(relations.detach().cpu().numpy())
        
        return {
        'prediction': prediction,
        'relation': relation,
        }


    def transfer(self, trainX, trainY, train_num=None, n_epoch=None, 
                 loss_func=None, optim_EM=None, optim_RM=None,
                 is_normalization=True, is_same_class=True):
        if is_same_class:
            self.embedding_module.train()
            self.relation_module.train()
            
        if loss_func is not None: self.loss_func = loss_func
        if optim_EM is not None: self.optim_EM = optim_EM 
        if optim_RM is not None: self.optim_RM = optim_RM 
        
        if n_epoch is None: n_epoch = self.n_epoch
        if is_normalization:
            trainX, self.zscore_mu, self.zscore_sigma = utils.zscore(trainX, dim=0)
        
        if train_num is not None:  
            trainX, trainY = get_sample_data(trainX, trainY, num=train_num)
        
        self.loss = []
        for epoch in range(n_epoch):
 
            Xs, Xq, Ys, Yq = self.get_support_query(trainX, trainY, num_support=1)
            
            Zs = self.embedding_module(Xs)
            Zq = self.embedding_module(Xq)

            relation_pairs = self.get_pairs(Zs, Zq)
            #label_pairs = self.get_pairs(Ys.view(-1,1), Yq.view(-1,1))
            relations = self.relation_module(relation_pairs).view(-1,self.n_class)
            
            label = F.one_hot(torch.from_numpy(np.array(Yq)).long(), 
               num_classes=self.n_class).to(self.device)   
            
            loss = self.loss_func(relations, label.type_as(relations))
                       
            self.optim_EM.zero_grad()
            self.optim_RM.zero_grad()
            loss.backward()
            self.optim_EM.step()  
            self.optim_RM.step() 
            
            self.loss.append(loss.detach().cpu().numpy())
            
            _, predict = relations.max(1)
            _, real = label.max(1)
            acc = torch.eq(predict, real).float().mean()
            print("[Epoch %d/%d] [Trans_Loss : %f] [Trans_acc : %f]" % (epoch, n_epoch, loss.data, acc.data))  
        return self

def get_sample_data(X, Y, num=10, classes=None):
    if classes is None: classes = len(np.unique(Y))
    
    Xs, Ys = [], [],
    for i in range(len(np.unique(classes))):
        idx = np.where(Y==i)
        idx = random.sample(list(idx[0]), num)
        
        Xs.extend(X[idx])    
        Ys.extend(Y[idx])    
    return np.array(Xs), np.array(Ys)


class embedding_module(nn.Module): 
    def __init__(self, in_size=1024, in_dim=1, hid_dim=32, z_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            self.conv_block(in_dim, hid_dim, 15),
            self.conv_block(hid_dim, hid_dim, 7),
            self.conv_block(hid_dim, hid_dim, 7),
            self.conv_block(hid_dim, z_dim, 7),
            )
        
    def conv_block(self, in_channels, out_channels, ks=9):
        p = int(ks/2)
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=ks, padding=p),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.AvgPool1d(2)
        )        
        
    def forward(self, x):    
        x = x.reshape(-1, 1, x.shape[1])
        out = self.encoder(x)
        out = torch.flatten(out, 1)
        return out    


class relation_module(nn.Module): 
    def __init__(self, in_size=1024, in_dim=2, hid_dim=32, z_dim=32, flatten_size=None):
        super().__init__()
        self.conv = nn.Sequential(
            self.conv_block(in_dim, hid_dim, 15),
            self.conv_block(hid_dim, hid_dim, 7),
            self.conv_block(hid_dim, z_dim, 7),
            #self.conv_block(hid_dim, z_dim, 7),
            )
        
        if flatten_size is None: flatten_size = int(0.5*z_dim*in_size/pow(2,len(self.conv)))
        self.fc = nn.Sequential(
            nn.Linear(flatten_size, 1),
            #nn.ReLU(),
            nn.Sigmoid(),
            )
        
    def conv_block(self, in_channels, out_channels, ks=9):
        p = int(ks/2)
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=ks, padding=p),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.AvgPool1d(2)
        )        
        
    def forward(self, x):    
        x = x.reshape(x.shape[0], 2, -1)
        out = self.conv(x)
        out = torch.flatten(out, 1)
        
        out = self.fc(out)
        return out    
