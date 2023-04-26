import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable
from YW_packages import YW_utils as utils
import random


class Protonet(nn.Module):
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

    def get_support_query(self, X=None, Y=None, num_support=None):
        if Y is None: Y = self.trainY
        if X is None: X = self.trainX
        
        if num_support==None: 
            num_support = int(0.5*X.shape[0]/self.n_class)
        
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
        Ys = np.array(Ys)
        Yq = np.array(Yq)
        '''
        Yq = F.one_hot(torch.from_numpy(np.array(Yq)).long(), 
                       num_classes=self.n_class).to(self.device)     
        '''
        self.n_support = num_support
        self.n_query = len(idx)-num_support
        return Xs, Xq, Ys, Yq


    def get_prototype(self, x, y):
        Z = []
        for i in range(self.n_class):
            idx = np.where(y==i)[0]
            Z.append(x[idx].mean(0))
        Z = torch.stack(Z)        

        '''
        out = out.reshape(self.n_class, -1, out.shape[1])
        Z = out.mean(1)
        '''
        return Z

    def get_loss(self, Zq, Z_proto):
        
        target_inds = torch.arange(0, self.n_class).view(self.n_class, 1, 1).expand(self.n_class, self.n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)
        target_inds = target_inds.to(self.device)        
        dists = euclidean_dist(Zq, Z_proto)
        log_p_y = F.log_softmax(-dists, dim=1).view(self.n_class, self.n_query, -1)
           
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)        
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
        
        return loss_val, {
          'loss': loss_val.item(),
          'acc': acc_val.item(),
          'y_hat': y_hat
          }     
        
    
    def train(self, encoder, optim=None, loss_func=None):   
        self.encoder = encoder.to(self.device)
        self.loss_func = torch.nn.MSELoss() if loss_func is None else loss_func
        self.optim = torch.optim.RMSprop(self.encoder.parameters(), lr=0.001) if optim is None else optim
        
        self.loss = []
        self.prototypes = []
        for epoch in range(self.n_epoch):  
            Xs, Xq, Ys, Yq = self.get_support_query(num_support=None)
            
            embedding = self.encoder(torch.cat((Xs, Xq), axis=0))
            Zs = embedding[:self.n_support*self.n_class]
            Z_proto = self.get_prototype(Zs, Ys)
            
            Zq = embedding[self.n_support*self.n_class:]
            
            loss, output = self.get_loss(Zq, Z_proto)
            
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()  
            
            self.loss.append(loss.detach().cpu().numpy())
            print("[Epoch %d/%d] [Loss : %f] [acc : %f]" % (epoch, self.n_epoch, loss.data, output['acc']))    
        self.prototypes = Z_proto
        return self


    def test(self, X):
        self.encoder.eval()
        num = X.shape[0]

        prediction = []
        embedding = []
        distance = []
        for i in range(num):
            testX = X[i]
            if self.is_normalization: 
                testX = utils.normalize(testX, self.zscore_mu, self.zscore_sigma)  
                
            testX = Variable(torch.from_numpy(testX).to(self.device))
            embed = self.encoder(testX.view(1,-1))
            dists = euclidean_dist(embed, self.prototypes)
            log_p_y = F.log_softmax(-dists, dim=1)

            _, pred = log_p_y.max(1)     
            
            prediction.extend(pred.detach().cpu().numpy())
            embedding.extend(embed.detach().cpu().numpy())
            distance.extend(dists.detach().cpu().numpy())
        
        return {
        'prediction': prediction,
        'embedding': embedding,
        'distance': distance,
        }


    def transfer(self, trainX, trainY, train_num=None, n_epoch=None, 
                 loss_func=None, optim=None,
                 is_normalization=True, is_same_class=True):
        if is_same_class:
            self.encoder.train()
            
        if loss_func is not None: self.loss_func = loss_func
        if optim is not None: self.optim = optim 
        
        if n_epoch is None: n_epoch = self.n_epoch
        if is_normalization:
            trainX, self.zscore_mu, self.zscore_sigma = utils.zscore(trainX, dim=0)
            
        if train_num is not None:  
            trainX, trainY = get_sample_data(trainX, trainY, num=train_num)
        
        self.loss = []
        self.prototypes = []
        for epoch in range(n_epoch):
 
            Xs, Xq, Ys, Yq = self.get_support_query(trainX, trainY, num_support=None)
            
            embedding = self.encoder(torch.cat((Xs, Xq), axis=0))
            Zs = embedding[:self.n_support*self.n_class]
            Z_proto = self.get_prototype(Zs, Ys)
            
            Zq = embedding[self.n_support*self.n_class:]

            loss, output = self.get_loss(Zq, Z_proto)
            
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()  
            
            self.loss.append(loss.detach().cpu().numpy())
            print("[Epoch %d/%d] [Trans_Loss : %f] [Trans_acc : %f]" % (epoch, n_epoch, loss.data, output['acc']))    
        self.prototypes = Z_proto

        return self

def euclidean_dist(x, y):
    """
    Computes euclidean distance btw x and y
    Args:
        x (torch.Tensor): shape (n, d). n usually n_way*n_query
        y (torch.Tensor): shape (m, d). m usually n_way
    Returns:
        torch.Tensor: shape(n, m). For each query, the distances to each centroid
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)
    
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    
    return torch.pow(x - y, 2).sum(2)

def get_sample_data(X, Y, num=10, classes=None):
    if classes is None: classes = len(np.unique(Y))
    
    Xs, Ys = [], [],
    for i in range(len(np.unique(classes))):
        idx = np.where(Y==i)
        idx = random.sample(list(idx[0]), num)
        
        Xs.extend(X[idx])    
        Ys.extend(Y[idx])    
    return np.array(Xs), np.array(Ys)


class encoder(nn.Module): 
    def __init__(self, x_dim=1, hid_dim=32, z_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            self.conv_block(x_dim, hid_dim, 15),
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


