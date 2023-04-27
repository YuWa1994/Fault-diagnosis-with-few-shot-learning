# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 10:44:30 2021

@author: yuwa
"""

import numpy as np
from scipy import stats
import scipy.io as sio


# The frequency features are cited from 
# https://www.kaggle.com/code/oybekeraliev/frequency-domain-feature-extraction-methods


def x_peak(signal, axis=1, frame_size=None, hop_length=None):
    if frame_size is None: frame_size=len(signal)
    if hop_length is None: hop_length=1    
    signal = signal[:]
    peak = []
    for i in range(0, signal.shape[1], hop_length):
        L = signal[:,i:i+frame_size]
        current_peak = np.amax(np.abs(L), axis)
        peak.append(current_peak)    
    return np.array(peak)

def x_mean(signal, axis=1, frame_size=None, hop_length=None):
    if frame_size is None: frame_size=len(signal)
    if hop_length is None: hop_length=1    
    signal = signal[:]
    mean = []
    for i in range(0, signal.shape[1], hop_length):
        L = signal[:,i:i+frame_size]
        current_mean = np.mean(L, axis)
        mean.append(current_mean)    
    return np.array(mean)

def x_var(signal, axis=1, frame_size=None, hop_length=None):
    if frame_size is None: frame_size=len(signal)
    if hop_length is None: hop_length=1    
    signal = signal[:]
    var = []
    for i in range(0, signal.shape[1], hop_length):
        L = signal[:,i:i+frame_size]
        current_var = np.var(L, axis)
        var.append(current_var)    
    return np.array(var)    


def x_std(signal, axis=1, frame_size=None, hop_length=None):
    if frame_size is None: frame_size=len(signal)
    if hop_length is None: hop_length=1    
    signal = signal[:]
    std = []
    for i in range(0, signal.shape[1], hop_length):
        L = signal[:,i:i+frame_size]
        current_std = np.std(L, axis)
        std.append(current_std)    
    return np.array(std)    


def x_rms(signal, axis=1, frame_size=None, hop_length=None):
    if frame_size is None: frame_size=len(signal)
    if hop_length is None: hop_length=1    
    signal = signal[:]
    rms = []
    for i in range(0, signal.shape[1], hop_length):
        L = signal[:,i:i+frame_size]
        current_rms = np.sqrt(np.mean(np.square(L), axis))
        rms.append(current_rms)    
    return np.array(rms)    


def x_skew(signal, axis=1, frame_size=None, hop_length=None):
    if frame_size is None: frame_size=len(signal)
    if hop_length is None: hop_length=1    
    signal = signal[:]
    skew = []
    for i in range(0, signal.shape[1], hop_length):
        L = signal[:,i:i+frame_size]
        current_skew = stats.skew(L,axis)
        skew.append(current_skew)    
    return np.array(skew)    


def x_kurt(signal, axis=1, frame_size=None, hop_length=None):
    if frame_size is None: frame_size=len(signal)
    if hop_length is None: hop_length=1    
    signal = signal[:]
    kurt = []
    for i in range(0, signal.shape[1], hop_length):
        L = signal[:,i:i+frame_size]
        current_kurt = stats.kurtosis(L,axis)
        kurt.append(current_kurt)    
    return np.array(kurt)    


def x_crest(signal, axis=1, frame_size=None, hop_length=None):
    if frame_size is None: frame_size=len(signal)
    if hop_length is None: hop_length=1    
    signal = signal[:]
    crest = []
    for i in range(0, signal.shape[1], hop_length):
        L = signal[:,i:i+frame_size]
        current_crest = (x_peak(L, axis)/x_rms(L, axis))
        crest.append(current_crest)    
    return np.array(crest)    

def x_impulse(signal, axis=1, frame_size=None, hop_length=None):
    if frame_size is None: frame_size=len(signal)
    if hop_length is None: hop_length=1    
    signal = signal[:]
    impulse = []
    for i in range(0, signal.shape[1], hop_length):
        L = signal[:,i:i+frame_size]
        current_impulse = (x_peak(L, axis)/np.mean(np.abs(L), axis))
        impulse.append(current_impulse)    
    return np.array(impulse)    


def x_clearance(signal, axis=1, frame_size=None, hop_length=None):
    if frame_size is None: frame_size=len(signal)
    if hop_length is None: hop_length=1    
    signal = signal[:]
    clearance = []
    for i in range(0, signal.shape[1], hop_length):
        L = signal[:,i:i+frame_size]
        current_clearance = (x_peak(L, axis)/np.std(L, axis))
        clearance.append(current_clearance)    
    return np.array(clearance)    


def x_shape(signal, axis=1, frame_size=None, hop_length=None):
    if frame_size is None: frame_size=len(signal)
    if hop_length is None: hop_length=1    
    signal = signal[:]
    shape = []
    for i in range(0, signal.shape[1], hop_length):
        L = signal[:,i:i+frame_size]
        current_shape = (x_rms(L, axis)/np.mean(np.abs(L), axis))
        shape.append(current_shape)    
    return np.array(shape)    

def x_margin(signal, axis=1, frame_size=None, hop_length=None):
    if frame_size is None: frame_size=len(signal)
    if hop_length is None: hop_length=1    
    signal = signal[:]
    margin = []
    for i in range(0, signal.shape[1], hop_length):
        L = signal[:,i:i+frame_size]
        current_margin = x_peak(L, axis)/np.square(np.mean(np.sqrt(np.abs(L)), axis))
        margin.append(current_margin)    
    return np.array(margin)    

def x_EE(signal, axis=1, frame_size=None, hop_length=None, numOfShortBlocks=20):
    if frame_size is None: frame_size=len(signal)
    if hop_length is None: hop_length=1    
    signal = signal[:]
    EE = []
    for i in range(0, signal.shape[1], hop_length):
        L = signal[:,i:i+frame_size]
        current_EE = EnergyEentropy(L, numOfShortBlocks)
        EE.append(current_EE)    
    return np.array(EE)    


def x_IE(signal, axis=1, frame_size=None, hop_length=None, numOfShortBlocks=20):
    if frame_size is None: frame_size=len(signal)
    if hop_length is None: hop_length=1    
    signal = signal[:]
    IE = []
    for i in range(0, signal.shape[1], hop_length):
        L = signal[:,i:i+frame_size]
        current_IE = histogram_entropy(L, numOfShortBlocks)
        IE.append(current_IE)    
    return np.array(IE)   



def ShannonEntropy(p, axis=1, eps=1e-8):
    # Calculate the Shannon's Entropy
    # p is a m*n sized probability matrix, m is the number of sample, n is the number of
    # element, the sum of each raw is 1.
    return -1*np.sum(p*np.log2(p+eps), axis)


def EnergyEentropy(x, numOfShortBlocks=20, eps=1e-8):
    EE = []
    for i in range(x.shape[0]):    
        _x = x[i,:]
        xLength = len(_x)
        subxLength = np.floor(xLength / numOfShortBlocks)
        
        if len(_x) != subxLength * numOfShortBlocks:
            _x = _x[0:int(subxLength* numOfShortBlocks)]
            
        E = np.sum(np.square(_x))
        # get sub-x:
        subX = _x.reshape(numOfShortBlocks, int(subxLength))
        
        # compute normalized sub-frame energies:
        P = np.sum(np.square(subX), 1) / (E+eps)
        
        # compute entropy of the normalized sub-frame energies:
        EE.append(-1*np.sum(P*np.log2(P+eps)))
    return EE


def histogram_entropy(x,splits=20, eps=1e-8):
    # the raw of x is sample number, the column of x is dimension
    IE = []
    for i in range(x.shape[0]):
        _x = x[i,:]
        out, _ = np.histogram(_x, bins=splits)
        P = out / np.sum(out)
        IE.append(-1*np.sum(P*np.log2(P+eps)))
    return IE


def cal_IEPF(X, dim=8):
    eps = 1E-10
    
    X_energy = np.sum(np.square(X), axis=1)
    X_energy = X_energy.reshape(-1,dim)
    
    T = np.sum(X_energy,axis=1).reshape(-1,1)
    pd = X_energy / T
    Entropy = -1*np.sum(pd * np.log2(pd+eps), axis=1)
    
    th = 1/dim
    maximum = -1*np.sum(np.tile(th * np.log2(th), dim))

    IEPF = -1*(Entropy - maximum)
    return IEPF



def get_mean_freq(signal, frame_size=None, hop_length=None):
    if frame_size is None: frame_size=len(signal)
    if hop_length is None: hop_length=1
    signal = signal[:]
    mean = []
    for i in range(0, signal.shape[1], hop_length):
        y = abs(np.fft.fft(signal[i:i+frame_size]/frame_size))[:int(frame_size/2)]
        current_mean = np.sum(y)/frame_size
        mean.append(current_mean)
    return np.array(mean)

def get_variance_freq(signal, frame_size=None, hop_length=None):
    if frame_size is None: frame_size=len(signal)
    if hop_length is None: hop_length=1    
    signal = signal[:]
    var = []
    for i in range(0, signal.shape[1], hop_length):
        L = frame_size
        y = abs(np.fft.fft(signal[i:i+frame_size]/L))[:int(L/2)]
        current_var = (np.sum((y - (np.sum(y)/frame_size))**2))/(frame_size-1)
        var.append(current_var)
    return np.array(var)


def get_third_freq(signal, frame_size=None, hop_length=None):
    if frame_size is None: frame_size=len(signal)
    if hop_length is None: hop_length=1
    signal = signal[:]
    third = []
    for i in range(0, signal.shape[1], hop_length):
        L = frame_size
        y = abs(np.fft.fft(signal[i:i+frame_size]/L))[:int(L/2)]
        current_third = (np.sum((y - (np.sum(y)/frame_size))**3))/(frame_size * (np.sqrt((np.sum((y - (np.sum(y)/frame_size))**2))/(frame_size-1)))**3)
        third.append(current_third)
    return np.array(third)

def get_forth_freq(signal, frame_size=None, hop_length=None):
    if frame_size is None: frame_size=len(signal)
    if hop_length is None: hop_length=1  
    signal = signal[:]
    forth = []
    for i in range(0, signal.shape[1], hop_length):
        L = frame_size
        y = abs(np.fft.fft(signal[i:i+frame_size]/L))[:int(L/2)]
        current_forth = (np.sum((y - (np.sum(y)/frame_size))**4))/(frame_size * ((np.sum((y - (np.sum(y)/frame_size))**2))/(frame_size-1))**2)
        forth.append(current_forth)
    return np.array(forth)


def get_grand_freq(signal, frame_size=None, hop_length=None, d=1):
    # d: Sample spacing (inverse of the sampling rate). Defaults to 1.
    if frame_size is None: frame_size=len(signal)
    if hop_length is None: hop_length=1    
    signal = signal[:]
    grand = []
    for i in range(0, signal.shape[1], hop_length):
        L = frame_size
        y = abs(np.fft.fft(signal[i:i+frame_size]/L))[:int(L/2)]
        f = np.fft.fftfreq (L, d)[:int(L/2)] 
        current_grand = np.sum(f * y)/np.sum(y)
        grand.append(current_grand)
    return np.array(grand)


def get_std_freq(signal, frame_size=None, hop_length=None, d=1):
    # d: Sample spacing (inverse of the sampling rate). Defaults to 1.
    if frame_size is None: frame_size=len(signal)
    if hop_length is None: hop_length=1    
    signal = signal[:]
    std = []
    for i in range(0, signal.shape[1], hop_length):
        y = abs(np.fft.fft(signal[i:i+frame_size]/frame_size))[:int(frame_size/2)]
        f = np.fft.fftfreq (frame_size, d)[:int(frame_size/2)] 
        current_std = np.sqrt(np.sum((f-(np.sum(f * y)/np.sum(y)))**2 * y)/frame_size)
        std.append(current_std)
    return np.array(std)


def get_Cfactor_freq(signal, frame_size=None, hop_length=None, d=1):
    # d: Sample spacing (inverse of the sampling rate). Defaults to 1.
    if frame_size is None: frame_size=len(signal)
    if hop_length is None: hop_length=1    
    signal = signal[:]
    cfactor = []
    for i in range(0, signal.shape[1], hop_length):
        y = abs(np.fft.fft(signal[i:i+frame_size]/frame_size))[:int(frame_size/2)]
        f = np.fft.fftfreq (frame_size, d)[:int(frame_size/2)] 
        current_cfactor = np.sqrt(np.sum(f**2 * y)/np.sum(y))
        cfactor.append(current_cfactor)
    return np.array(cfactor)


def get_Dfactor_freq(signal, frame_size=None, hop_length=None, d=1):
    # d: Sample spacing (inverse of the sampling rate). Defaults to 1.
    if frame_size is None: frame_size=len(signal)
    if hop_length is None: hop_length=1    
    signal = signal[:]
    dfactor = []
    for i in range(0, signal.shape[1], hop_length):
        y = abs(np.fft.fft(signal[i:i+frame_size]/frame_size))[:int(frame_size/2)]
        f = np.fft.fftfreq (frame_size, d)[:int(frame_size/2)] 
        current_dfactor = np.sqrt(np.sum(f**4 * y)/np.sum(f**2 * y))
        dfactor.append(current_dfactor)
    return np.array(dfactor)

def get_Efactor_freq(signal, frame_size=None, hop_length=None, d=1):
    # d: Sample spacing (inverse of the sampling rate). Defaults to 1.
    if frame_size is None: frame_size=len(signal)
    if hop_length is None: hop_length=1    
    signal = signal[:]
    efactor = []
    for i in range(0, signal.shape[1], hop_length):
        y = abs(np.fft.fft(signal[i:i+frame_size]/frame_size))[:int(frame_size/2)]
        f = np.fft.fftfreq (frame_size, d)[:int(frame_size/2)] 
        current_efactor = np.sqrt(np.sum(f**2 * y)/np.sqrt(np.sum(y) * np.sum(f**4 * y)))
        efactor.append(current_efactor)
    return np.array(efactor)


def get_Gfactor_freq(signal, frame_size=None, hop_length=None, d=1):
    # d: Sample spacing (inverse of the sampling rate). Defaults to 1.
    if frame_size is None: frame_size=len(signal)
    if hop_length is None: hop_length=1    
    signal = signal[:]
    gfactor = []
    for i in range(0, signal.shape[1], hop_length):
        y = abs(np.fft.fft(signal[i:i+frame_size]/frame_size))[:int(frame_size/2)]
        f = np.fft.fftfreq (frame_size, d)[:int(frame_size/2)] 
        current_gfactor = (np.sqrt(np.sum((f-(np.sum(f * y)/np.sum(y)))**2 * y)/frame_size))/(np.sum(f * y)/np.sum(y))
        gfactor.append(current_gfactor)
    return np.array(gfactor)


def get_third1_freq(signal, frame_size=None, hop_length=None, d=1):
    # d: Sample spacing (inverse of the sampling rate). Defaults to 1.
    if frame_size is None: frame_size=len(signal)
    if hop_length is None: hop_length=1    
    signal = signal[:]
    third1 = []
    for i in range(0, signal.shape[1], hop_length):
        y = abs(np.fft.fft(signal[i:i+frame_size]/frame_size))[:int(frame_size/2)]
        f = np.fft.fftfreq (frame_size, d)[:int(frame_size/2)] 
        current_third1 = np.sum((f - (np.sum(f * y)/np.sum(y)))**3 * y)/(frame_size * (np.sqrt(np.sum((f-(np.sum(f * y)/np.sum(y)))**2 * y)/frame_size))**3)
        third1.append(current_third1)
    return np.array(third1)


def get_forth1_freq(signal, frame_size=None, hop_length=None, d=1):
    # d: Sample spacing (inverse of the sampling rate). Defaults to 1.
    if frame_size is None: frame_size=len(signal)
    if hop_length is None: hop_length=1
    signal = signal[:]
    forth1 = []
    for i in range(0, signal.shape[1], hop_length):
        y = abs(np.fft.fft(signal[i:i+frame_size]/frame_size))[:int(frame_size/2)]
        f = np.fft.fftfreq (frame_size,d)[:int(frame_size/2)] 
        current_forth1 = np.sum((f - (np.sum(f * y)/np.sum(y)))**4 * y)/(frame_size * (np.sqrt(np.sum((f-(np.sum(f * y)/np.sum(y)))**2 * y)/frame_size))**4)
        forth1.append(current_forth1)
    return np.array(forth1)

def get_Hfactor_freq(signal, frame_size=None, hop_length=None, d=1):
    # d: Sample spacing (inverse of the sampling rate). Defaults to 1.
    if frame_size is None: frame_size=len(signal)
    if hop_length is None: hop_length=1    
    signal = signal[:]
    hfactor = []
    for i in range(0, signal.shape[1], hop_length):
        y = abs(np.fft.fft(signal[i:i+frame_size]/frame_size))[:int(frame_size/2)]
        f = np.fft.fftfreq (frame_size, d)[:int(frame_size/2)] 
        current_hfactor = np.sum(np.sqrt(abs(f - (np.sum(f * y)/np.sum(y)))) * y)/(frame_size * np.sqrt(np.sqrt(np.sum((f-(np.sum(f * y)/np.sum(y)))**2 * y)/frame_size)))
        hfactor.append(current_hfactor)
    return np.array(hfactor)


def get_Jfactor_freq(signal, frame_size=None, hop_length=None, d=1):
    # d: Sample spacing (inverse of the sampling rate). Defaults to 1.
    if frame_size is None: frame_size=len(signal)
    if hop_length is None: hop_length=1    
    signal = signal[:]
    jfactor = []
    for i in range(0, signal.shape[1], hop_length):
        y = abs(np.fft.fft(signal[i:i+frame_size]/frame_size))[:int(frame_size/2)]
        f = np.fft.fftfreq (frame_size, d)[:int(frame_size/2)] 
        current_jfactor = np.sum(np.sqrt(abs(f - (np.sum(f * y)/np.sum(y)))) * y)/(frame_size * np.sqrt(np.sqrt(np.sum((f-(np.sum(f * y)/np.sum(y)))**2 * y)/frame_size)))
        jfactor.append(current_jfactor)
    return np.array(jfactor)


def get_all_timefetures(signal, axis=1, frame_size=None, hop_length=None, feature_selector=None):
    if frame_size is None: frame_size=len(signal)
    if hop_length is None: hop_length=1        
    
    list_features_function = [x_peak, x_mean, x_var, 
                                  x_std, x_rms, x_skew, 
                                  x_kurt, x_crest, x_impulse, 
                                  x_clearance, x_shape, x_margin,
                                  x_EE, x_IE]  

    if feature_selector is not None: 
        list_features_function = list_features_function[feature_selector]
    
    time_features = []
    for func in list_features_function:
        f = func(signal, frame_size, hop_length)
        time_features.append(f)
    return time_features


def get_all_freqfetures(signal, frame_size=None, hop_length=None, d=1, feature_selector=None):
    if frame_size is None: frame_size=len(signal)
    if hop_length is None: hop_length=1        
    
    list_features_function = [get_mean_freq, get_variance_freq, get_third_freq, 
                                  get_forth_freq, get_grand_freq,get_std_freq, 
                                  get_Cfactor_freq, get_Dfactor_freq, get_Efactor_freq, 
                                  get_Gfactor_freq,get_third1_freq, get_forth1_freq, 
                                  get_Hfactor_freq, get_Jfactor_freq]    

    if feature_selector is not None: 
        list_features_function = list_features_function[feature_selector]
        
    freq_features = []
    for func in list_features_function:
        f = func(signal, frame_size, hop_length)
        freq_features.append(f)
    return freq_features










