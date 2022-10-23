#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 18:08:27 2022

@author: arshdeep
"""
#%% layer-wise important filter index computation. 

import numpy as np
from time import process_time
from scipy.spatial import distance
# import os
import matplotlib.pyplot as plt
# from sklearn import datasets, svm
# from sklearn.kernel_approximation import Nystroem


#%% load pre-trained weights

W_dcas=list(np.load('/home/arshdeep/Pruning/SPL/VGGish_applied_acoustics/cosine_similarity/Baseline_65/best_weights_VGGish.npy',allow_pickle=True))
 # extract pre-trained layer wise weights.
#%% Please note that the indexes {0,2,4,6,8,10} in "W_dcas" represent first to sixth convolutional layer in VGGish_Net.


# W_dcas = list(np.load('/home/arshdeep/Pruning/SPL/ICLR_anonymize_code_test/DCASE21_Net/unpruned_model_DCASE21_Net_48.58/unpruned_model_weights.npy',allow_pickle=True))
# Rank-1 approximation of a given data.
#%% Please note that the indexes {0,6,12} in "W_dcas" represent first, second and third convolutional layer in DCASE 2021 TASK1A baseline network.


#%% 
def rank1_apporx(data):
    u,w,v= np.linalg.svd(data)
    M = np.matmul(np.reshape(u[:,0],(-1,1)),np.reshape(v[0,:],(1,-1)))
    M_prototype = M[:,0]/np.linalg.norm(M[:,0],2)
    return M_prototype

def compareList(l1,l2):
   l1.sort()
   l2.sort()
   if(l1==l2):
      return "Equal"
   else:
      return "Non equal"

def rankK_apporx(data,k):
    u,s,v= np.linalg.svd(data)
    U_k = u[:,:k]
    S_k = np.diag(s[:k])
    Vh_k = v[:k]
    rank_k_approximation = np.dot(U_k, np.dot(S_k, Vh_k))
    return rank_k_approximation


def nystrom_approx(N,d,m,k):
    C = np.zeros((d,m))
    for i in range(d):
        for j in range(m):
            C[i,j] = C[i,j] + np.dot(N[:,i],N[:,j])#distance.cosine(N[:,i],N[:,j])##distance.cosine(N[:,i],N[:,j])#np.dot(N[:,i],N[:,j])   #
    W = C[0:m,0:m]   
    W_K = rankK_apporx(W, k)
    W_inv = np.linalg.pinv(W_K)
    S_apprx = np.matmul(C, np.matmul(W_inv, C.T))             
    return S_apprx




 # choose 0,2,4,6,8,10
 
def CS_Pruning(Z):

    a,b,c,d=np.shape(Z) # "d" is number of filters, "c" is number of channels, "a" and "b" represent length of the filter.
    
    A= np.reshape(Z,(-1,c,d)) # reshape filters
    
    
    #%% Approximate each filter using rank-1 approximation
    
    N = np.zeros((a*b,d)) 
    
    for i in range(d):
        data= A[:,:,i]
        N[:,i]=rank1_apporx(data)
    
    #%% pair-wise similarity calculation in filter space approximated by Rank-1 method.
    W_1= np.zeros((d,d))
    
    for i in range(d):
        for j in range(d):
            W_1[i,j] = W_1[i,j] + distance.cosine(N[:,i],N[:,j]) #np.dot(N[:,i],N[:,j])#
    
    # W = np.ones((d,d)) - W_1
    W =np.around(W_1, decimals =5)
    
    #%% store closest filter to a given filter and store each pair with the closest distance. 
    Q=[]
    S=[]
    for i in range(np.shape(W)[0]):
        n=np.argsort(W[i,:])[1]
        Q.append([i,n,W[i,n]])  # store closest pairs with their distance.
        S.append(W[i,n])   # store closest distance for each filter (ordered pairwise distance)
        
            
    #%%
    Q_sort=[]
    q=list(np.argsort(S)) # save the indexes of filters with closest pairwise distance.
            
    for i in q: 
        Q_sort.append(Q[i]) # sort closest filter pairs.
        
        
    #%% select important filter indexes.
    
    imp_list=[]
    red_list=[]
    
    for i in range(np.shape(W)[0]): 
        index_imp = Q_sort[i][0]
        index_red = Q_sort[i][1]
        if index_imp not in red_list:
            imp_list.append(index_imp)
            red_list.append(index_red)
    return imp_list, W, N , q, Q_sort


#%%


def Efficient_CS_Pruning(Z,m,k):

    a,b,c,d=np.shape(Z) # "d" is number of filters, "c" is number of channels, "a" and "b" represent length of the filter.
    
    A= np.reshape(Z,(-1,c,d)) # reshape filters
    
    
    #%% Approximate each filter using rank-1 approximation
    
    N = np.zeros((a*b,d)) 
    
    for i in range(d):
        data= A[:,:,i]
        N[:,i]=rank1_apporx(data)
    
    #%% pair-wise similarity calculation in filter space approximated by Rank-1 method.
    S_app = nystrom_approx(N,d,m,k) #np.zeros((d,d))
    W = np.ones((d,d)) - S_app
    W = np.around(W, decimals =5)
    # W.astype('float16')
    
    # for i in range(d):
    #     for j in range(l):
    #         W[i,j] = W[i,j] + distance.cosine(N[:,i],N[:,j])
    
    
    #%% store closest filter to a given filter and store each pair with the closest distance. 
    Q=[]
    S=[]
    for i in range(np.shape(W)[0]):
        n=np.argsort(W[i,:])[1]
        Q.append([i,n,W[i,n]])  # store closest pairs with their distance.
        S.append(W[i,n])   # store closest distance for each filter (ordered pairwise distance)
        
            
    #%%
    Q_sort=[]
    q=list(np.argsort(S)) # save the indexes of filters with closest pairwise distance.
            
    for i in q: 
        Q_sort.append(Q[i]) # sort closest filter pairs.
        
        
    #%% select important filter indexes.
    
    imp_list=[]
    red_list=[]
    
    for i in range(np.shape(W)[0]): 
        index_imp = Q_sort[i][0]
        index_red = Q_sort[i][1]
        if index_imp not in red_list:
            imp_list.append(index_imp)
            red_list.append(index_red)
    return imp_list, W, N, q, Q_sort
        
#%%

Z= W_dcas[0]  # Select index of the layer (For DCASE 2021 network, index for C1, C2 and C3 layers are 0, 6 and 12. FOr VGGish, indexes are {0,2,4,6,8}) for C1, C2, ..., C6 layers respectively.
error = []

opt_m = []
NYS_time = [ ]
No_Nys_time = []
imp_list, W , N, q, Q = CS_Pruning(Z)  # Groundtruth,  list of important filters generated using complete distance matrix.
for  t in range(1,np.shape(Z)[3]):
    
    m = t
    
    k= m


    t1_start = process_time()
    imp_list_NYS, W_NYS , N_NYS, q_NYS, Q_NYS = Efficient_CS_Pruning(Z,m,k)
    t1_end = process_time()
    diff_NYS = t1_end - t1_start


    # t1_start = process_time()
    # imp_list, W , N, q, Q = CS_Pruning(Z)
    # t1_end = process_time()
    # diff = t1_end - t1_start
    error.append(np.linalg.norm(W-W_NYS,2))
    print(diff_NYS, 'NYS...')
    # print(diff,'.....wihtout')
    NYS_time.append(diff_NYS)
    # No_Nys_time.append(diff)
    print(len(imp_list_NYS), 'NYS_list')
    # print(len(imp_list), 'CS_list')
    # print(compareList(imp_list,imp_list_NYS))
    if compareList(imp_list,imp_list_NYS) == 'Equal':
        opt_m.append(k)
     
        

print('optimal_m is  ',  opt_m[0])
print(NYS_time)#np.average(NYS_time), '    AVERAGE TIME OVER 100')
plt.plot(error)
       
