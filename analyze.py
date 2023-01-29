# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 20:07:21 2023

@author: ASUS
"""

import pickle 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle 
from datetime import datetime

import matplotlib.pylab as plt

def heatmap2d(arr: np.ndarray):
    
    plt.imshow(arr, cmap='seismic')
    plt.xlabel("User ids that are closest neighbor to given user ids")
    plt.ylabel("User Ids")
    plt.colorbar()
    plt.savefig('visuals/heatmap.png', dpi=1200)
    plt.show()
def normalize(x):
    x_norm = (x-np.min(x))/(np.max(x)-np.min(x))
    return x_norm
def discrete(x):
    x=np.where( x>0, 1, x)
    
    x=np.where( x<0, -1, x)
    return x
with open('data/neighbor_matrix', 'rb') as fp:
       neighbors_matrix= pickle.load(fp)

#df = pd.read_csv('data/edited_rating.csv')
df=pd.read_csv('data/very_small_rating.csv') 
df = df.filter(['movieId','userId','rating'], axis=1)
n_a=df.to_numpy()
#x,y=np.meshgrid(n_a[:,0], n_a[:,1])

n = len(neighbors_matrix)
relation_matrix=np.zeros([n,n])


for i in range(n):
        for k in range(len(neighbors_matrix[i])):
            print(k)
            j=neighbors_matrix[i][k][1]
            relation_matrix[i,j]=-neighbors_matrix[i][k][0]
relation_matrix[np.isnan(relation_matrix)] = 0
heatmap2d(discrete(relation_matrix))