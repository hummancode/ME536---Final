# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 23:57:10 2023

@author: ASUS
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import keras
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from keras.layers import Dropout, BatchNormalization, Activation
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
def make_square_axes(ax):

    ax.set_aspect(1 / ax.get_data_ratio())
def heatmap2d(arr: np.ndarray):
    #plt.figure(1, figsize=(4, 6))
    plt.imshow(arr, cmap='seismic')
    plt.ylim([0, 300])
    #plt.gca().set_box_aspect(1)
    make_square_axes(plt.gca())
    plt.xlabel("User ids that are closest neighbor to given user ids")
    plt.ylabel("Users ids (picky users are colored) ")

    
    plt.savefig('visuals/picky_users.png', dpi=1200)
    plt.show()
def track_novelty(df,new_list):
    user_ids=df['userId'].unique()
    y_out=np.zeros([len(user_ids),2])
    
    for i in range(len(user_ids)):
        item=user_ids[i]
        data={'userId':[item]*len(new_list), 'movie_idx': new_list, 'rating': [0]*len(new_list)}
        df_new=pd.DataFrame(data )
        y=model.predict(x=[df_new.userId.values, df_new.movie_idx.values])
        y_out[i][0]=y.mean()
        y_out[i][1]=item
    indices= np.where(y_out[:,0] <-1)
    return y_out[indices][:,1] #return picky users


with open('data/new_user_id_map.pkl', 'rb') as f:
    new_user_id_map = pickle.load(f)
with open('data/new_movie_id_map.pkl', 'rb') as f:
    new_movie_id_map= pickle.load(f)
with open('data/neighbor_matrix', 'rb') as fp:
       neighbors_matrix= pickle.load(fp)
reversed_movie_map=dict(reversed(list(new_movie_id_map.items())))
movie_list=reversed_movie_map.keys()
model = keras.models.load_model('models/resnet')
df2=pd.read_csv('data/cold_test.csv')
df3=pd.read_csv('data/less_cold_test.csv')


    


picky_users_colder=track_novelty(df2,movie_list)
new_picky_colder=[new_user_id_map[i] for i in picky_users_colder]
picky_users_cold=track_novelty(df3,movie_list)
new_picky_cold=[new_user_id_map[i] for i in picky_users_cold]
pickies=new_picky_cold+new_picky_colder

n = len(neighbors_matrix)
relation_matrix=np.zeros([n,n])
Social=True
for i in range(n):
        if i in pickies: #if our user is picky(which is the novelty)
            if len(neighbors_matrix[i])>24: #if our user has many neighbors
                Social=True
            else: 
                Social=False
            for m in range(n):
                if Social:
                    color=0.15
                else: 
                    color=-0.15
                relation_matrix[i,m]=color
            for k in range(len(neighbors_matrix[i])):
                j=neighbors_matrix[i][k][1] #id of user that is neighbor to i 
                relation_matrix[i,j]=int(np.sign(color))
relation_matrix[np.isnan(relation_matrix)] = 0
heatmap2d(relation_matrix) # such that picky users are divided into two groups and shown     
