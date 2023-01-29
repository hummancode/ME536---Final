# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 16:47:44 2023

@author: ASUS
"""
from __future__ import print_function, division
from builtins import range, input


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
with open('dict/cold_usermovie2rating_test.json','rb') as f:
    usermovie2rating_cold_test=pickle.load(f)
with open('dict/colder_usermovie2rating_test.json','rb') as f:
    usermovie2rating_colder_test=pickle.load(f)

with open('data/neighbor_matrix', 'rb') as fp:
       neighbors_matrix= pickle.load(fp)
with open('data/averages_matrix', 'rb') as fp:
       averages_matrix= pickle.load(fp)       
with open('data/deviations_matrix', 'rb') as fp:
      deviations_matrix= pickle.load(fp)       
with open('dict/main_usermovie2rating_test.json','rb') as f:
    usermovie2rating_test=pickle.load(f)       
def predict(i,m,neighbors,averages,deviations):
    numerator=0
    denominator=0
    
    for neg_w,j in neighbors[i]:
        try:
            numerator+=-neg_w*deviations[j][m]
            denominator+=abs(neg_w)
            
        except KeyError:
            
            
            pass
        if denominator==0:
            prediction=averages[i]
        else:
            prediction=numerator/denominator+averages[i]
            
        prediction=min(5,prediction)
        prediction=max(0.5,prediction)
        return prediction

def predict_lists(items,neighbor_maxtrix,averages_maxtrix,deviations_maxtrix):
    predictions=[]
    targets=[]
    for(i,m),target in items:
        prediction=predict(i,m,neighbor_maxtrix,averages_maxtrix,deviations_maxtrix)
        predictions.append(prediction)
        targets.append(target)
    return predictions, targets

def mse(p,t):
    
    df1 = pd.DataFrame (p, columns = ['p'])
    df2 = pd.DataFrame (t, columns = ['p'])
    mu=df1.mean()
    p1 =  df1.fillna(mu)# df1.dropna().reset_index(drop=True)
 
    t1= df2.dropna() #df2.dropna().reset_index(drop=True)  
    return float(((p1-t1)**2).mean())

def mae(p,t):
    df1 = pd.DataFrame (p, columns = ['p'])
    df2 = pd.DataFrame (t, columns = ['t'])

    p1 = df1.dropna().iloc[0,0]
 
    t1= df2.dropna().iloc[0,0]    

    return np.mean(abs(p1-t1))       
       
print('Resnet results')
       
model = keras.models.load_model('models/resnet')
df2=pd.read_csv('data/cold_test.csv')
df3=pd.read_csv('data/less_cold_test.csv')
df = pd.read_csv('data/test_data.csv')

df_test = shuffle(df)

mu1=df2.rating.mean()
mu2 = df3.rating.mean()
print('cold test prediction')
scores=model.evaluate(x=[df3.userId.values, df3.movie_idx.values],
y=df3.rating.values-mu2 )
print('colder test prediction')
scores=model.evaluate(x=[df2.userId.values, df2.movie_idx.values],
y=df2.rating.values-mu1 )



### use neigbors -from collabarative filter 
test_predictions,test_targets = predict_lists(usermovie2rating_test.items() ,neighbors_matrix,averages_matrix,deviations_matrix)
cold_predictions,cold_targets = predict_lists(usermovie2rating_cold_test.items(),      neighbors_matrix,averages_matrix,deviations_matrix)
colder_predictions,colder_targets = predict_lists(usermovie2rating_colder_test.items(), neighbors_matrix,averages_matrix,deviations_matrix)
print('Collabartive filter results')

print('cold test mse:',mse(cold_predictions,cold_targets))
print('colder test mse:',mse(colder_predictions,colder_targets))
