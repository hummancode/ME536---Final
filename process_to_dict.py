# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 16:14:35 2022

@author: ASUS
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

df1=pd.read_csv('data/small_rating_normal.csv')
df2=pd.read_csv('data/cold_users.csv')
df3=pd.read_csv('data/less_cold_users.csv')
def process_to_json(df,name,isTest=False):
    name=str('dict/'+name)
    N=df.userId.max()+1
    M=df.movie_idx.max()+1
    
    df=shuffle(df)
    if isTest:
        cutoff_ratio=0.001
    else:
        cutoff_ratio=0.8
    
    cutoff=int(cutoff_ratio*len(df))
    df_train=df.iloc[:cutoff]
    df_test=df.iloc[cutoff:]
    
    user2movie={}
    movie2user={}
    usermovie2rating={}

    def update_user2movie_and_movie2user(row):

            
        i=int(row.userId)
        j=int(row.movie_idx)
        if i not in user2movie:
            user2movie[i]=[j]
        else:
            user2movie[i].append(j)
        if j not in movie2user:
            movie2user[j]=[i]
        else:
            movie2user[j].append(i)
        
        usermovie2rating[(i,j)]=row.rating
    df_train.apply(update_user2movie_and_movie2user,axis=1)
    
    usermovie2rating_test={}
    print("Calling : update_usermovie2rating_test")

    def update_usermovie2rating_test(row):

        i=int(row.userId)
        j=int(row.movie_idx)
        usermovie2rating_test[(i,j)]=row.rating
    df_test.apply(update_usermovie2rating_test,axis=1)
    if not isTest: 
        with open(str(name+'user2movie.json'),'wb') as f:
            pickle.dump(user2movie,f)
        with open(str(name+'movie2user.json'),'wb') as f:
            pickle.dump(movie2user,f)
        with open(str(name+'usermovie2rating.json'),'wb') as f:
            pickle.dump(usermovie2rating,f)
    with open(str(name+'usermovie2rating_test.json'),'wb') as f:
        pickle.dump(usermovie2rating_test,f)
process_to_json(df1,'main_',False)
process_to_json(df2,'colder_',True)
process_to_json(df3,'cold_',True)