# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 11:08:29 2022

@author: ASUS
"""

import pandas as pd

df=pd.read_csv('data/rating.csv')

df.userId = df.userId-1
unique_movie_ids=set(df.movieId.values)
movie2idx={}
count=0
for movie_id in unique_movie_ids:
    movie2idx[movie_id]=count
    count+=1
df['movie_idx']=df.apply(lambda row: movie2idx[row.movieId], axis=1)

df=df.drop(columns=['timestamp'])
df.to_csv('data/edited_rating.csv')