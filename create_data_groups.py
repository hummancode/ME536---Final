# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 18:00:27 2023

@author: ASUS
"""
from __future__ import print_function, division

# Note: you may need to update your version of future
# sudo pip install -U future

import pickle
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
def split_df_half(df,name):
    # count the number of repetitions of all users
    counts = df[name].value_counts()
    # sort the users based on the number of repetitions
    counts = counts.sort_values()
    # create empty dataframe
    df1 = pd.DataFrame(columns=df.columns)
    df2 = pd.DataFrame(columns=df.columns)
    
    #iterate through the users
    for user in counts.index:
                if int(counts[user])>5:
                    df1 = df1.append(df[df[name]==user].iloc[int(counts[user]/2):,:])
         
                    df2 = df2.append(df[df[name]==user].iloc[:int(counts[user]/2),:])
    
    return df1,df2

rating_count=5000

# load in the data
df = pd.read_csv('data/edited_rating.csv')



# split into train and test

rating_count=5000
df=pd.read_csv('data/edited_rating.csv')
grouped = df.groupby('userId')

# Use the filter function to only keep groups where the size is equal to 3
repeated_times = grouped.filter(lambda x: len(x) < 25)

# select the first 100 rows
cold_users=repeated_times.head(rating_count).copy()


# Filter the user_counts dataframe to only include users that appear between 10 and 20 times
user_counts_between_25_and_50 = grouped.filter(lambda x: len(x) > 25 & len(x)<50)

# Select only the users that are between 10 and 20 times
users_between_25_and_50 = user_counts_between_25_and_50.index
less_cold_users=user_counts_between_25_and_50.head(rating_count)

less_cold_train,less_cold_test = split_df_half(less_cold_users,'userId')
cold_train,cold_test = split_df_half(cold_users,'userId')





#drop test cold data from the main data
df = df.loc[df.index.difference(cold_test.index)]
df= df.loc[df.index.difference(less_cold_test.index)]

cutoff = int(0.8*len(df))

df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]

df_train.to_csv('data/train_data.csv') 
df_test.to_csv('data/test_data.csv') 

cold_test.to_csv('data/cold_test.csv') 
less_cold_test.to_csv('data/less_cold_test.csv') 

cold_train.to_csv('data/cold_train.csv') 
less_cold_train.to_csv('data/less_cold_train.csv') 


