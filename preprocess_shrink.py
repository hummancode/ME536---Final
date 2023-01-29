# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 15:47:59 2022

@author: ASUS
"""

import pickle 
import numpy as np
import pandas as pd
import collections

from collections import Counter
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

#######Select most common data
N=df.userId.max()+1
M=df.movie_idx.max()+1

user_ids_count=Counter(df.userId)
movie_ids_count=Counter(df.movie_idx)

n=int(1500*0.2)
m=int(300*0.2)

user_ids=[u for u, c in user_ids_count.most_common(n)]
movie_ids=[m for m, c in movie_ids_count.most_common(m)]
df_small_tops = df[df.userId.isin(user_ids) & df.movie_idx.isin(movie_ids)].copy()
##merge cold start and most common data
frames = [less_cold_train, cold_train, df_small_tops]
frames2 = [less_cold_train,less_cold_test, cold_train,cold_test, df_small_tops]
df_smll = pd.concat(frames)
df_all=pd.concat(frames2)
#merge ids
user_ids=df_all['userId'].unique().tolist() #new user ids of modified dataframe
movie_ids=df_all['movie_idx'].unique().tolist() #new user ids of modified dataframe

### assign new ids
new_user_id_map = {}
i = 0
for old in user_ids:
  new_user_id_map[old] = i
  i += 1
print("i:", i)

new_movie_id_map = {}
j = 0
for old in movie_ids:
  new_movie_id_map[old] = j
  j += 1
print("j:", j)

print("Setting new ids")
df_small=df_smll.copy()
df_cold_users=cold_test.copy()
df_less_cold_users=less_cold_test.copy()
    

df_small.loc[:, 'userId'] = df_small.apply(lambda row: new_user_id_map[row.userId], axis=1)
df_small.loc[:, 'movie_idx'] = df_small.apply(lambda row: new_movie_id_map[row.movie_idx], axis=1)

df_cold_users.loc[:, 'userId']= df_cold_users.apply(lambda row: new_user_id_map[row.userId], axis=1)
df_cold_users.loc[:, 'movie_idx']= df_cold_users.apply(lambda row: new_movie_id_map[row.movie_idx], axis=1)

df_less_cold_users.loc[:, 'userId']= df_less_cold_users.apply(lambda row: new_user_id_map[row.userId], axis=1)

df_less_cold_users.loc[:, 'movie_idx']= df_less_cold_users.apply(lambda row: new_movie_id_map[row.movie_idx], axis=1)


#writing new user groups 
df_small.to_csv('data/small_rating_normal.csv')
df_cold_users.to_csv('data/cold_users.csv')
df_less_cold_users.to_csv('data/less_cold_users.csv')

















