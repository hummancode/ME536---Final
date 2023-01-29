# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 16:43:35 2022

@author: ASUS
"""

import pickle 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle 
from datetime import datetime
from sortedcontainers import SortedList
import os
if not os.path.exists('dict/main_user2movie.json') or\
   not os.path.exists('dict/main_movie2user.json') or\
   not os.path.exists('dict/main_usermovie2rating.json') or\
   not os.path.exists('dict/main_usermovie2rating_test.json'):
   import process_to_dict
with open('dict/main_user2movie.json','rb') as f:
    user2movie=pickle.load(f)
with open('dict/main_movie2user.json','rb') as f:
    movie2user=pickle.load(f)
with open('dict/main_usermovie2rating.json','rb') as f:
    usermovie2rating=pickle.load(f)
with open('dict/main_usermovie2rating_test.json','rb') as f:
    usermovie2rating_test=pickle.load(f)
with open('dict/cold_usermovie2rating_test.json','rb') as f:
    usermovie2rating_cold_test=pickle.load(f)
with open('dict/colder_usermovie2rating_test.json','rb') as f:
    usermovie2rating_colder_test=pickle.load(f)
#the test set may contain movies that does not exist in train set, so take both possibilities with m1 and m2
N=np.max(list(user2movie.keys()))+1
m1=np.max(list(movie2user.keys()))
m2=np.max([m for (u,m),r in usermovie2rating_test.items()])
M=max(m1,m2)+1
print("N:",N,"M:",M)



K=25 #number of neighbors that will considered at the last
limit=5 #number of common movies in order to consider users neighbor
neighbors=[] #store neighbors in the list
averages=[] #each users average rating
deviations=[] #each users deviation
for i in range(N): # for userId in N which is max user count
    movies_i=user2movie[i]
    movies_i_set=set(movies_i)
    #calculate average and deviations
    ratings_i={movie:usermovie2rating[(i,movie)] for movie in movies_i}
    avg_i=np.mean(list(ratings_i.values()))
    dev_i={movie:(rating-avg_i) for movie,rating in ratings_i.items()}
    dev_i_values=np.array(list(dev_i.values()))
    sigma_i=np.sqrt(dev_i_values.dot(dev_i_values))
    
    averages.append(avg_i)
    deviations.append(dev_i)
    sl=SortedList()
    for j in range(N):
        if j!=i:
            movies_j=user2movie[j]
            movies_j_set=set(movies_j)
            common_movies=(movies_i_set & movies_j_set)
            if len(common_movies)>limit:
                ratings_j={movie:usermovie2rating[(j,movie)] for movie in movies_j}
                avg_j=np.mean(list(ratings_j.values()))
                dev_j={movie:(rating-avg_j)for movie, rating in ratings_j.items()}
                dev_j_values=np.array(list(dev_j.values()))
                sigma_j=np.sqrt(dev_j_values.dot(dev_j_values))
                
                #use cosine similarity to calculate weights
                numerator=sum(dev_i[m]*dev_j[m] for m in common_movies)
                w_ij=numerator/(sigma_i*sigma_j)
                
                #sort all neighbors according to weigght
                sl.add((-w_ij,j))
                if len(sl)>K: #only take first 25 neighbors
                    del sl[-1]
    neighbors.append(sl)

    if i%1==0:
        print(i)

def predict(i,m):
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

def predict_lists(items):
    predictions=[]
    targets=[]
    for(i,m),target in items:
        prediction=predict(i,m)
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
train_predictions,train_targets = predict_lists(usermovie2rating.items())
test_predictions,test_targets = predict_lists(usermovie2rating_test.items())
cold_predictions,cold_targets = predict_lists(usermovie2rating_cold_test.items())
colder_predictions,colder_targets = predict_lists(usermovie2rating_colder_test.items())

print('train mae:',mae(train_predictions,train_targets))
print('test mae:',mae(test_predictions,test_targets))
print('cold test mae:',mae(cold_predictions,cold_targets))
print('colder test mae:',mae(colder_predictions,colder_targets))
    
print('train mse:',mse(train_predictions,train_targets))
print('test mse:',mse(test_predictions,test_targets))
print('cold test mse:',mse(cold_predictions,cold_targets))
print('colder test mse:',mse(colder_predictions,colder_targets))
        
with open('data/neighbor_matrix', 'wb') as fp:
    pickle.dump(neighbors, fp)    
    
    
with open('data/averages_matrix', 'wb') as fp:
    pickle.dump(averages, fp)   
    
with open('data/deviations_matrix', 'wb') as fp:
    pickle.dump(deviations, fp)   
   
    
    
    
    
