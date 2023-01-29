# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 17:33:42 2023

@author: ASUS
"""

import pandas as pd

# Read the CSV file into a dataframe
df = pd.read_csv('data/ml-1m/ratings.dat')

# Get the number of times each user appears in the dataframe
user_counts = df['userId'].value_counts()

# Get the minimum number of times a user appears
min_repeatition = user_counts.min()
print(min_repeatition)