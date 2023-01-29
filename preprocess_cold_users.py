# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 16:30:06 2023

@author: ASUS
"""

import pandas as pd
rating_count=100000
# Read the CSV file into a dataframe
df = pd.read_csv('data/edited_rating.csv')

# Group the dataframe by the user column
grouped = df.groupby('userId')

# Use the filter function to only keep groups where the size is equal to 3
repeated_times = grouped.filter(lambda x: len(x) < 25)

# select the first 100 rows
df_cold_users=repeated_times.head(rating_count)
df_cold_users.to_csv('data/cold_users.csv')

# Filter the user_counts dataframe to only include users that appear between 10 and 20 times
user_counts_between_25_and_50 = grouped.filter(lambda x: len(x) > 25 & len(x)<50)

# Select only the users that are between 10 and 20 times
users_between_25_and_50 = user_counts_between_25_and_50.index
df_less_cold_users=repeated_times.head(rating_count)
df_less_cold_users.to_csv('data/less_cold_users.csv')