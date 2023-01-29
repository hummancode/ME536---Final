# ME536---Final
Reccomender system using movielens dataset
### DATASET
##### Movilens 20 M Dataset --> rating.csv is used
Link: https://grouplens.org/datasets/movielens/20m/
rating.csv in data/ folder, for the project!
#### predata_analysis.py
Plots infographics graphs in the presentation and saves them into the visual folder.
## Instructions 
#### preprocess_first_step.py 
This first process drops some unnecessary columns such as time stamp and saves new file to data/edited_rating.csv
#### create_data_groups.py
this code takes edited rating and creates user groups such as cold users. Note that half of the cold users are for training and half is for testing. (for the future).

### Preprocess
#### preprocess_to_shrink_new.py 
(other preprocess_to_shrink.py is used in older versions, loaded for the sake of inspection)
reads cold users test/train csv files and train_data.csv files:
a small subset of user-movie matrix is created for the inspection in user-user collabarative filtering including cold users. These users and movies are re-indexed with index_maps saved to data folder to be reused in the future (such as detecting novelty and applying connection between k-means algorithm and residual learning network). 
#### process_to_dict.py 
Our relation matrix is sparse, so they can be repsented in a efficient way using dictionaries.
https://stackoverflow.com/questions/46114309/python-convert-a-sparse-matrix-to-json
Here are important notice about data:
* Previosuly obtained cold_users and less_cold_users are read from csv. Note that these files are for testing in collabarative filter, so in the releated function they are only used for user2movie_testing. 
* a small subset of rating data to be used in collabarative filtering, is converted into dictionary relations. 

### Collabarative Filtering
#### collab_filter.py
https://surprise.readthedocs.io/en/stable/getting_started.html#getting-started
https://realpython.com/build-recommendation-engine-collaborative-filtering/
https://towardsdatascience.com/various-implementations-of-collaborative-filtering-100385c6dfe0
https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b
Key points: 
* K: maximum neighbor count boundary
* limit: minimum number of common movies to be considered neighbor
* neighbors matrix: for a given index(which coresspends to a userId), a sorted list with weights and user Ids given. So for a given userId, it's neighbor's with relative weights are stored. This matrix is saved to data/ to be used later. Weights are taken negative to make sorting operating easier. 
* With deviations and average rating of each user considered, prediction can be made using neighbors matrix.

### Training Residual learning network
For matrix factorization in keras, following article is followed:
https://medium.com/@yashsonar213/how-to-build-a-recommendation-systems-matrix-factorization-using-keras-778931cc666f
#### train_resnet.py
training.py is vanilla neural network, which is not used in the project directly. Matrix factorization is used as main line and neural network as residual to find nonlinearities in the user-movie ratings to hopefully adress cold start problem. Training and loss function with respect to epochs are saved as image.

## Cold Start Problem 
Both residual learning model and collabarative filter neighborhoox matrix is saved. And cold users are the common theme in both methods even though sampling is smaller in user-user collabarative filter and larger in residual learning. How is the suggestion performance for cold start users? 
 * Note that in the training data same amount of information is added via cold users training data which is added to training data in grouping data section.
 It shows that while validation/training scores are not very far, residual learning has significantly low error. The test is done in the file:
#### test.py
Note that sufficient learning model is saved, and other small data is already provided. So user should be able to run this step without running previous steps. 
Relevant files are loaded and thus cold users predictions error are compared between collabarative filter and residual learning. Scores are printed. 

## Detecting Novelty
Cold Start Problem, I believe is a novelty-type problem since a specific user is new; and detecting its similarity with other users/data requires methods such as SVD etc. However for the sake of assignments task, a novelty detection algorithm is also formed. It is noticed to residual learning is more succesfull then collabarative filter in cold user prediction. So for our cold users, can we try to assume their future behaviour?
For this both collabarative filter method (neighbor matrix) and residual learning is used. So all previously saved files such as neighbors_matrix, id maps etc. are useful to link user and movies in residual learning and collabarative filtering. 

#### detecting novelty.py 

Using residual learning, all movie ids exist in neighbor matrix is is suggested to model to predict cold users ratings. If these ratings are very low, we deduce that these users are outliers in the meaningful sense that they like not broad range of movie categoires, only specific movies hence they are picky. Mean threshold is taken as -1 for this purpose. 
Whats more interesting is that this novel group is divided into two, social ones and alone ones. It is interesting that there is no middle ground, alone group has really few neighbors while social groups has near 25 neighbors with the neighborhood matrix formed previously. Alone users are more common, which is actually predictable since their likings are outlier. That's interesting that a large network model produces meaninful results and interpreation in a small subsection of data. 


