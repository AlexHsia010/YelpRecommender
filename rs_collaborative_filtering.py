#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import json

# Model based recommender system using surprise
from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise import SVD
from surprise.model_selection import KFold


class MemoryBasedRecommender():

    def __init__(self, n_neighbor):

        self.n_neighbor = n_neighbor
        self.df_utility = None 
        self.df_utility_values = None
        self.item_sim_mat = None 


    def utility_matrix(self, df):

        # Get business_id, user_id, stars for recommender
        df_stars = df[['business_id', 'user_id', 'stars']]
        df_user_counts = df_stars['user_id'].value_counts()
        # user counts larger than 5 is defined as active
        df_users = df_user_counts[df_user_counts > 50]
        # filter the rating table
        df_cleaned = df_stars.set_index('user_id').loc[df_users.index,:].reset_index()

        # convert to document, word like matrix
        self.df_utility = pd.pivot_table(data=df_cleaned,
                                    values='stars', # fill with stars
                                    index='user_id', # rows
                                    columns='business_id', # columns
                                    fill_value=0) # fill missings
        self.df_utility_values = self.df_utility.values



    def _predict_one_user(self, user_id):
        
        n_users = self.df_utility.shape[0]
        n_items = self.df_utility.shape[1]

        # argsort to get the index of sorted similarity
        least_to_most_sim_indexes = np.argsort(self.item_sim_mat, axis=1)
        # number of neighborhoods (hyperparameters)
        neighborhoods = least_to_most_sim_indexes[:, -self.n_neighbor-1:-1]
        # get items that have been rated by this user
        items_rated_by_this_user = self.df_utility.loc[user_id,:].values.nonzero()[0]

        # Just initializing so we have somewhere to put rating preds
        out = np.zeros(n_items)
        for item_to_rate in tqdm(range(n_items)):
            relevant_items_idx = np.intersect1d(neighborhoods[item_to_rate],
                                            items_rated_by_this_user,
                                            assume_unique=True)  # assume_unique speeds up intersection op
            relevant_items = self.df_utility.columns[relevant_items_idx].values
            out[item_to_rate] = self.df_utility.loc[user_id, relevant_items].values.dot(
                self.item_sim_mat[item_to_rate, relevant_items_idx].T) / (self.item_sim_mat[item_to_rate, relevant_items_idx].sum()+0.0001)
        
        pred_ratings = np.nan_to_num(out)

        return pred_ratings


    def predict(self):

        n_users = self.df_utility.shape[0]
        pred_mat = []

        for user_id in tqdm(self.df_utility.index):
            pred_ratings = self._predict_one_user(user_id)
            pred_mat.append(pred_ratings)

        return np.array(pred_mat)


    def evaluation(self):

        # split self.item_sim_mat into train and validate
        # simply mask some ratings (record and then set them to 0)
        # get index of nonzero ratings 
        nonzero_index = np.array(self.df_utility_values.nonzero())
        # total number of nonzero ratings
        n_nonzero = len(nonzero_index[0])
        # shuffle and split 20% of total as test
        idx = list(range(n_nonzero))
        np.random.shuffle(idx)
        # get index of index
        test_idx_idx = idx[:int(0.2 * n_nonzero)]
        train_idx_idx = idx[int(0.2 * n_nonzero):]
        # get index 
        test_index = tuple(nonzero_index[:, test_idx_idx])
        train_index = tuple(nonzero_index[:, train_idx_idx])
        # get test set
        y_test = self.df_utility_values[test_index]
        # set test index rating to 0
        self.df_utility_values[test_index] = 0
        # Item-Item Similarity Matrix
        self.item_sim_mat = cosine_similarity(self.df_utility_values.T)
        # make prediction
        pred_mat = self.predict()
        # get prediction y
        y_pred = pred_mat[test_index]

        # RMSE
        rmse = np.sqrt(np.mean((y_pred - y_test)**2))
        # MAE
        mae = np.mean(np.abs(y_pred - y_test))

        return rmse, mae


    def recommender(self, n=10):

        n_users = self.df_utility.shape[0]
        recommend_dict = {}

        for user_id in tqdm(self.df_utility.index):

            pred_ratings = self._predict_one_user(user_id)
            # Get item indexes sorted by predicted rating
            item_index_sorted_by_pred_rating = list(np.argsort(pred_ratings))
            # Find items that have been rated by user
            item_index_rated_by_this_user = self.df_utility.loc[user_id,:].values.nonzero()[0]
            # We want to exclude the items that have been rated by user
            unrated_items_by_pred_rating = [item for item in item_index_sorted_by_pred_rating
                                            if item not in item_index_rated_by_this_user]
            # recommend n top restaurants
            res = self.df_utility.columns[unrated_items_by_pred_rating[-n:]].values
            recommend_dict[user_id] = list(res)

        with open('./model/memory_based_recommender.json', 'w') as fp:
            json.dump(recommend_dict, fp)



class ModelBasedRecommender():

    def __init__(self):

        return None

    def read_data(self, df):

        df_stars = df[['user_id', 'business_id', 'stars']]
        # A reader is needed with rating_scale param
        reader = Reader(rating_scale=(1, 5))
        # The columns must correspond to user id, item id and ratings (in that order).
        data = Dataset.load_from_df(df_stars, reader)

        return data

    def fit_SVD(self, data):

        # KFold
        kf = KFold(random_state=42)
        # cross validation using SVD
        out = cross_validate(SVD(), data, ['rmse', 'mae'], kf)
        mean_rmse = np.mean(out['test_rmse'])
        mean_mae = np.mean(out['test_mae'])
        total_time = np.sum(out['fit_time'])
        print ("The results of Model-based Recommender: ")
        print ("RMSE: %.3f"%mean_rmse)
        print ("MAE: %.3f"%mean_mae)
        print ("Total running time: %.2f s"%total_time)

































