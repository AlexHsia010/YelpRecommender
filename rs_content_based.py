#!/usr/bin/env python
#-*- coding: utf-8 -*-

# Authors: Eric@FLAGDream <eric.d@flagdream.com>
import numpy as np
import pandas as pd
# skip all warnings
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import CountVectorizer
# use svd to reduce dimension
from sklearn.decomposition import TruncatedSVD

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, train_test_split


def feature_engineer(df):

    # group by business_id, then average numerical features
    df_average = df.groupby(['business_id']).mean()
    # group by business_id, extract categories data
    categories_series = df.groupby(['business_id']).categories.apply(np.unique)
    # convert categories data to string
    categories_series = categories_series.str.join('')

    # business_id, categories table
    vectorizer = CountVectorizer()
    categories_mat = vectorizer.fit_transform(categories_series).toarray()
    categories = vectorizer.get_feature_names()
    df_categories = pd.DataFrame(categories_mat,
                             columns=categories, 
                             index=categories_series.index)

    # initialize a SVD
    svd = TruncatedSVD(n_components=75, random_state=42)
    # fit SVD on categories_mat
    svd.fit(categories_mat)
    # transform original data
    categories_svd = svd.transform(categories_mat)
    # create a new dataframe
    df_categories_svd = pd.DataFrame(categories_svd,
                                 index=categories_series.index)
    # join df_average and df_categories_svd
    df_business = df_average.join(df_categories_svd)

    return df_business


if __name__ == "__main__":

    # read data
    df = pd.read_csv("../data/last_1_years_restaurant_reviews.csv")
    # prepare for modeling
    data = feature_engineer(df)

    # get X and y
    target = 'stars'
    features = [x for x in data.columns if x not in ['avg_stars','stars']]
    y = data[target].values >= 4.0
    X = data[features].values

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # candidate models
    models = [LogisticRegression(), 
              DecisionTreeClassifier(), 
              RandomForestClassifier(),
              AdaBoostClassifier(),
              XGBClassifier()]
    
    labels = ['lr','dt','rf','ada','xgb']
    
    print('5-fold cross validation:\n')
    # train models
    for model, label in zip(models,labels):
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
        print("Accuracy: %0.4f (+/- %0.4f) [%s]" 
              % (scores.mean(), scores.std(), label))

    # You might import modules from LendingClub or TipPrediction project
    # to tune hyperparameters and make predictions on test set

















