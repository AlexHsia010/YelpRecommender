#!/usr/bin/env python
#-*- coding: utf-8 -*-

# Authors: Eric@FLAGDream <eric.d@flagdream.com>
import numpy as np
import pandas as pd
from rs_collaborative_filtering import MemoryBasedRecommender, ModelBasedRecommender



if __name__ == "__main__":

    # read data
    df = pd.read_csv("../data/last_1_years_restaurant_reviews.csv")

    # Memory based recommender system
    memobased = MemoryBasedRecommender(n_neighbor=75)
    memobased.utility_matrix(df)
    rmse, mae = memobased.evaluation()
    print ()
    print ("The results of Memory-based Recommender: ")
    print ("RMSE: %.3f"%rmse)
    print ("MAE: %.3f"%mae)
    # save the results
    memobased.recommender()

    # Model based recommender system
    print ()
    modelbased = ModelBasedRecommender()
    data = modelbased.read_data(df)
    modelbased.fit_SVD(data)







