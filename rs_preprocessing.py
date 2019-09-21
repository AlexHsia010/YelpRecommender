#!/usr/bin/env python
#-*- coding: utf-8 -*-

# Authors: Eric@FLAGDream <eric.d@flagdream.com>

import json
import pandas as pd
from tqdm import tqdm


def process_business(df_business):

    # city, check if it contains "Las Vegas"
    filter_city = df_business['city'].apply(str).str.contains("Las Vegas")
    # categories, check if it is null
    filter_category_not_null = ~df_business["categories"].isnull()
    # categories, check if it contains "Restaurants"
    filter_category_restaurant = df_business["categories"].apply(str).str.contains("Restaurants")
    # filter DataFrame, and name it df_filtered
    df_filtered = df_business[filter_city & filter_category_not_null & filter_category_restaurant]
    # keep relevant columns
    df_selected_business = df_filtered[['business_id', 'name', 'categories', 'stars']]
    # Rename
    df_selected_business.rename(columns={"stars":"avg_stars"}, inplace=True)
    # set index
    df_selected_business = df_selected_business.set_index('business_id')

    return df_selected_business


def merge_review(df_review, df_selected_business):

    # filter. You might want to alter the datetime, in order to get more data
    df_review_filtered = df_review[df_review['date']>'2018-10-01'].set_index('business_id')
    # inner join df_left and df_right
    df_final = df_selected_business.join(df_review_filtered, how='inner')
    # reset the index 
    df_final = df_final.reset_index()


    return df_final



if __name__ == "__main__":

    # read business data into DataFrame
    print ("Read business yelp dataset...")
    total_business_lines = sum(1 for line in open('./yelp_dataset/business.json','r'))
    with open('./yelp_dataset/business.json', encoding='utf-8') as f:
        df_business = pd.DataFrame(json.loads(line) for line in tqdm(f, total=total_business_lines))

    # read review data. Warning!!! 5GB review data
    print ("Read review yelp dataset...(might take up to 5 mins)")
    total_review_lines = sum(1 for line in open('./yelp_dataset/review.json','r'))
    with open('./yelp_dataset/review.json', encoding='utf-8') as f:
        df_review = pd.DataFrame(json.loads(line) for line in tqdm(f, total=total_review_lines))

    # process business
    df_selected_business = process_business(df_business)

    print ("Start merging business and review data...")
    # merge data
    df_final = merge_review(df_review, df_selected_business)

    # write tmp data
    # Save to data/last_1_years_restaurant_reviews.csv for your next task
    df_final.to_csv('./yelp_dataset/last_1_years_restaurant_reviews.csv',index=False)






















