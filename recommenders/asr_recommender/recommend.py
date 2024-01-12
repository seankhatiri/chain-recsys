import numpy as np
from tqdm import tqdm
import pandas as pd
import json
import csv
import random
import os
import sys
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from surprise import KNNBasic
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate
from mlxtend.frequent_patterns import apriori, association_rules


# Function to recommend items
def recommend(rules, user_items):
    # Selecting rules where the antecedent is in the user_items
    applicable_rules = rules[rules['antecedents'].apply(lambda x: set(x).issubset(set(user_items)))]
    # Sorting rules by confidence
    applicable_rules = applicable_rules.sort_values(by='confidence', ascending=False)
    # Getting consequents of the rules
    recommendations = applicable_rules['consequents'].tolist()
    return recommendations

# Load the rules
rules = pd.read_csv('rules.csv')

# Get user id from command line
user_id = sys.argv[1]

# Load the user interactions for this user
# Suppose df_user_interactions is a dataframe that contains all the user interactions
df_user_interactions = pd.read_csv('user_interactions.csv')

# Get user interactions for this user
user_interactions = df_user_interactions.loc[df_user_interactions['address'] == user_id]

# Remove -1 interactions
user_interactions = user_interactions[user_interactions != -1]

recommendations = recommend(rules, user_interactions)
print(recommendations)
