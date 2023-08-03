import numpy as np
from tqdm import tqdm
import pandas as pd
import json
import csv
import random
import os
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from surprise import KNNBasic
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.frequent_patterns import fpgrowth


def create_rules(df, min_support=0.01, min_threshold=1):
    # Removing 'address' column
    df = df.drop(['address'], axis=1)

    # Generate frequent itemsets
    frequent_itemsets = fpgrowth(df, min_support=0.01, use_colnames=True)

    # Generate rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
    
    return rules

df = pd.read_csv('dataset/adj_matrix.csv')

# Converting your data to one-hot encoded DataFrame
df_encoded = pd.get_dummies(df, columns=df.columns[1:], prefix='', prefix_sep='')

# Removing '-1' interactions
df_encoded = df_encoded.loc[:, df_encoded.columns != '-1']

rules = create_rules(df_encoded)
rules.to_csv('dataset/rules.csv', index=False)