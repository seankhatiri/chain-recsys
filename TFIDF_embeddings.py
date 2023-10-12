import torch
import pandas as pd
import numpy as np

def get_item_feat_tfidf(items_df, save_path='tfidf_embeddings.npy'):
    contract_top_words_df = pd.read_parquet('dataset/contract_top_words.parquet')
    contract_top_words_df = contract_top_words_df.rename(columns={'contract_name': 'name'})
    contracts_df_top_words = items_df.merge(contract_top_words_df, on='name', how='left')
    contracts_df_top_words['keywords'] = contracts_df_top_words['keywords'].fillna('')
    items_df = contracts_df_top_words
    items_df.set_index('itemId', inplace=True)
    # f =5 # ratio to determine the number of top keywords selected for each contract to construct item_feat
    items_df['truncated_keywords'] = items_df['keywords'].apply(lambda x: ','.join(x.split(',')))
    item_feat = items_df['truncated_keywords'].str.get_dummies(',')
    np.save(save_path, item_feat.values)
    print(f"Embeddings saved to {save_path}.")
    return item_feat

items_ratings_df = pd.read_parquet('dataset/user_contract_rating.parquet')
items_ratings_df = items_ratings_df[:100000]
items_df = {}
items_df['name'] = items_ratings_df['item'].unique()
items_df['itemId'], unique_names = pd.factorize(items_df['name'])
# items_df['itemId'] = items_df['itemId'] + 1 #TODO test commenting this line didn't breal anything
items_df = pd.DataFrame(items_df, columns=['itemId', 'name'])

get_item_feat_tfidf(items_df)