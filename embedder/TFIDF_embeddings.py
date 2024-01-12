import torch
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer

def get_item_feat_tfidf(items_df, save_path='tfidf_embeddings_100k_dim_1400.npy'):

    def get_top_keywords():
        # contract_top_words_df = pd.read_parquet('dataset/contract_top_words.parquet')
        dir_path = 'backup_data/contract_codes/parsed'
        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.parquet')]
        dfs = [pd.read_parquet(f) for f in files]
        parsed_contracts_all = pd.concat(dfs, ignore_index=True)
        
        parsed_contracts_all.fillna('', inplace=True)
        parsed_contracts = parsed_contracts_all.groupby('contract_name').agg({
            'class_documentation': ' '.join,
            # 'func_documentation': ' '.join
        }).reset_index()
        parsed_contracts = parsed_contracts
        parsed_contracts['documentation'] = parsed_contracts['class_documentation'] # + ' ' + parsed_contracts['func_documentation']
        vectorizer = TfidfVectorizer(max_df=0.85, max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(parsed_contracts['documentation'])

        def get_top_n_keywords(row, features, top_n=5):
            tfidf_scores = [(features[col], row[col]) for col in range(len(features))]
            sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)[:top_n]
            return ', '.join([score[0] for score in sorted_scores])


        feature_names = vectorizer.get_feature_names_out()
        parsed_contracts['keywords'] = tfidf_matrix.toarray().tolist()
        parsed_contracts['keywords'] = parsed_contracts['keywords'].apply(lambda row: get_top_n_keywords(row, feature_names))
        contract_top_words = parsed_contracts[['contract_name', 'keywords']]

        return contract_top_words

    contract_top_words_df = get_top_keywords()
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

item_feat = get_item_feat_tfidf(items_df)
print(item_feat.shape)