#TODO: one time sBERT item_feat embedding and save
from sentence_transformers import SentenceTransformer
import torch
import re
import pandas as pd
import numpy as np
from tqdm import tqdm

def get_item_feat_sbert(items_df, save_path='sbert_embeddings_full.npy'):
    contract2comments = pd.read_parquet('dataset/contracts2comment.parquet')
    c2c_main_class = contract2comments[contract2comments['contract_name'] == contract2comments['class_name']]

    def reorder_text(text):
        lines = text.split("\n")
        notice_lines = [line for line in lines if "@notice" in line]
        other_lines = [line for line in lines if "@notice" not in line]
        reorderd_text = "\n".join(notice_lines + other_lines)
        return reorderd_text

    def preprocess_text(text):
        text = reorder_text(text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove special characters, numbers, etc.
        text = re.sub(r'\W', ' ', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        text = text[:512] if len(text) > 512 else text
        return text

    sentences = []
    for i, item in tqdm(items_df.iterrows(), total=len(items_df)):
        comment_class = c2c_main_class[c2c_main_class['contract_name'] == item['name']]
        if not comment_class.empty and comment_class['class_documentation'].iloc[0] != '':
            sentences.append(comment_class['class_documentation'].iloc[0])
        else:
            class_names = contract2comments[contract2comments['contract_name'] == item['name']]['class_name']
            sentences.append(' '.join(class_names))

    preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]

    # Use SentenceTransformer to obtain embeddings
    model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
    item_feat = model.encode(preprocessed_sentences)

    # Save embeddings to a file
    np.save(save_path, item_feat)
    print(f"Embeddings saved to {save_path}.")
    
    return item_feat

items_ratings_df = pd.read_parquet('dataset/user_contract_rating.parquet')
items_ratings_df = items_ratings_df #[:100000]
items_df = {}
items_df['name'] = items_ratings_df['item'].unique()
items_df['itemId'], unique_names = pd.factorize(items_df['name'])
# items_df['itemId'] = items_df['itemId'] + 1 #TODO test commenting this line didn't breal anything
items_df = pd.DataFrame(items_df, columns=['itemId', 'name'])

get_item_feat_sbert(items_df)
