from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.data import Dataset
import pandas as pd
import numpy as np
from lightfm.datasets import fetch_movielens
from sklearn.model_selection import train_test_split

# movielens = fetch_movielens()
# train = movielens['train']
# test = movielens['test']
# model = LightFM(learning_rate=0.05, loss='bpr')
# model.fit(train, epochs=10)
# for k in [1, 2, 3, 4, 5, 10]:
#     precision = precision_at_k(model, test, k=k, train_interactions=train).mean()
#     print(f"Precision at k={k}: {precision}")


df = pd.read_parquet("dataset/user_contract_rating.parquet")
df = df.groupby('user').filter(lambda x: len(x) > 5)
print(len(df))
df = df[:1000]

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

dataset = Dataset()
dataset.fit(df['user'], df['item'])

(interactions, _) = dataset.build_interactions((row['user'], row['item'], row['rating']) for index, row in df.iterrows())
(train_interactions, train_interactions_weight) = dataset.build_interactions((row['user'], row['item'], row['rating']) for index, row in train_df.iterrows())
(test_interactions, _) = dataset.build_interactions((row['user'], row['item'], row['rating']) for index, row in test_df.iterrows())

model = LightFM(loss='warp')
model.fit(train_interactions, epochs=30, num_threads=2, sample_weight=train_interactions_weight)

# ranks = model.predict_rank(test_interactions) # to see the inner rankings, it says within preds, the true_positive rank a what position
# print(ranks)

for k in [1, 2, 3, 4, 5]:
    precision = precision_at_k(model, test_interactions, k=k).mean()
    print(f"Precision at k={k}: {precision}")