from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.data import Dataset
import pandas as pd
import numpy as np
from lightfm.datasets import fetch_movielens
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from collie.movielens import read_movielens_df
df = pd.read_csv('ml-latest-small/ratings.csv')
# shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df.rename(columns = {'userId': 'user', 'movieId': 'item'})
df = df.drop(columns=['timestamp'])
# print(df[df['user'] == 450 ]['item'])
print(len(df['user'].unique()))
print(len(df['item'].unique()))

# movielens = fetch_movielens()
# train = movielens['train']
# test = movielens['test']
# model = LightFM(learning_rate=0.05, loss='bpr')
# model.fit(train, epochs=10)
# for k in [1, 2, 3, 4, 5]:
#     precision = precision_at_k(model, test, k=k).mean()
#     print(f"Precision at k={k}: {precision}")


# df = pd.read_parquet("dataset/user_contract_rating.parquet")
# df = df.groupby('user').filter(lambda x: len(x) > 10)
# df = df[df['item'] != '']
# print(len(df))

def add_neg_samples(df):
    len_pos_samples = len(df)
    len_neg_samples = 0
    neg_samples = []
    all_items = set(df['item'].unique())
    popular_items = set(df[df['item'] < 500]['item'].unique())
    count = 0

    for user, user_data in tqdm(df.groupby('user'), total = len(df.groupby('user'))):
        # if count == 0:
        #     count += 1 
        #     continue
        pos_items = set(user_data['item'])
        neg_items = all_items - pos_items

        selected_neg_items = list(np.random.choice(list(neg_items), size= min(len(pos_items), len(neg_items)), replace=False))
        # print(pos_items)
        # print(selected_neg_items)
        
        neg_samples.extend([(user, neg_item) for neg_item in selected_neg_items])
        len_neg_samples += len(selected_neg_items)

        if len_neg_samples >= len_pos_samples:
            break

    neg_samples_df = pd.DataFrame(neg_samples, columns=['user', 'item'])
    neg_samples_df['rating'] = 0
    df = pd.concat([df, neg_samples_df], ignore_index=True)
    return df

def apply_rating_scale(rating):
    if rating == 0:
        return 0
    else:
        return 1
    # if rating == 1:
    #     return 1
    # elif rating <= 5:
    #     return 2
    # elif rating <= 15:
    #     return 3
    # elif rating <= 30:
    #     return 4
    # else:
    #     return 5

df['rating'] = df['rating'].apply(apply_rating_scale)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
test_df = add_neg_samples(test_df)

print(len(test_df))


# train_df_prime, test_df_prime = train_test_split(df_prime, test_size=0.2, random_state=42)

# dataset = Dataset()
# dataset.fit(df['user'], df['item'])
# user_ids_mapping, _, item_ids_mapping, _ = dataset.mapping()

# (interactions, _) = dataset.build_interactions((row['user'], row['item'], row['rating']) for index, row in df.iterrows())
# (train_interactions, train_interactions_weight) = dataset.build_interactions((row['user'], row['item'], row['rating']) for index, row in train_df.iterrows())
# (test_interactions, _) = dataset.build_interactions((row['user'], row['item'], row['rating']) for index, row in test_df.iterrows())

# model = LightFM(loss='warp')
# model.fit(train_interactions, epochs=30, num_threads=2, sample_weight=train_interactions_weight)


# ranks = model.predict_rank(test_interactions) # to see the inner rankings, it says within preds, the true_positive rank a what position
# print(ranks)

# for k in [1, 2, 3, 4, 5]:
#     precision = precision_at_k(model, test_interactions, k=k).mean()
#     print(f"Precision at k={k}: {precision}")


########### NEW GNN EVAL ###########

def AP_at_K(model, edgelist_test, edgelist_train, k):
    user_nodes = set([edge[0] for edge in edgelist_test])
    
    avg_precisions = []
    count = 0
    
    # for user in tqdm(user_nodes, total=len(user_nodes)):
    for user in user_nodes:
        contract_nodes = set([edge[1] for edge in edgelist_test if edge[0] == user])
        contract_nodes_ground_truth = set([edge[1] for edge in edgelist_test if edge[0] == user and edge[2] == 1])
        contract_nodes_ground_truth_train = set([edge[1] for edge in edgelist_train if edge[0] == user and edge[2] == 1])
        # print(user)
        # print(contract_nodes)
        # print(contract_nodes_ground_truth_train)
        # print(contract_nodes_ground_truth)

        user_id_internal = user_ids_mapping[user]
        contract_nodes_internal = [item_ids_mapping[item] for item in contract_nodes]
        scores = model.predict(user_id_internal, contract_nodes_internal)

        contract_nodes = list(contract_nodes)
        sorted_indices = scores.argsort()[::-1]
        top_k_contracts = [contract_nodes[i] for i in sorted_indices[:k]]
        
        relevant_items = set(top_k_contracts).intersection(contract_nodes_ground_truth)
        precision_k = len(relevant_items) / k
        count += 1
        avg_precisions.append(precision_k)

        # if count % 1000 == 0 : print(np.mean(avg_precisions))
        
    return np.mean(avg_precisions)

k_values = [1, 2, 3, 4, 5]
edgelist_test = [(row['user'], row['item'], row['rating']) for _, row in test_df.iterrows()]
edgelist_train = [(row['user'], row['item'], row['rating']) for _, row in train_df.iterrows()]

for k in k_values:
    average_precision = AP_at_K(model, edgelist_test, edgelist_train, k)
    print(f"AP@{k}:", average_precision)