import pandas as pd
from lightfm import LightFM
import numpy as np
from lightfm.data import Dataset
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

user_topic_df = pd.read_parquet("dataset/user_topic_rating.parquet")
user_item_df = pd.read_parquet("dataset/user_contract_rating.parquet")
# user_topic_df = user_topic_df.groupby('user').filter(lambda x: len(x) > 5) #didn't test with this line for all data
contract_to_topic_df = pd.read_parquet("dataset/contract_name_topic.parquet")
# user_topic_df = user_topic_df[:10000]
# user_item_df = user_item_df[:10000]

# user_topic_df = user_topic_df.groupby(['user', 'topic']).agg({'rating': 'sum'}).reset_index() # with this line the result is weaker

print('len of unique topics', len(user_topic_df['topic'].unique()))
valid_contract_names = contract_to_topic_df['contract_name'].unique() # Filter contracts that didn't get any topic from trained LDA
user_item_df = user_item_df[user_item_df['item'].isin(valid_contract_names)]

def apply_rating_scale(rating):
    if rating == 1:
        return 1
    elif rating <= 5:
        return 2
    elif rating <= 15:
        return 3
    elif rating <= 30:
        return 4
    else:
        return 5

user_topic_df['rating'] = user_topic_df['rating'].apply(apply_rating_scale)
user_item_df['rating'] = user_item_df['rating'].apply(apply_rating_scale)

topic_dataset = Dataset()
topic_dataset.fit(user_topic_df['user'], user_topic_df['topic'])
#TODO: first split the trainset and testset, then agg user_topic_df
topic_train_df, topic_test_df = train_test_split(user_topic_df, test_size=0.2, random_state=42)
name_train_df, name_test_df = train_test_split(user_item_df, test_size=0.2, random_state=42)

(topic_test_interactions, ـ) = topic_dataset.build_interactions(
    (row['user'], row['topic'], row['rating']) for index, row in topic_test_df.iterrows())
(topic_train_interactions, weight_topic_train_interactions) = topic_dataset.build_interactions(
    (row['user'], row['topic'], row['rating']) for index, row in topic_train_df.iterrows())

model = LightFM(loss='warp')
model.fit(topic_train_interactions, epochs=30, num_threads=2, sample_weight=weight_topic_train_interactions) # Training

n_users, n_topics = topic_test_interactions.shape
print('n_users:', n_users, 'n_topics:', n_topics)

user_ids_mapping, _, item_ids_mapping, _ = topic_dataset.mapping()
user_ids = list(user_ids_mapping.keys())
item_ids = list(item_ids_mapping.keys())
# Possible BUG: there are some ranking value equal to 0
ranks = model.predict_rank(topic_test_interactions)

dense_ranks = ranks.toarray()
dense_ranks[dense_ranks == 0] = np.inf
#TODO: if there is a zero, it returns that as the min, if it turns out the zero is a bug, we should return min except 0
min_rank_indices = np.argmin(dense_ranks, axis=1)
best_recommendations = {}

for user_index, item_index in enumerate(min_rank_indices):
    user_id = user_ids[user_index]
    item_id = item_ids[item_index]
    best_recommendations[user_id] = item_id

top_k = 5 
top_k_contracts = []

# top-k popular contract associated to user's predicted topic within testset
def topic_popular_contracts(name_test_df):
    merged_df = pd.merge(name_test_df, contract_to_topic_df, left_on='item', right_on='contract_name', how='inner')
    merged_df.rename(columns={'most_probable_topic': 'topic'}, inplace=True)
    grouped_df = merged_df.groupby(['topic', 'item']).agg({'rating': 'sum'}).reset_index()
    grouped_df.sort_values(['topic', 'rating'], ascending=[True, False], inplace=True)
    topic_contract_dict = {}
    for topic, group in grouped_df.groupby('topic'):
        topic_contract_dict[topic] = [record['item'] for record in group[['item', 'rating']].to_dict('records')]
    return topic_contract_dict

topic_contract_dict_testset = topic_popular_contracts(name_test_df) # BUG: This should be name_test_df but it fails
topic_contract_dict_fullset = topic_popular_contracts(user_item_df)

sum_precision = 0
total_users = 0
K_values = [1, 2, 3, 4, 5]  # Change this to the value of K you are interested in

for K in K_values:
    #TODO: How to skip contracts that user had interaction with in trainset to calculate percision when predicting score for new contracts?
    for user, topic_value in best_recommendations.items():
        recommended_contracts = topic_contract_dict_testset[topic_value][:K] if topic_value in topic_contract_dict_testset.keys() else topic_contract_dict_fullset[topic_value][:K]
        actual_contracts = name_test_df[name_test_df['user'] == user]['item'].tolist()  # Get actual contracts for the user from the test set
        
        hits = 0
        for contract in recommended_contracts:
            if contract in actual_contracts:
                hits += 1
        
        precision_at_k = hits / K
        sum_precision += precision_at_k
        total_users += 1

    if total_users > 0:
        AP_at_K = sum_precision / total_users
    else:
        AP_at_K = 0  # Handle case where total_users is 0 to avoid division by zero

    print(f"Average Precision at {K}: {AP_at_K}")
