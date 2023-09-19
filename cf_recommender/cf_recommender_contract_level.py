# NOT UPDATED, UPDATE BASED ON CONTRACT MFR NOTEBOOK
from surprise import KNNBasic # Not working on Apple silicon chip
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import cross_validate, KFold, train_test_split
from mlxtend.frequent_patterns import apriori, association_rules
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from tqdm import tqdm

# For recommendation, first sum all interactions for each user with same topic, then use the same CF MF code and MAP@K eval
user_topic_df = pd.read_parquet("dataset/user_topic_rating.parquet")
user_topic_df = user_topic_df.groupby(['user', 'topic']).agg({'rating': 'sum'}).reset_index()
print(len(user_topic_df))
data = user_topic_df[:100000]

###### DATA PREPROCESSING #######
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
data['rating'] = data['rating'].apply(apply_rating_scale)

def MAP_at_K_MF_batch(model=None, data=None, K=None, batch_size=100):
    total_map = 0.0
    count_users = 0
    
    user_ids = [uid for uid, _, _ in data]
    unique_user_ids = list(set(user_ids))
    
    # Divide unique_user_ids into batches
    batches = [unique_user_ids[i:i + batch_size] for i in range(0, len(unique_user_ids), batch_size)]
    
    def process_batch(batch):
        # print('batch processing started')
        nonlocal total_map, count_users
        item_ids = [iid for _, iid, _ in data]
        
        batch_map = 0.0
        batch_count = 0
        
        for uid in batch:
            # print(f'processing user {uid} started')
            true_relevant = [iid for (u, iid, r) in data if u == uid]
            if len(true_relevant) == 0:
                continue
            
            predicted_items = get_prediction(K, uid, item_ids, model)
            hits = 0
            sum_precisions = 0.0
            
            for k in range(1, K + 1):
                item_at_k = predicted_items[k - 1]
                if item_at_k in true_relevant:
                    hits += 1
                    precision_at_k = hits / k
                    sum_precisions += precision_at_k
                    
            average_precision = sum_precisions / min(len(true_relevant), K)
            batch_map += average_precision
            batch_count += 1
            # print(f'processing user {uid} finished')
        
        # print('batch processing finished')
        return batch_map, batch_count
    
    def get_prediction(K, uid, item_ids, model):
        predicted_scores = [model.predict(uid, iid).est for iid in item_ids]
        top_k_indices = np.argsort(predicted_scores)[::-1][:K]
        results = [item_ids[i] for i in top_k_indices]
        return results

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_batch, batches), total=len(batches)))
    
    for batch_map, batch_count in results:
        total_map += batch_map
        count_users += batch_count
    
    MAP_at_K = total_map / count_users if count_users > 0 else 0
    return MAP_at_K

reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(data[['user', 'topic', 'rating']], reader)
# data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=0.2)
print(trainset.n_users)
model = SVD()
model.fit(trainset)
print('model fit Finished')
K_values = [1, 2, 3, 4, 5]

with ThreadPoolExecutor() as executor:
    futures = {executor.submit(MAP_at_K_MF_batch, model=model, data=testset, K=k, batch_size=5): k for k in K_values}
    for future in as_completed(futures):
        k = futures[future]
        try:
            map_result = future.result()
            print(f'MAP @ {k}: {map_result}')
        except Exception as e:
            print(f'Failed to compute MAP @ {k} due to {e}')
