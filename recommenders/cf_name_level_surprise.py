import pandas as pd
from surprise import SVD
from surprise import Reader, Dataset, KNNBasic
from surprise.model_selection import cross_validate
from surprise.model_selection import cross_validate, KFold, train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import numpy as np
from random import shuffle, sample

############# RUNNING BASELINE RECOMMENDERS ###############

data = pd.read_parquet('dataset/user_contract_rating.parquet')
data = data[data['item'] != '']

# filtered_data = data.groupby('item').filter(lambda x: len(x) > 0) # This removes diverse contracts with low interactions

filtered_data = data.groupby('user').filter(lambda x: len(x) > 5) # This keeps the user with diverse set of contract interactions

data = filtered_data[:10000]
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


################# AUTO-EVAL SVD CF #####################
# algo = SVD()
# print('cf started')
# cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

################# AUTO-EVAL KNN CF #####################
# sim_options = {
#     'name': 'cosine',
#     'user_based': True
# }
# algo = KNNBasic(sim_options=sim_options)
# cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

#################### CF RECOMMENDER #######################
'''The average percision just tell us how relevant our recommendations are, 
    but dosn't tell us how good our ranking is
'''
def MAP_at_K_MF_batch(model=None, testset=None, K=None, batch_size=100, fullset=None):
    total_map = 0.0
    count_users = 0
    
    user_ids = [uid for uid, _, _ in testset]
    item_ids = [iid for _, iid, _ in testset]
    unique_user_ids = list(set(user_ids))
    unique_item_ids = list(set(item_ids))
    
    batches = [unique_user_ids[i:i + batch_size] for i in range(0, len(unique_user_ids), batch_size)]
    
    def process_batch(batch):
        nonlocal total_map, count_users
        batch_map = 0.0
        batch_count = 0
        
        for uid in batch:
            testset_true_relevant = [iid for (u, iid, r) in testset if u == uid]
            # fullset_true_relevant = fullset.loc[fullset['user'] == uid]['item']
            # fullset_true_relevant = set(fullset_true_relevant)
            if len(testset_true_relevant) == 0: # or len(fullset_true_relevant) == 0:
                continue

            predicted_items = get_prediction(K, uid, unique_item_ids, model)
            hits = 0
            sum_precisions = 0.0
            
            # MAP@K V1
            for k in range(1, K+1):
                item_at_k = predicted_items[k - 1]
                if item_at_k in testset_true_relevant:
                    hits += 1
                    precision_at_k = hits / k
                    sum_precisions += precision_at_k
            average_precision = sum_precisions / K #min(len(testset_true_relevant), K)
            
            # # MAP@K V2 # Not sure its correct or not
            # for k in range(1, K + 1):
            #     item_at_k = predicted_items[:K]
            #     if any(item in true_relevant for item in item_at_k): 
            #         hits += 1
            #         precision_at_k = hits / K
            #         sum_precisions += precision_at_k
            # average_precision = sum_precisions / min(len(true_relevant), K)

            # # AP@K
            # item_at_k = predicted_items
            # if any(item in true_relevant for item in item_at_k): 
            #     hits += 1
            # average_precision = hits / min(len(true_relevant), K)

            # print('actual', len(true_relevant), 'average_precision', average_precision)
            batch_map += average_precision
            batch_count += 1
        
        return batch_map, batch_count
    
    def get_prediction(K, uid, unique_item_ids, model):
        predicted_scores = [model.predict(uid, iid).est for iid in unique_item_ids]
        top_k_indices = np.argsort(predicted_scores)[::-1][:K]
        results = [unique_item_ids[i] for i in top_k_indices]
        return results

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_batch, batches), total=len(batches)))
    for batch_map, batch_count in results:
        total_map += batch_map
        count_users += batch_count
    MAP_at_K = total_map / count_users if count_users > 0 else 0
    return MAP_at_K
    # temp = process_all(unique_user_ids) # Created this to test process_data() method, validating process_batch results
    # return temp

reader = Reader(rating_scale=(1,5))
fullset = data
data = Dataset.load_from_df(data[['user', 'item', 'rating']], reader)
# data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=0.2)
print(trainset.n_users)
model = SVD()
model.fit(trainset)
print('model fit Finished')
K_values = [1, 2, 3, 4, 5] # , 10, 15, 20]
# print(MAP_at_K_MF_batch(model=model, data=testset, K=10, batch_size=5, fullset=fullset))

with ThreadPoolExecutor() as executor:
    futures = {executor.submit(MAP_at_K_MF_batch, model=model, testset=testset, K=k, batch_size=5, fullset=fullset): k for k in K_values}
    for future in as_completed(futures):
        k = futures[future]
        # map_result = future.result()
        try:
            map_result = future.result()
            print(f'MAP @ {k}: {map_result}')
        except Exception as e:
            print(f'Failed to compute MAP @ {k} due to {e}')


