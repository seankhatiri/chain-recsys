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
# data = filtered_data[:10000]

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

############# POPULAR CONTRACT RECOMMENDER ###############
def MAP_at_K_PCR(testset=None, K=None, trainset=None):
    total_map = 0.0
    count_users = 0
    predicted_items = trainset.groupby('item')['rating'].sum().sort_values(ascending=False).head(K).index.tolist()
    
    for user in testset['user'].unique():
        true_items = testset[testset['user'] == user]['item'].tolist()
        if len(true_items) == 0:
            continue
        
        hits = 0
        sum_precisions = 0.0

        for k in range(1, K+1):
            item_at_k = predicted_items[k - 1]
            if item_at_k in true_items:
                hits += 1
                precision_at_k = hits / k
                sum_precisions += precision_at_k
        average_precision = sum_precisions / K #min(len(true_items), K)

        # MAP@K V2
        # item_at_k = predicted_items
        # if any(item in true_items for item in item_at_k): 
        #     hits += 1
        # average_precision = hits / min(len(true_items), K)

        total_map += average_precision
        count_users += 1
    
    MAP_at_K = total_map / count_users if count_users > 0 else 0
    return MAP_at_K

def PCR_split(dataset):
    train_data = pd.DataFrame(columns=['user', 'item', 'rating'])
    test_data = pd.DataFrame(columns=['user', 'item', 'rating'])
    mask_percentage = 0.2

    for user, group in data.groupby('user'):
        n_total = len(group)
        n_mask = int(n_total * mask_percentage)
        
        if n_mask == 0 and n_total > 1:
            n_mask = 1
        
        mask = np.full(n_total, False)
        mask[:n_mask] = True
        np.random.shuffle(mask)
        
        test_group = group[mask]
        train_group = group[~mask]
        
        train_data = pd.concat([train_data, train_group])
        test_data = pd.concat([test_data, test_group])

    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    
    return train_data, test_data

K_values = [1, 2, 3, 4, 5]
trainset, testset = PCR_split(data)

with ThreadPoolExecutor() as executor:
    futures = {executor.submit(MAP_at_K_PCR, testset=testset, K=k, trainset=trainset): k for k in K_values}

    for future in as_completed(futures):
        k = futures[future]
        try:
            map_result = future.result()
            print(f'MAP @ {k}: {map_result}')
        except Exception as e:
            print(f'Failed to compute MAP @ {k} due to {e}')