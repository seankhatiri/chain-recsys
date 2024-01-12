import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import cross_validate, KFold, train_test_split

############ MovieLen builtin ##############
# data = Dataset.load_builtin("ml-100k")

###### 1 USER_CONTRACT INTRACTIONS #########
data = pd.read_parquet('dataset/user_contract_rating.parquet')
data = data[:100000]

###### COMBINED USER_CONTRACT INTRACTIONS #########
all_user_ids = data['user'].unique()
all_item_ids = data['item'].unique()
all_pairs = pd.MultiIndex.from_product([all_user_ids, all_item_ids], names=['user', 'item']).to_frame(index=False)
positive_pairs = data.drop('rating', axis=1)
negative_pairs = all_pairs.merge(positive_pairs, on=['user', 'item'], how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
negative_pairs['rating'] = 0
data_combined = pd.concat([data, negative_pairs], axis=0)
data_combined.to_parquet('dataset/user_contract_rating_combined.parquet')

reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(data_combined[['user', 'item', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)
algo = SVD()
algo.fit(trainset)
predictions = algo.test(testset)

def precision_at_k(predictions, k=10, threshold=2):
    '''Return precision at k using Surprise predictions'''
    
    user_est_true = {}
    for uid, _, true_r, est, _ in predictions:
        current = user_est_true.get(uid, [])
        current.append((est, true_r))
        user_est_true[uid] = current

    precisions = dict()
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])
        
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

    return sum(prec for prec in precisions.values()) / len(precisions)

print(f'Precision@k: {precision_at_k(predictions, k=20)}')