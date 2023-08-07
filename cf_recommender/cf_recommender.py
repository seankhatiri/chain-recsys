import pandas as pd
from surprise import SVD
from surprise import Reader, Dataset, KNNBasic
from surprise.model_selection import cross_validate

################# Implicit Feedback CF #################
'''
    We just fed user-contract interactions as data to our SVD CF model
'''

data = pd.read_parquet('dataset/user_contract_rating.parquet')
reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(data[['user', 'item', 'rating']], reader)


################# SVD CF #####################
# algo = SVD()
# print('cf started')
# cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

################# KNN CF #####################
sim_options = {
    'name': 'cosine',
    'user_based': True
}
algo = KNNBasic(sim_options=sim_options)
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

