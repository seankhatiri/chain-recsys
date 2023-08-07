import pandas as pd
from surprise import SVD
from surprise import Reader, Dataset, KNNBasic
from surprise.model_selection import cross_validate

################# Implicit Feedback CF #################
'''
    We just fed user-contract interactions as data to our SVD CF model
'''
def data_loader(adj_matrix_path, format):
    df = pd.read_csv(adj_matrix_path + format) if format == '.csv' else pd.read_parquet(adj_matrix_path + format)
    df = df[:1]
    data = []

    if adj_matrix_path == 'dataset/adj_matrix_kmean_tags':
        for i, row in df.iterrows():
            for j in range(1, 51):
                if row[j] != -1:
                    data.append([row[0], row[j], 1])
    
    if adj_matrix_path == 'dataset/adj_matrix_contract_names':
        for i, row in df.iterrows():
            for j in range(1, len(df.columns)):
                if row[j] != 0:
                    data.append([row.name, df.columns[j], 1])

    data = pd.DataFrame(data, columns=['user', 'item', 'rating'])
    return data


# data = data_loader('dataset/adj_matrix_kmean_tags', '.csv')
data = data_loader('dataset/adj_matrix_contract_names', '.parquet')
reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(data[['user', 'item', 'rating']], reader)

################# SVD CF #####################
algo = SVD()
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

################# KNN CF #####################
# sim_options = {
#     'name': 'cosine',
#     'user_based': True
# }
# algo = KNNBasic(sim_options=sim_options)
# cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

