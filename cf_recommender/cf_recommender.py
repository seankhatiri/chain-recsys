import pandas as pd
from surprise import Reader, Dataset, KNNBasic
from surprise.model_selection import cross_validate

# Load the data from your csv file
df = pd.read_csv('dataset/adj_matrix.csv')
df = df[:50651]
# Preprocess the dataframe
# Flatten the dataframe and create interaction matrix
data = []
for i, row in df.iterrows():
    for j in range(1, 51):
        if row[j] != -1:
            data.append([row[0], row[j], 1])  # if an interaction exists, rate it as 1

# Convert list into dataframe
data = pd.DataFrame(data, columns=['user', 'item', 'rating'])

# Define the rating scale
reader = Reader(rating_scale=(0, 1))  # scale is 0 to 1, where 1 is an interaction

# Load the data into Surprise dataset
data = Dataset.load_from_df(data[['user', 'item', 'rating']], reader)

# Use user-based collaborative filtering
sim_options = {
    'name': 'cosine',
    'user_based': True  # compute similarities between users
}

algo = KNNBasic(sim_options=sim_options)

# Run 5-fold cross-validation and print results
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)