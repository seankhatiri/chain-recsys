import numpy as np
from tqdm import tqdm
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import train_test_split, KFold
from mlxtend.frequent_patterns import fpgrowth
from sklearn.metrics import precision_score, recall_score


# def create_rules(df, min_support=0.01, min_threshold=1):
#     df = df.drop(['address'], axis=1)
#     frequent_itemsets = fpgrowth(df, min_support=0.01, use_colnames=True)
#     rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
#     return rules

# df = pd.read_csv('dataset/adj_matrix.csv')
# df_encoded = pd.get_dummies(df, columns=df.columns[1:], prefix='', prefix_sep='')
# df_encoded = df_encoded.loc[:, df_encoded.columns != '-1']
# rules = create_rules(df_encoded)


def data_loader(adj_matrix_path, format):
    df = pd.read_csv(adj_matrix_path + format) if format == '.csv' else pd.read_parquet(adj_matrix_path + format)
    print('user-item interaction matrix loaded')
    df = df[:500]
    data = []
    for i, row in df.iterrows():
        for j in range(1, len(df.columns)):
            if row[j] != 0:
                data.append([row.name, df.columns[j], 1])

    data = pd.DataFrame(data, columns=['user', 'item', 'rating'])
    return data

data = data_loader('dataset/adj_matrix_contract_names', '.parquet')
df = data.pivot(index='user', columns='item', values='rating').fillna(0)
density = df.sum().sum() / (df.shape[0] * df.shape[1])
print(f"Density of the matrix: {density:.4f}")
train, test = train_test_split(df, test_size=0.2, random_state=42)
precision_scores = []
recall_scores = []

# Using 5-fold cross-validation on the training set
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(train):
    print('kf started')
    train_fold, test_fold = train.iloc[train_index], train.iloc[test_index]
    
    # Generate frequent itemsets using the Apriori algorithm
    frequent_itemsets = apriori(train_fold, min_support=0.05, use_colnames=True)
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    
    # Make predictions on test_fold using the rules
    # Here we assume a simple method: if a user has interacted with an item in the rule's antecedent,
    # we predict they will interact with the item in the rule's consequent
    predictions = test_fold.copy()
    for _, rule in rules.iterrows():
        antecedent = list(rule['antecedents'])[0]  # assuming single-item antecedents for simplicity
        consequent = list(rule['consequents'])[0]
        
        mask = test_fold[antecedent] == 1
        predictions.loc[mask, consequent] = 1
    
    # Calculate precision and recall for this fold
    precision = precision_score(test_fold.values.flatten(), predictions.values.flatten())
    recall = recall_score(test_fold.values.flatten(), predictions.values.flatten())
    
    precision_scores.append(precision)
    recall_scores.append(recall)
    print('kf finished')

# Compute average precision and recall across all folds
avg_precision = sum(precision_scores) / len(precision_scores)
avg_recall = sum(recall_scores) / len(recall_scores)
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")

# rules.to_csv('dataset/rules.csv', index=False)