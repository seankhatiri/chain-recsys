# TODO: Create a system design architecture of all steps (DONE)
# BUGFIX: Revise the user-item-rating dataset to add num of interactions with a contract instead of returning always 1 (DONE)
# TODO: add recall, percision eval for CF recommender (DONE)
# TODO: Add another baseline to return popular items as recommended items (DONE)
# TODO: Add custom data loader to graph recommender (FUTURE: Internal transactions) (DONE)
# BUGFIX: Change the collaborative filtering (baselines) to compare the top-k (calculate the probability for all test users given all items) (DONE)
# TODO: Find the addresses with more exactly 50 transactions, re-fetch their transactions without a limit, run the EDAs, re-run the collaborative filtering
# TODO: Run the LDA topic modeling inference to tag all the contracts -> then run the CF for user-tag interaction
# TODO: For topic modeling run the coherent k approach to find the best K for the dataset


# TODO: Train simple GCN given node_names (user address or contract_names) and edge_names (function_names) with weight (number of transaction between given user and contract ith buckling) (IN PROGRESS)
# TODO: Add the time stamp as edge feature, and contract code as node feature, train the GCN (TO DO)
# TODO: Instead of GCN let's use the Graphormer structure and train with the same graph representation as GCN (Node and edge features) (TO DO)
# TODO: Create the graph representation with weighted edges for user-contracts transactions (IN PROGRESS)


# TODO: Find the paper in which demonstrate recommending by news name is much more effective than news body embedding
# TODO: Add another tagging approach and run CF as a new baseline: search for established category top-5 keyword and assign the most similar category to each contract

# Comparing with running collaborative filtering on user-item-interactions, what would be the simplest next step considering graph-based embedding of users and contracts?
# QUestion: should we keep user-user edges in our transaction graph too?
# First step: create the graph structure data (it can be user-contract adjacency matrix), but how to handle node and edge features?
# Second step given users and contract_names as nodes and function_name and time stamp as edge (features) train a GCN
# Third try to improve node representation by adding contract code and update the node embedding with the same expriment design as before
# Forth try to exploit graphormer instead of GCN to see how well it would do on our dataset

Considering the below recommender, based on user-contractinteractions dataset, we define the concept aof being relevant as follows: the intersection among our top-k recommendations among item set with all user-contract true interactions. The reason that we don't just consider the top-k user interactions is the fact that in transaction context any interactio with a contract will be accepted as a sign that user is interested to the contract. In other words, unlike movielens dataset that users rate to each movie and they can dislike a movie, in our context each interaction is a strong sign of being interested.
Popular contract recommender results:
old version
K=5
Average Precision: 0.1418240614893758
Average Recall: 0.2610141225726914
k=10
Average Precision: 0.14177953789193998
Average Recall: 0.3361631204012056
k=15
Average Precision: 0.1274031025917036
Average Recall: 0.37482553690083065
k=20
Average Precision: 0.11067746168629868
Average Recall: 0.44381573135649094

new version Popular contract recommender results:
For len(user_item_interactions) = 10K
MAP @ 1: 0.2236842105263158
MAP @ 20: 0.21588790670502636
MAP @ 10: 0.20795416962268984
MAP @ 5: 0.21625073099415143
MAP @ 15: 0.2120367406216425

For len(user_item_interactions) = 100K

For len(user_item_interactions) = 400K
MAP @ 1: 0.23065566855696676
MAP @ 15: 0.20843115590375655
MAP @ 5: 0.2133290046502658
MAP @ 20: 0.21250816298391695
MAP @ 10: 0.2062692781380868


# TODO: definitly using surprise library and predicting for any two pair of user item dosn't make sense and is computationally inefficient, we need to desgin the latent matrixes and with one matrix multiplication calculate the final matrix, now given any i, j we have the predicted value weather i interacts with j or not.

## Matrix Factorization
### Name Level
For len(user_item_interactions) = 10K
MAP @ 20: 0.2810518363529898
MAP @ 1: 0.025606469002695417
MAP @ 5: 0.0770440251572327
MAP @ 10: 0.1533179309459633
MAP @ 15: 0.22453364993796263

For len(user_item_interactions) = 100K
MAP @ 1: 0.006260671599317018
MAP @ 5: 0.016703898690950486
MAP @ 10: 0.04576288231517656
MAP @ 15: 0.08634326142013071
MAP @ 20: 0.1355948135331553

For len(user_item_interactions) = 400K


### Contract Level

For LDA K=15 we have:
For len(user_topic_interactions) = 10K
MAP @ 1: 0.23585598824393827
MAP @ 2: 0.33945628214548124
MAP @ 3: 0.4867744305657605
MAP @ 4: 0.6465833945628219
MAP @ 5: 0.8082292432035263

For len(user_topic_interactions) = 100K




Note: we have 29k users with more than 50 k