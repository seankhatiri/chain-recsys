# TODO: Create a system design architecture of all steps (DONE)
# BUGFIX: Revise the user-item-rating dataset to add num of interactions with a contract instead of returning always 1 (DONE)
# TODO: add recall, percision eval for CF recommender (DONE)
# TODO: Add another baseline to return popular items as recommended items (DONE)
# TODO: Add custom data loader to graph recommender (FUTURE: Internal transactions) (DONE)
# TODO: Create the graph representation with weighted edges for user-contracts transactions (IN PROGRESS)
# BUGFIX: Change the collaborative filtering (baselines) to compare the top-k (calculate the probability for all test users given all items)
# TODO: Find the addresses with more exactly 50 transactions, re-fetch their transactions without a limit, run the EDAs, re-run the collaborative filtering


# TODO: Train simple GCN given node_names (user address or contract_names) and edge_names (function_names) with weight (number of transaction between given user and contract ith buckling) (IN PROGRESS)
# TODO: Add the time stamp as edge feature, and contract code as node feature, train the GCN (TO DO)
# TODO: Instead of GCN let's use the Graphormer structure and train with the same graph representation as GCN (Node and edge features) (TO DO)
# TODO: For topic modeling run the coherent k approach to find the best K for the dataset
# TODO: Run the LDA topic modeling inference to tag all the contracts -> then run the CF for user-tag interaction

# TODO: Find the paper in which demonstrate recommending by news name is much more effective than news body embedding
# TODO: Add another tagging approach and run CF as a new baseline: search for established category top-5 keyword and assign the most similar category to each contract
# BUGFIX: We have lots of -1 in kmean-tags adj_matrix, find why and fix it

# Comparing with running collaborative filtering on user-item-interactions, what would be the simplest next step considering graph-based embedding of users and contracts?
# QUestion: should we keep user-user edges in our transaction graph too?
# First step: create the graph structure data (it can be user-contract adjacency matrix), but how to handle node and edge features?
# Second step given users and contract_names as nodes and function_name and time stamp as edge (features) train a GCN
# Third try to improve node representation by adding contract code and update the node embedding with the same expriment design as before
# Forth try to exploit graphormer instead of GCN to see how well it would do on our dataset

Average Precision: 0.1418240614893758
Average Recall: 0.2610141225726914