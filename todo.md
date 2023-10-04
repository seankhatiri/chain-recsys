MF and Popular Recs
# TODO: Create a system design architecture of all steps (DONE)
# BUGFIX: Revise the user-item-rating dataset to add num of interactions with a contract instead of returning always 1 (DONE)
# TODO: add recall, percision eval for CF recommender (DONE)
# TODO: Add another baseline to return popular items as recommended items (DONE)
# TODO: Add custom data loader to graph recommender (FUTURE: Internal transactions) (DONE)
# BUGFIX: Change the collaborative filtering (baselines) to compare the top-k (calculate the probability for all test users given all items) (DONE)
# TODO: Find the addresses with more exactly 50 transactions, re-fetch their transactions without a limit, run the EDAs, re-run the collaborative filtering
# TODO: Run the LDA topic modeling inference to tag all the contracts -> then run the CF for user-tag interaction
# TODO: For topic modeling run the coherent k approach to find the best K for the dataset
# TODO: Note: we have 29k users with == 5o transactions in user transactions



Popular contract recommender results: (Outdated)
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

Popular contract recommender results: (Outdated CUZ the bug to fed trainset when evaluating)
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


TODO: (We finally end up to use Lightbf with some tweak into its design) definitly using surprise library and predicting for any two pair of user item dosn't make sense and is computationally inefficient, we need to desgin the latent matrixes and with one matrix multiplication calculate the final matrix, now given any i, j we have the predicted value weather i interacts with j or not.

Matrix Factorization
Name Level (Outdated)
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


Contract Level (Outdated)

For LDA K=15 we have:
For len(user_topic_interactions) = 10K
MAP @ 1: 0.23585598824393827
MAP @ 2: 0.33945628214548124
MAP @ 3: 0.4867744305657605
MAP @ 4: 0.6465833945628219
MAP @ 5: 0.8082292432035263

For len(user_topic_interactions) = 100K

# Name Level MF Recommender surprise V2:
For len() = 10k
MAP @ 1: 0.06540084388185655
MAP @ 5: 0.12709798406000938
MAP @ 10: 0.24006666778291255
MAP @ 15: 0.3595615277049876
MAP @ 20: 0.4797060051773822

For len() = 100k
MAP @ 1: 0.020055325034578148
MAP @ 5: 0.014941025049946183
MAP @ 10: 0.017924112950893873
MAP @ 15: 0.024026289637646577
MAP @ 20: 0.025551288816438242

Note: It sucks, I tested the same MAP eval for MovieLens and it gives me the same result (~0.02). Maybe instead of predicting the score for the uid and all iids in testset, we should just use the prediction output from library (basically predict one score for each entry in our testset in form of: uid, iid, true_r, predicted_r). In this case 

# Contract Level MF Recommender surprise V2:
for len() = 50k
r = 5
MAP @ 1: 0.21908370651050638
MAP @ 5: 0.10143730623492932
MAP @ 10: 0.0877293908083895
MAP @ 15: 0.09714189438531029
MAP @ 20: 0.0949172668246906

r = 1
MAP @ 1: 0.21717877094972068
MAP @ 5: 0.11288107153941647
MAP @ 10: 0.0708795992791902
MAP @ 15: 0.06331292729493819
MAP @ 20: 0.06229326813653393

Note: as we can see the MAP will reduce when k grows, this may be because of number of topic we had for LDA

# Popular Contract Recommender (After fixed the prev bug)
For len(user-item-rating) = 100k
MAP @ 1: 0.05108991825613079
MAP @ 5: 0.07756893985265907
MAP @ 10: 0.09126641274887184
MAP @ 15: 0.09552536905142488
MAP @ 20: 0.09744209517667898

for len() = all, and change the denominator to k
MAP @ 5: 0.02023273879808665
MAP @ 1: 0.05086694425455223
MAP @ 2: 0.037140326890808525
MAP @ 3: 0.028281855281419754
MAP @ 4: 0.02350110310111264

## name-level mf with lighttbf:
Testset (just kept users with more than 5 unique interactions)
Not passing the trainset to precision_k method (Note: in line 1303 of lightfm predict_ranks cython implementaiton, if we pass the trainset, it will skip testset_items that was in trainset, in other words just predicting the interaction score if the item is new to user and skip the items that user has iteracted with before)
Precision at k=1: 0.10146462917327881
Precision at k=2: 0.09653647989034653
Precision at k=3: 0.0896432027220726
Precision at k=4: 0.08510270714759827
Precision at k=5: 0.08069270104169846
Precision at k=10: 0.06327376514673233

Passing trainset to percision_k:
Precision at k=1: 0.17414332926273346
Precision at k=2: 0.14961771667003632
Precision at k=3: 0.13255344331264496
Precision at k=4: 0.11918524652719498
Precision at k=5: 0.10929439961910248

## MovieLens mf with lighttbf:
Not passing the trainset to precision_k method:
Precision at k=1: 0.11346765607595444
Precision at k=2: 0.10286320000886917
Precision at k=3: 0.10533756762742996
Precision at k=4: 0.1036585345864296
Precision at k=5: 0.1056203693151474

Passing trainset to percision_k:
Precision at k=1: 0.3722163438796997
Precision at k=2: 0.33616119623184204
Precision at k=3: 0.30364084243774414
Precision at k=4: 0.27757158875465393
Precision at k=5: 0.2557794451713562

## contract level mf with lightbf:
Testset (just kept users with more than 5 unique interactions)
Average Precision at 1: 0.10935206029167074
Average Precision at 2: 0.09803513751590487
Average Precision at 3: 0.09091873022091086
Average Precision at 4: 0.08391957032399772
Average Precision at 5: 0.07767984731335678


################################################################################################################
# Graph Recommender

TODO: Train simple GCN given node_names (user address or contract_names) and edge_names (function_names) with weight (number of transaction between given user and contract ith buckling) (IN PROGRESS)

TODO: Add the time stamp as edge feature, and contract code as node feature, train the GCN (TO DO)

TODO: Instead of GCN let's use the Graphormer structure and train with the same graph representation as GCN (Node and edge features) (TO DO)

TODO: Create the graph representation with weighted edges for user-contracts transactions (IN PROGRESS)

TODO: Find the paper in which demonstrate recommending by news name is much more effective than news body embedding

TODO: Add another tagging approach and run CF as a new baseline: search for established category top-5 keyword and assign the most similar category to each contract

Question: should we keep user-user edges in our transaction graph too?

Just load the user-contract-interactions, ignore the user-user interactions in user_transactions. In other words we are ignoring the social relationship, but one level of ablation can be these social relationships (basically it enhances user representation).
now nodes are contract and users and edges just can exist between user anc contracts.
Node features: address, code |||| edge features: timestamp, function naem (and input), 






Goal: FInish the first draft by mid Oct. 
To do so: Find a decent way to evaluate the GNN model (the current one is supr slow and very low AP@K). Maybe first test Movielens dataa and compare the AP@k.
Then run the user/item CSP eval, then run the diversity eval of models. Then write the result section of experiment chapter. Then write the summery section of conclusion, then rewrite the future work. At the end rewrite the previous sections (remove hard words and check the correctness). Finally add the blank citation and increase number of them.

