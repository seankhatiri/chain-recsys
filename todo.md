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



GNN model result for f=1 (f is the number of top_keywords we considered as part of contract_feat):

Epoch: 001, Loss: 0.3561
Epoch: 002, Loss: 0.2615
Epoch: 003, Loss: 0.2268
Epoch: 004, Loss: 0.2014
Epoch: 005, Loss: 0.1817

AP@1: 0.5978497312164021
AP@2: 0.4682710338792349
AP@3: 0.376030337125474
AP@4: 0.30758219777472184
AP@5: 0.25652873275826144

for f = 5
Epoch: 001, Loss: 0.3390
Epoch: 002, Loss: 0.2384
Epoch: 003, Loss: 0.2038
Epoch: 004, Loss: 0.1818
Epoch: 005, Loss: 0.1648

AP@1: 0.6026836687919324
AP@2: 0.47208401050131266
AP@3: 0.37796391215568614
AP@4: 0.3098449806225778
AP@5: 0.2584489727882653

for f = 150
Epoch: 001, Loss: 0.3278
Epoch: 002, Loss: 0.2341
Epoch: 003, Loss: 0.2014
Epoch: 004, Loss: 0.1793
Epoch: 005, Loss: 0.1656

AP@1: 0.5984331374755177
AP@2: 0.4709130307955161
AP@3: 0.37887513716992405
AP@4: 0.3106242446972538
AP@5: 0.25916572904946455

Runing the GNN for movieLens100k:
Epoch: 001, Loss: 0.4408
Epoch: 002, Loss: 0.3486
Epoch: 003, Loss: 0.3267
Epoch: 004, Loss: 0.3105
Epoch: 005, Loss: 0.3008

AP@1: 0.8058778802605044
AP@2: 0.7927865383979635
AP@3: 0.7640032177372255
AP@4: 0.7404129062117756
AP@5: 0.7144434526761215

Running LightFm with GNN neg_sample eval for MovieLens100k:
AP@1: 0.9713679745493107
AP@2: 0.9644750795334041
AP@3: 0.9494521032166843
AP@4: 0.9318663838812301
AP@5: 0.9117709437963945

LighFm with lighfm Precision_at_k eval with just having users with >10 interactions:
Precision at k=1: 0.11810652166604996
Precision at k=2: 0.10770318657159805
Precision at k=3: 0.09884855896234512
Precision at k=4: 0.089724600315094
Precision at k=5: 0.08322671800851822

Ablation study result: (given f = 150, contract_feat.shape = num_of_contracts, all_unique_keywords, we put zero for all)
Epoch: 001, Loss: 0.3655
Epoch: 002, Loss: 0.2729
Epoch: 003, Loss: 0.2364
Epoch: 004, Loss: 0.2093
Epoch: 005, Loss: 0.1902
Validation AUC: 0.9574

AP@1: 0.5992165687377589
AP@2: 0.46865024794766014
AP@3: 0.37563584336931
AP@4: 0.30736342042755344
AP@5: 0.25676709588698593


Abblation study (remove movie_feat from MovieLens GNN model):
Epoch: 001, Loss: 0.4429
Epoch: 002, Loss: 0.3515
Epoch: 003, Loss: 0.3276
Epoch: 004, Loss: 0.3147
Epoch: 005, Loss: 0.2983

AP@1: 0.7851499223114814
AP@2: 0.7704717511322688
AP@3: 0.7300296428532072
AP@4: 0.6998991702205032
AP@5: 0.6747528843928724


The MF model running in tmux: the GNN eval for lightfm model on contract dataset (we don't need it)
AP@1: 0.8118550368550369
AP@2: 0.6658323095823095
AP@3: 0.5612612612612613
AP@4: 0.48083538083538085
AP@5: 0.41711916461916465

Oct4: a summary of today:
First we updated the MF recommender eval (to be comparable to GNN). 
Then we generalized the GNN model architecture we had for the user_contract dataset. To do so:
1. we created contract df which has, itemId, contract_name, and contract_top_keywords

for contract top_keywords, we fit a TFIDF on contracts comments to get their n top keywords, then create a tensor with (len(contracts), len(unqiue_keywords in contracts_top_keywords))

2. We trained the GNN model both for movieLens and contract dataset. our observation:
- For both the performance of GNN model was lower than the matrix factorization with the comparable eval
- MovieLens GNN model was closer to its MF counterpart rather than the contract dataset
- When removed the contract_feat (by putting zero for all contracts along the dimenssion) the performance didn't changed drastically which determine the contract_feat dosn't have valuable information for the recommendation task (binary link prediction)

hard decision: do we accept that contracts dosn't have additional information? if yes we need to update the contract_level MF eval to have the comparable eval to name_level and GNN neg_sample eval


Oct5:
We made the universal eval for both name-leve MF and GNN. 
For Contracts:
GNN:
AP@1: 0.5986248281035129
AP@2: 0.46705838229778723
AP@3: 0.37348279646066873
AP@4: 0.3057444680585073
AP@5: 0.2549235321081802

MF:
AP@1: 0.5163062049422844
AP@2: 0.37505938242280285
AP@3: 0.28580517009070583
AP@4: 0.22717839729966247
AP@5: 0.18663999666624997

For MovieLens:
GNN:
AP@1: 0.7718271678402592
AP@2: 0.7664220304803464
AP@3: 0.7477161779452763
AP@4: 0.7173542927038911
AP@5: 0.6859268074977686
Observation: I changed the epoch to as high as 20, the more I increased the epoch (up to 10) the model eval was better, so the model dose not overfit. after 10 the loss just changed .01 for each epoch and eval didn't changed (hit@k). Best Result:
AP@1: 0.8047869351052928
AP@2: 0.7768686568151013
AP@3: 0.7456885627073071
AP@4: 0.7220403980296869
AP@5: 0.6954081126648816

MF:
AP@1: 0.3861615260008595
AP@2: 0.3345399847928857
AP@3: 0.2895412520524095
AP@4: 0.24752884392872493
AP@5: 0.21366656748983437


GNN For MovieLens
with movie_feat:
AP@1: 0.7718271678402592
AP@2: 0.7664220304803464
AP@3: 0.7477161779452763
AP@4: 0.7173542927038911
AP@5: 0.6859268074977686

without movie_feat:
AP@1: 0.8273661939237661
AP@2: 0.7891004661311117
AP@3: 0.7662512259358436
AP@4: 0.7393467552646369
AP@5: 0.7137029323283415



Challange: why without movie_feat the results are better?
one problem can be the num of edge_label_index in val_loader is > train_loader: I think in train_loader we just have the positive edges, but in val_loader we have negative edges.
len(val_loader.edge_label) = 
len(train_loader.edge_label) = 
neg_sample_ratio = 2


Oct5 mid night: There was a bug in item_feat for MovieLens, actually the movie_feat tensor order was not compatible with the item_ids. The itsm_ids was mapped_id but the movie_feat was the original movie_id-1 in movies_df. After fixing the bug:
GNN For MovieLens
with movie_feat:
AP@1: 0.8493503917484876
AP@2: 0.8143574994214685
AP@3: 0.7811387704276724
AP@4: 0.7461816919567589
AP@5: 0.7131541538563257

without movie_feat:
AP@1: 0.8370524645442825
AP@2: 0.8039274025587623
AP@3: 0.7712321068465073
AP@4: 0.7382558101094251
AP@5: 0.713154153856326

GNN for contracts
with contract_feat:
AP@1: 0.5986248281035129
AP@2: 0.46705838229778723
AP@3: 0.37348279646066873
AP@4: 0.3057444680585073
AP@5: 0.2549235321081802


without contract_feat:
AP@1: 0.5837146309955411
AP@2: 0.4531566445805726
AP@3: 0.36130349627036706
AP@4: 0.295380672584073
AP@5: 0.24670750510480477