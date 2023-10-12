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

Note: the reason why the train_loader_df len is less than val_loader_df: we have ~100k interactions. The reason is 30% of edges are used for supervision, we have 80k total train_egde, 0.3*80k=24k for supervision and 0.7*80k=56k for msg passing. Also we have 10k val_edges (positive) and with neg_ratio=2 we'll have 20k neg_edges in val_data and we create the neg samples during the training. ALso there is not any intersection between train and val which makes it transductive link prediction meaning during training model didn't see the positive edges. 

Note: the egde_index in tes_data is 80% of all egdes, so I guess it's train_edges, so the edge_labels_index is the true test edges, keep in mind

Oct 7: We modified the way we get out prediction from GNN. The main intention was to be able to have hit@k eval. Todo so we need the prediction, given a user_id, for all possible items, then rerank and return the top_k. After doing so and updating the NMF and CMF alongside the GNN model and the Hit@k eval we had these results:
''' GNN contract
AP@1: 0.04191263282172373 -> .41
AP@5: 0.030342384887839437 -> .30
'''

''' NMF contract
AP@1: 0.03571428571428571 -> .35
AP@5: 0.024321133412042506 -> .24
'''

''' GNN MovieLens
AP@1: 0.0755336617405583
AP@5: 0.06371100164203612
'''

''' NMF MovieLens
AP@1: 0.07635467980295567
AP@5: 0.06962233169129721
'''

For movieLens we didn't even slice the test_data, since the unique number of users and items are limited (609, 5099), so we calculate the model predictions for all possible pairs (609*5099). However, as we can see the GNN does not show any better performance here. WHY? let's increase the slice_rate from .1 to .4 for contracts:

''' GNN contract
AP@1: 0.059756627553237726
AP@5: 0.04050412863972186
'''

''' NMF contract
AP@1: 0.04758800521512386
AP@5: 0.03489787049109083
'''
As we can see:

Next_step: for movie_lens increase the epoch from 6 to 10:



Oct 8: 
Note: For MF models,we cannot give model a u_id and get the top_k items, instead we have (u_id, i_id) pairs (can be neg or pos). We fed the model and get the pred for each pair, then rank based on pred and see in top_k how many ground_truth == 1 we contain. so basically we don't need items in the last step, just see the ground_truth column.
After fixing the CMF model:
'''
(in prev results did we fed interactions or all of them?) 
CMF contract: 100k interactions, slice_rate=0.1
AP@1: 0.07765263781861292
AP@5: 0.015530527563722585
'''
'''
NMF contract: 100k interactions, slice_rate=0.1
AP@1: 0.024734982332155476
AP@5: 0.018138987043580686
'''
'''
CMF contract: 100k interactions, slice_rate=0.5
AP@1: 0.06323585263957462
AP@5: 0.012647170527914925
'''
NMF contract: 100k interactions, slice_rate=0.5
AP@1: 0.03589061906570452
AP@5: 0.02939612609191037
'''

Then we fixed the CSP bug and run the CSP-user and CSP-item experiments (note that slice_rate for csp is 1):
'''
UCSP, GNN, contract (100k interactions)
AP@1: 0.16943042537851477
AP@5: 0.09891852919971161
'''
'''
UCSP, NMF, contract (100k interactions)
AP@1: 0.10093727469358327
AP@5: 0.07238644556596972
'''
'''
UCSP, CMF, contract (100k interactions)
AP@1: 0.0
AP@5: 0.0
'''
############
'''
ICSP, GNN, contract (100k interactions)
AP@1:
AP@5:
'''
'''
ICSP, NMF, contract (100k interactions)
AP@1:
AP@5:
'''
'''
ICSP, CMF, contract (100k interactions)
AP@1:
AP@5:
'''

Note: we changed the denominator of hit@k from k to min(num_ones, k). the num_ones is the num of positive interactions in ground_truth. It is basically the total number of user actual interaction in testset. Now both hit@k and NDCG@k will grow proportional to k.


Oct 9: added sBERT as contract representation experiment.

For 2k contract interaction data:
''' TFIDF
HIT@1: 0.05625
HIT@5: 0.07854166666666666
NDCG@1: 0.05625
NDCG@5: 0.06638135940822126
MAP@1: 0.05625
MAP@5: 0.07510416666666667
'''
''' sBERT
HIT@1: 0.04
HIT@5: 0.15033333333333335
NDCG@1: 0.04
NDCG@5: 0.09938733033336235
MAP@1: 0.04
MAP@5: 0.10577777777777778
'''

we replaced the senternce-transformers library with transformers since the first package had some conflict with torch==1.13.1 and tested the sBERT embedding as item_feat with the new package. Then added the social edges to the edge_index tensor. to do so we shifted the item mappedID by the len(edge_label_index[0].unique()). Then shifted the item_ids and item_feat ids as same as the previous one. With this trick we could add (user, user) pair as edges since users and items after the previous shifting change has different unique ids. In test time, when wanna create the all2all possible edges between user and items, since all_items now contain both users and items, we skip the item_ids < len(edge_label_index[0].unique()). Since the user ids are before item ids in all_items, we don't consider them. However, when creaing the test and train data, we contain some (user, user) edges in test data edge_index. Therefore though we don't add any new user, user edges to all2all edges, but the test_data edge_index contain some social_edges. If the model dose not work well, we should remove these social edges completely from test_data edge_index.



Oct 10:
To get the result, try to reduce the number of neg edges for each user (instead of getting all possible edges, just add a up to a treshhold (like 100 edges for each unique user, the result should be faster and better))

Observations: 
1. we added pop_rec, but the result for contract dataset is too good (better than NMF and GNN), so there is sth wrong with its eval or pred gen.
2. For both datasets, when we skip the all2all gen, diff metrics in eval always are 1, why? That was because somehow the neg_ration in NeighbourSampler was fixed to 0, I changed that to 2 and that worked.
3. When running one time from start to end, then rerun some sections without restarting, the hit@k will be NaN since the ground_truth somehow will be all 0 somwhow, don't know why.

Oct 11: 

Observation: we used NeighbourLoader neg_sampler for test_data, and skipped internal test_data_all2all gen, there wasn't any difference between MFN and GNN.
Observation: we didn't fed the MF model with weights when calling .fit(), but still is as good as the neural
Observation: when we increased the number of interactions to 300k -> 
$$$$$$ gnn $$$$$$
HIT@1: 0.6035285238065287
HIT@5: 0.6566136234983332
HIT@10: 0.663092018365935
HIT@15: 0.6633121579973583
HIT@20: 0.6633121579973583
NDCG@1: 0.6035285238065287
NDCG@5: 0.6035627068209014
NDCG@10: 0.6203944474899137
NDCG@15: 0.6232462524890124
NDCG@20: 0.623406528235947
MAP@1: 0.6035285238065287
MAP@5: 0.6107831292673893
MAP@10: 0.5985780012692634
MAP@15: 0.5963048260384168
MAP@20: 0.5962010358819575
$$$$$$ mfn $$$$$$
HIT@1: 0.5605698471601988
HIT@5: 0.633278822567457
HIT@10: 0.6589408138876659
HIT@15: 0.6632492609598087
HIT@20: 0.6633121579973583
NDCG@1: 0.5605698471601988
NDCG@5: 0.5404481548760584
NDCG@10: 0.5708759552076074
NDCG@15: 0.5834993774982412
NDCG@20: 0.5850611038286065
MAP@1: 0.5605698471601988
MAP@5: 0.5695563487570847
MAP@10: 0.5475193327966564
MAP@15: 0.536316917220932
MAP@20: 0.5350186298885398
$$$$$$ mfc $$$$$$
HIT@1: 0.09491162966224291
HIT@5: 0.24605321089376692
HIT@10: 0.6168941442858041
HIT@15: 0.6629347757720612
HIT@20: 0.6633121579973583
NDCG@1: 0.09491162966224291
NDCG@5: 0.1118332281349217
NDCG@10: 0.2596835803321522
NDCG@15: 0.3123269298827346
NDCG@20: 0.3178683071415946
MAP@1: 0.09491162966224291
MAP@5: 0.14385316861298056
MAP@10: 0.18855700248455182
MAP@15: 0.18962071055936622
MAP@20: 0.18849444891036
$$$$$$ pop $$$$$$
HIT@1: 0.46040631486257
HIT@5: 0.22120730863576327
HIT@10: 0.21781235367703067
HIT@15: 0.21777004793153606
HIT@20: 0.21776991689604117

Observation: in Data_loader the component that takes most of the time is the item_feat calculation, one good way can be to calculate it once and save it

result for effectiveness 100k:
$$$$$$ gnn $$$$$$
HIT@1: 0.5545463058338795
HIT@5: 0.6414458282610731
HIT@10: 0.6580204138964323
HIT@15: 0.6621406498735837
HIT@20: 0.662421575053844
NDCG@1: 0.5545463058338795
NDCG@5: 0.5430931079896212
NDCG@10: 0.5725421740251601
NDCG@15: 0.5839337962678809
NDCG@20: 0.5861671580583855
MAP@1: 0.5545463058338795
MAP@5: 0.5684420877943211
MAP@10: 0.5474776226442966
MAP@15: 0.5382535007302091
MAP@20: 0.5362291983476681
$$$$$$ mfn $$$$$$
HIT@1: 0.5095982769922277
HIT@5: 0.6002434684895589
HIT@10: 0.6365764584698942
HIT@15: 0.658394980803446
HIT@20: 0.662421575053844
NDCG@1: 0.5095982769922277
NDCG@5: 0.4727385037227605
NDCG@10: 0.504153977502396
NDCG@15: 0.5288602697607397
NDCG@20: 0.5403127961330417
MAP@1: 0.5095982769922277
MAP@5: 0.526096518608692
MAP@10: 0.5013454434570421
MAP@15: 0.4810087881028189
MAP@20: 0.470341854896513
$$$$$$ mfc $$$$$$
HIT@1: 0.10740706058619721
HIT@5: 0.3077067141118082
HIT@10: 0.4745762711864407
HIT@15: 0.6409776196273059
HIT@20: 0.6623279333270905
NDCG@1: 0.10740706058619721
NDCG@5: 0.138349568785159
NDCG@10: 0.20336856555967817
NDCG@15: 0.2803961777785049
NDCG@20: 0.30890019934656926
MAP@1: 0.10740706058619721
MAP@5: 0.17414174756271392
MAP@10: 0.18748665623088726
MAP@15: 0.18773516240894944
MAP@20: 0.18048903355006354
$$$$$$ pop $$$$$$
HIT@1: 0.4634329057027812
HIT@5: 0.22506476886100446
HIT@10: 0.22169957504492568
HIT@15: 0.22164617306718626
HIT@20: 0.22164617306718626

contract rep experiment:
sBERT:
$$$$$$ gnn $$$$$$
HIT@1: 0.5545463058338795
HIT@5: 0.6414458282610731
HIT@10: 0.6580204138964323
HIT@15: 0.6621406498735837
HIT@20: 0.662421575053844
NDCG@1: 0.5545463058338795
NDCG@5: 0.5430931079896212
NDCG@10: 0.5725421740251601
NDCG@15: 0.5839337962678809
NDCG@20: 0.5861671580583855
MAP@1: 0.5545463058338795
MAP@5: 0.5684420877943211
MAP@10: 0.5474776226442966
MAP@15: 0.5382535007302091
MAP@20: 0.5362291983476681

TFIDF:
HIT@1: 0.5428625023447758
HIT@5: 0.6344025511161133
HIT@10: 0.6539110861001688
HIT@15: 0.6586006377790283
HIT@20: 0.65888201087976
NDCG@1: 0.5428625023447758
NDCG@5: 0.5324252582144876
NDCG@10: 0.5633633600123634
NDCG@15: 0.5757228555503907
NDCG@20: 0.5784592047295187
MAP@1: 0.5428625023447758
MAP@5: 0.5599634736030347
MAP@10: 0.5388797402901704
MAP@15: 0.5288210458179221
MAP@20: 0.5263080748159996

Question: when creating social egdes, the user ids are the same, but we are creating essentially new nodes, so how the model can learn anything from them? in reality the edge is between two users, but in edge_index its between a user to a new item (need to be answered).

GNN Full (with social edges): (it's just with a portion of social edges-200k user transactions, run for all)
$$$$$$ gnn $$$$$$
HIT@1: 0.5441851747727485
HIT@5: 0.6439883797207384
HIT@10: 0.6612313747540062
HIT@15: 0.6645112922875082
HIT@20: 0.6651672757942086
NDCG@1: 0.5441851747727485
NDCG@5: 0.5438047366114123
NDCG@10: 0.5741196367583175
NDCG@15: 0.5846468621739428
NDCG@20: 0.5868769006794022
MAP@1: 0.5441851747727485
MAP@5: 0.5657662512104458
MAP@10: 0.5464664087563964
MAP@15: 0.5389417997575754
MAP@20: 0.5371799291397468

GNN without social edges:
$$$$$$ gnn $$$$$$
HIT@1: 0.5545463058338795
HIT@5: 0.6414458282610731
HIT@10: 0.6580204138964323
HIT@15: 0.6621406498735837
HIT@20: 0.662421575053844
NDCG@1: 0.5545463058338795
NDCG@5: 0.5430931079896212
NDCG@10: 0.5725421740251601
NDCG@15: 0.5839337962678809
NDCG@20: 0.5861671580583855
MAP@1: 0.5545463058338795
MAP@5: 0.5684420877943211
MAP@10: 0.5474776226442966
MAP@15: 0.5382535007302091
MAP@20: 0.5362291983476681


GNN without item_feat:
$$$$$$ gnn $$$$$$
HIT@1: 0.5377552932359003
HIT@5: 0.6327524826681656
HIT@10: 0.6609518456061457
HIT@15: 0.6707888326775342
HIT@20: 0.6210043095371937
NDCG@1: 0.5377552932359003
NDCG@5: 0.5191332845281006
NDCG@10: 0.5502206846158535
NDCG@15: 0.5684698256234662
NDCG@20: 0.574111501550854
MAP@1: 0.5377552932359003
MAP@5: 0.5563746538837883
MAP@10: 0.5354388897758217
MAP@15: 0.5197741805898527
MAP@20: 0.5141720197244387

GNN without item_feat and social edges:
$$$$$$ gnn $$$$$$
HIT@1: 0.5299089116348953
HIT@5: 0.6224997652361725
HIT@10: 0.6472908254296178
HIT@15: 0.6546154568504085
HIT@20: 0.6555545121607663
NDCG@1: 0.5299089116348953
NDCG@5: 0.51194176412685
NDCG@10: 0.543318232185577
NDCG@15: 0.5596857319716052
NDCG@20: 0.5643333725769923
MAP@1: 0.5299089116348953
MAP@5: 0.5461479168623032
MAP@10: 0.5243807197435444
MAP@15: 0.511145426929444
MAP@20: 0.5069065871935584


Observation: for sparsity analysis, when we have usparsity, we just keep users with interactions > u, but in this case the number of unique items will be more than normal (with u=1). So if we put a limit on item_ratings to just be limited to 100k, the number of unique_items decrease and the following result: unique_users=5898, unique_items=16877
user sparsity, u=5
$$$$$$ gnn $$$$$$
HIT@1: 0.7183791115632417
HIT@5: 0.865038996269922
HIT@10: 0.8999660902000678
HIT@15: 0.9133604611732791
HIT@20: 0.917090539165819
After fixing the embedding (calculate full and slice until reach len(unique_item_ids)) for 150k interactions:
number of unique users: 12215
number of unique items: 20879
$$$$$$ gnn $$$$$$
HIT@1: 0.7220548596797209
HIT@5: 0.8620580307594736
HIT@10: 0.8953543681623592
HIT@15: 0.9107341049627398
HIT@20: 0.9127953068019661
Therefore we kep the above one, but still it's high, what other things we can do? We should have the same item_df as before, just the users will change in u-sparsity. But we dicided to change the each_user_all2all_new_edges to 20 from 10 which yield the GNN results for u=5 slightly better than the base (u=1)

u=1, u_sparsity=0.9463153284071205
number of unique users 10761
number of unique items 17310
$$$$$$ gnn $$$$$$
HIT@1: 0.5545463058338795
HIT@5: 0.6414458282610731
HIT@10: 0.6580204138964323
HIT@15: 0.6621406498735837
HIT@20: 0.662421575053844
NDCG@1: 0.5545463058338795
NDCG@5: 0.5430931079896212
NDCG@10: 0.5725421740251601
NDCG@15: 0.5839337962678809
NDCG@20: 0.5861671580583855
MAP@1: 0.5545463058338795
MAP@5: 0.5684420877943211
MAP@10: 0.5474776226442966
MAP@15: 0.5382535007302091
MAP@20: 0.5362291983476681
$$$$$$ mfn $$$$$$
HIT@1: 0.5095982769922277
HIT@5: 0.6002434684895589
HIT@10: 0.6365764584698942
HIT@15: 0.658394980803446
HIT@20: 0.662421575053844
NDCG@1: 0.5095982769922277
NDCG@5: 0.4727385037227605
NDCG@10: 0.504153977502396
NDCG@15: 0.5288602697607397
NDCG@20: 0.5403127961330417
MAP@1: 0.5095982769922277
MAP@5: 0.526096518608692
MAP@10: 0.5013454434570421
MAP@15: 0.4810087881028189
MAP@20: 0.470341854896513
$$$$$$ mfc $$$$$$
HIT@1: 0.10740706058619721
HIT@5: 0.3077067141118082
HIT@10: 0.4745762711864407
HIT@15: 0.6409776196273059
HIT@20: 0.6623279333270905
NDCG@1: 0.10740706058619721
NDCG@5: 0.138349568785159
NDCG@10: 0.20336856555967817
NDCG@15: 0.2803961777785049
NDCG@20: 0.30890019934656926
MAP@1: 0.10740706058619721
MAP@5: 0.17414174756271392
MAP@10: 0.18748665623088726
MAP@15: 0.18773516240894944
MAP@20: 0.18048903355006354
$$$$$$ pop $$$$$$
HIT@1: 0.4634329057027812
HIT@5: 0.22506476886100446
HIT@10: 0.22169957504492568
HIT@15: 0.22164617306718626
HIT@20: 0.22164617306718626


u=5, u_sparsity: 0.9140886147155499
number of unique users 6309
number of unique items 17013
$$$$$$ gnn $$$$$$
HIT@1: 0.6829925503249327
HIT@5: 0.8299255032493263
HIT@10: 0.8695514344587097
HIT@15: 0.8895229037882391
HIT@20: 0.8971310825804406
NDCG@1: 0.6829925503249327
NDCG@5: 0.6262517253135895
NDCG@10: 0.6698598411566745
NDCG@15: 0.6993737743981807
NDCG@20: 0.7145999450445164
MAP@1: 0.6829925503249327
MAP@5: 0.7089376287842765
MAP@10: 0.674477915252503
MAP@15: 0.6486701917439951
MAP@20: 0.6326233152364615
$$$$$$ pop $$$$$$
HIT@1: 0.6493897606593755
HIT@5: 0.25553706345432453
HIT@10: 0.2500962344043656
HIT@15: 0.249994550279857
HIT@20: 0.24999330711338766
$$$$$$ mfn $$$$$$
HIT@1: 0.6394040259946109
HIT@5: 0.7703281027104137
HIT@10: 0.8218418132826122
HIT@15: 0.8535425582501189
HIT@20: 0.8771596132509114
NDCG@1: 0.6394040259946109
NDCG@5: 0.5502465603343888
NDCG@10: 0.5855631055762114
NDCG@15: 0.6141816789540975
NDCG@20: 0.6364193605634272
MAP@1: 0.6394040259946109
MAP@5: 0.6639397421672742
MAP@10: 0.6311219284961304
MAP@15: 0.604053804119079
MAP@20: 0.5812870093924087
$$$$$$ mfc $$$$$$
HIT@1: 0.17530511967031226
HIT@5: 0.5178316690442225
HIT@10: 0.5782215882073228
HIT@15: 0.6926612775400222
HIT@20: 0.802504358852433
NDCG@1: 0.17530511967031226
NDCG@5: 0.21862699991926218
NDCG@10: 0.2468159096300633
NDCG@15: 0.2927600931542803
NDCG@20: 0.33781200966907016
MAP@1: 0.17530511967031226
MAP@5: 0.2850553001884433
MAP@10: 0.28824882247387135
MAP@15: 0.27787243083830354
MAP@20: 0.2680721408726483

u=10, u_sparsity: 0.8839256647841233
number of unique users 4271
number of unique items 15707
$$$$$$ gnn $$$$$$
HIT@1: 0.7145867478342308
HIT@5: 0.8859751814563334
HIT@10: 0.9239054085694217
HIT@15: 0.9435729337391712
HIT@20: 0.9524701475064388
NDCG@1: 0.7145867478342308
NDCG@5: 0.6301113557067942
NDCG@10: 0.6804236065750341
NDCG@15: 0.7171777712918381
NDCG@20: 0.7389532033530473
MAP@1: 0.7145867478342308
MAP@5: 0.7457614662192045
MAP@10: 0.6992985375966597
MAP@15: 0.6673086987337897
MAP@20: 0.6439228338626841
$$$$$$ pop $$$$$$
HIT@1: 0.7068602200889721
HIT@5: 0.23004370561148832
HIT@10: 0.2199213596310295
HIT@15: 0.21978297853970996
HIT@20: 0.21978297853970996
$$$$$$ mfn $$$$$$
HIT@1: 0.6663544837274643
HIT@5: 0.8276750175602904
HIT@10: 0.8784827909154764
HIT@15: 0.9126668227581363
HIT@20: 0.9360805431983142
NDCG@1: 0.6663544837274643
NDCG@5: 0.5504888067187561
NDCG@10: 0.5887714967699231
NDCG@15: 0.6250837668822511
NDCG@20: 0.6512018667058097
MAP@1: 0.6663544837274643
MAP@5: 0.6965146075600301
MAP@10: 0.6534248547880349
MAP@15: 0.6174318731413215
MAP@20: 0.5910248112209212
$$$$$$ mfc $$$$$$
HIT@1: 0.19409974244907516
HIT@5: 0.6038398501521892
HIT@10: 0.6658862093186607
HIT@15: 0.781315851088738
HIT@20: 0.8665417934909857
NDCG@1: 0.19409974244907516
NDCG@5: 0.2416114755186639
NDCG@10: 0.2735048474266584
NDCG@15: 0.3211516474950826
NDCG@20: 0.3639552446036534
MAP@1: 0.19409974244907516
MAP@5: 0.3265183147324332
MAP@10: 0.32822540601479544
MAP@15: 0.31392393079403896
MAP@20: 0.3003817519504908

u=15, u_sparsity: 0.8094774120141142
number of unique users 2396
number of unique items 12184
$$$$$$ gnn $$$$$$
HIT@1: 0.7166110183639399
HIT@5: 0.9131886477462438
HIT@10: 0.9590984974958264
HIT@15: 0.9766277128547579
HIT@20: 0.9816360601001669
NDCG@1: 0.7166110183639399
NDCG@5: 0.5973199389093855
NDCG@10: 0.656217538618894
NDCG@15: 0.7028840023881509
NDCG@20: 0.7303212304974399
MAP@1: 0.7166110183639399
MAP@5: 0.74991768688555
MAP@10: 0.6903980114854017
MAP@15: 0.6524156185820184
MAP@20: 0.6256359837459585
$$$$$$ pop $$$$$$
HIT@1: 0.6807178631051753
HIT@5: 0.18875208681135225
HIT@10: 0.17588706044465643
HIT@15: 0.17565238971749822
HIT@20: 0.17564653200142175
$$$$$$ mfn $$$$$$
HIT@1: 0.6828046744574291
HIT@5: 0.8710350584307178
HIT@10: 0.9173622704507512
HIT@15: 0.9444908180300501
HIT@20: 0.9682804674457429
NDCG@1: 0.6828046744574291
NDCG@5: 0.5536674420041156
NDCG@10: 0.5848172165638815
NDCG@15: 0.6224377228783049
NDCG@20: 0.6527927039613315
MAP@1: 0.6828046744574291
MAP@5: 0.7212275088109813
MAP@10: 0.6771871441799994
MAP@15: 0.6400716447600351
MAP@20: 0.6086864464144291
$$$$$$ mfc $$$$$$
HIT@1: 0.17779632721202004
HIT@5: 0.6585976627712855
HIT@10: 0.7232888146911519
HIT@15: 0.843906510851419
HIT@20: 0.924457429048414
NDCG@1: 0.17779632721202004
NDCG@5: 0.252939916979089
NDCG@10: 0.27788314880181764
NDCG@15: 0.3274287367964599
NDCG@20: 0.3769338297256061
MAP@1: 0.17779632721202004
MAP@5: 0.352515767019106
MAP@10: 0.35787099828028657
MAP@15: 0.3401511619558066
MAP@20: 0.322056360116857

u=20, u_sparsity: 0.6726008024660516
number of unique users 1319
number of unique items 8725
$$$$$$ gnn $$$$$$
HIT@1: 0.6747536012130402
HIT@5: 0.9219105382865808
HIT@10: 0.9719484457922669
HIT@15: 0.9863532979529946
HIT@20: 0.9916603487490523
NDCG@1: 0.6747536012130402
NDCG@5: 0.5504258785229257
NDCG@10: 0.5946450084365762
NDCG@15: 0.6551940372553726
NDCG@20: 0.6925590271922977
MAP@1: 0.6747536012130402
MAP@5: 0.7237637941201246
MAP@10: 0.6645439369623681
MAP@15: 0.6188744759661907
MAP@20: 0.5876966707648881
$$$$$$ pop $$$$$$
HIT@1: 0.5959059893858984
HIT@5: 0.1344452868334597
HIT@10: 0.11095617170294955
HIT@15: 0.11043619635355799
HIT@20: 0.11043619635355799
$$$$$$ mfn $$$$$$
HIT@1: 0.6520090978013646
HIT@5: 0.8786959818043972
HIT@10: 0.9302501895375285
HIT@15: 0.9575435936315391
HIT@20: 0.9772554965883244
NDCG@1: 0.6520090978013646
NDCG@5: 0.5220710651829621
NDCG@10: 0.546545821403372
NDCG@15: 0.5913756366436133
NDCG@20: 0.6251622407420596
MAP@1: 0.6520090978013646
MAP@5: 0.7051922752927301
MAP@10: 0.6536337480475581
MAP@15: 0.6145093031979189
MAP@20: 0.5827714021432491
$$$$$$ mfc $$$$$$
HIT@1: 0.1652767247915087
HIT@5: 0.66868840030326
HIT@10: 0.7293404094010614
HIT@15: 0.8756633813495072
HIT@20: 0.9484457922668689
NDCG@1: 0.1652767247915087
NDCG@5: 0.2539879425522662
NDCG@10: 0.27089585663154403
NDCG@15: 0.32182320604298553
NDCG@20: 0.37889382216959383
MAP@1: 0.1652767247915087
MAP@5: 0.36080679807935306
MAP@10: 0.3675062151260976
MAP@15: 0.352176176232866
MAP@20: 0.33097251270082306

item sparsity, 
i=1, i_sparsity=0.9463153284071205
number of unique users 10761
number of unique items 17310
$$$$$$ gnn $$$$$$
HIT@1: 0.5545463058338795
HIT@5: 0.6414458282610731
HIT@10: 0.6580204138964323
HIT@15: 0.6621406498735837
HIT@20: 0.662421575053844
NDCG@1: 0.5545463058338795
NDCG@5: 0.5430931079896212
NDCG@10: 0.5725421740251601
NDCG@15: 0.5839337962678809
NDCG@20: 0.5861671580583855
MAP@1: 0.5545463058338795
MAP@5: 0.5684420877943211
MAP@10: 0.5474776226442966
MAP@15: 0.5382535007302091
MAP@20: 0.5362291983476681
$$$$$$ mfn $$$$$$
HIT@1: 0.5095982769922277
HIT@5: 0.6002434684895589
HIT@10: 0.6365764584698942
HIT@15: 0.658394980803446
HIT@20: 0.662421575053844
NDCG@1: 0.5095982769922277
NDCG@5: 0.4727385037227605
NDCG@10: 0.504153977502396
NDCG@15: 0.5288602697607397
NDCG@20: 0.5403127961330417
MAP@1: 0.5095982769922277
MAP@5: 0.526096518608692
MAP@10: 0.5013454434570421
MAP@15: 0.4810087881028189
MAP@20: 0.470341854896513
$$$$$$ mfc $$$$$$
HIT@1: 0.10740706058619721
HIT@5: 0.3077067141118082
HIT@10: 0.4745762711864407
HIT@15: 0.6409776196273059
HIT@20: 0.6623279333270905
NDCG@1: 0.10740706058619721
NDCG@5: 0.138349568785159
NDCG@10: 0.20336856555967817
NDCG@15: 0.2803961777785049
NDCG@20: 0.30890019934656926
MAP@1: 0.10740706058619721
MAP@5: 0.17414174756271392
MAP@10: 0.18748665623088726
MAP@15: 0.18773516240894944
MAP@20: 0.18048903355006354
$$$$$$ pop $$$$$$
HIT@1: 0.4634329057027812
HIT@5: 0.22506476886100446
HIT@10: 0.22169957504492568
HIT@15: 0.22164617306718626
HIT@20: 0.22164617306718626

i=5, i_sparsity=0.8043798137624192
number of unique users 10646
number of unique items 3724
$$$$$$ gnn $$$$$$
HIT@1: 0.4705374280230326
HIT@5: 0.5833013435700576
HIT@10: 0.6089251439539347
HIT@15: 0.6189059500959693
HIT@20: 0.6234165067178503
NDCG@1: 0.4705374280230326
NDCG@5: 0.4699046301577729
NDCG@10: 0.49733061416906754
NDCG@15: 0.5115901055277804
NDCG@20: 0.5179538754371333
MAP@1: 0.4705374280230326
MAP@5: 0.4987534655576882
MAP@10: 0.4817759064491653
MAP@15: 0.47062525826716983
MAP@20: 0.46520315008651114
$$$$$$ pop $$$$$$
HIT@1: 0.48099808061420346
HIT@5: 0.26212252079334614
HIT@10: 0.2605397130061237
HIT@15: 0.2605191066131565
HIT@20: 0.2605191066131565
$$$$$$ mfn $$$$$$
HIT@1: 0.43464491362763913
HIT@5: 0.5370441458733205
HIT@10: 0.5735124760076775
HIT@15: 0.5952015355086372
HIT@20: 0.6119961612284069
NDCG@1: 0.43464491362763913
NDCG@5: 0.4169618958249574
NDCG@10: 0.4409731332371725
NDCG@15: 0.4570309345834593
NDCG@20: 0.4696560333793094
MAP@1: 0.43464491362763913
MAP@5: 0.4622583439965877
MAP@10: 0.448373635659567
MAP@15: 0.43632629335910494
MAP@20: 0.42461105397968185
$$$$$$ mfc $$$$$$
HIT@1: 0.09596928982725528
HIT@5: 0.26641074856046065
HIT@10: 0.3330134357005758
HIT@15: 0.45153550863723607
HIT@20: 0.5682341650671785
NDCG@1: 0.09596928982725528
NDCG@5: 0.12143649910737639
NDCG@10: 0.14627267576887298
NDCG@15: 0.18507803911491225
NDCG@20: 0.22828207060782446
MAP@1: 0.09596928982725528
MAP@5: 0.15087292066538707
MAP@10: 0.15457111369108673
MAP@15: 0.154997471978221
MAP@20: 0.1514265778522815
Note: why when sparsity decrease the effectiveness decrease?

item sparsity, i=10, i_sparsity=0.6320603848328488
number of unique users 10571
number of unique items 1646

$$$$$$ gnn $$$$$$
HIT@1: 0.42678974761857996
HIT@5: 0.5566139644505549
HIT@10: 0.5875478739074929
HIT@15: 0.5976627712854758
HIT@20: 0.6024747127565551
NDCG@1: 0.42678974761857996
NDCG@5: 0.44822807906979917
NDCG@10: 0.47495448651703587
NDCG@15: 0.4859835102185614
NDCG@20: 0.4914126657096388
MAP@1: 0.42678974761857996
MAP@5: 0.46385451787838117
MAP@10: 0.4511045579271714
MAP@15: 0.44392282709821246
MAP@20: 0.43985781990930517
$$$$$$ pop $$$$$$
HIT@1: 0.4850240597073554
HIT@5: 0.28202723493404036
HIT@10: 0.28102802055713766
HIT@15: 0.2810168611381417
HIT@20: 0.2810168611381417
$$$$$$ mfn $$$$$$
HIT@1: 0.3867229696553079
HIT@5: 0.5138957085338309
HIT@10: 0.55651576156339
HIT@15: 0.5792988313856428
HIT@20: 0.595011293332024
NDCG@1: 0.3867229696553079
NDCG@5: 0.3972879026627016
NDCG@10: 0.424090469759126
NDCG@15: 0.4392602680393322
NDCG@20: 0.4497517914941575
MAP@1: 0.3867229696553079
MAP@5: 0.4257599812323371
MAP@10: 0.4160170066913799
MAP@15: 0.4060863546283808
MAP@20: 0.3981224920219709
$$$$$$ mfc $$$$$$
HIT@1: 0.09810468427771776
HIT@5: 0.2370617696160267
HIT@10: 0.3121869782971619
HIT@15: 0.43307473239713246
HIT@20: 0.5519984287538053
NDCG@1: 0.09810468427771776
NDCG@5: 0.1164388945453228
NDCG@10: 0.1429144230234293
NDCG@15: 0.1810531709991729
NDCG@20: 0.22358957329138807
MAP@1: 0.09810468427771776
MAP@5: 0.1416426615164708
MAP@10: 0.14642150618362723
MAP@15: 0.14786552476343
MAP@20: 0.14493023528491103

i=15, i_sparsity=0.4313474360231395
number of unique users 10528
number of unique items 930
$$$$$$ gnn $$$$$$
HIT@1: 0.40028033640368443
HIT@5: 0.5502603123748498
HIT@10: 0.5821986383660392
HIT@15: 0.5902082498998799
HIT@20: 0.5937124549459352
NDCG@1: 0.40028033640368443
NDCG@5: 0.44250369489151753
NDCG@10: 0.4681677270507377
NDCG@15: 0.47720988637683404
NDCG@20: 0.480398912404129
MAP@1: 0.40028033640368443
MAP@5: 0.4479921739420638
MAP@10: 0.43830558562439204
MAP@15: 0.43208141000158384
MAP@20: 0.4295334189943048
$$$$$$ pop $$$$$$
HIT@1: 0.4813776531838206
HIT@5: 0.30072420237618475
HIT@10: 0.2999803335431088
HIT@15: 0.29997441735277136
HIT@20: 0.29997441735277136
$$$$$$ mfn $$$$$$
HIT@1: 0.34311173408089707
HIT@5: 0.4988986784140969
HIT@10: 0.5461553864637565
HIT@15: 0.5690828994793753
HIT@20: 0.5860032038446136
NDCG@1: 0.34311173408089707
NDCG@5: 0.37838667714880697
NDCG@10: 0.40452684039516046
NDCG@15: 0.4187091702860347
NDCG@20: 0.42887072587978187
MAP@1: 0.34311173408089707
MAP@5: 0.39542770769367686
MAP@10: 0.3893725951867865
MAP@15: 0.3816497484839585
MAP@20: 0.37462884641294475
$$$$$$ mfc $$$$$$
HIT@1: 0.0947136563876652
HIT@5: 0.22046455746896276
HIT@10: 0.302362835402483
HIT@15: 0.434521425710853
HIT@20: 0.5444533440128154
NDCG@1: 0.0947136563876652
NDCG@5: 0.11381673489035105
NDCG@10: 0.14182461558948148
NDCG@15: 0.18368597765318737
NDCG@20: 0.22107406737912688
MAP@1: 0.0947136563876652
MAP@5: 0.13458316646642635
MAP@10: 0.14008527163813508
MAP@15: 0.14293345689891782
MAP@20: 0.14106494770598568

i=20, i_sparsity=0.18271759056751213
number of unique users 10470
number of unique items 583

$$$$$$ gnn $$$$$$
HIT@1: 0.3691233832888524
HIT@5: 0.5233011701909259
HIT@10: 0.5579963046602341
HIT@15: 0.5707246971874358
HIT@20: 0.5765756518168754
NDCG@1: 0.3691233832888524
NDCG@5: 0.41543965155979606
NDCG@10: 0.44164258232413345
NDCG@15: 0.45105895247032224
NDCG@20: 0.4549109593203536
MAP@1: 0.3691233832888524
MAP@5: 0.4195121638267297
MAP@10: 0.4105199938198932
MAP@15: 0.4052145331054399
MAP@20: 0.4021019145660685
$$$$$$ pop $$$$$$
HIT@1: 0.5024635598439745
HIT@5: 0.3160490659002258
HIT@10: 0.3153598964392436
HIT@15: 0.3153598964392436
HIT@20: 0.3153598964392436
$$$$$$ mfn $$$$$$
HIT@1: 0.30065694929172654
HIT@5: 0.4759802915212482
HIT@10: 0.5252514884007391
HIT@15: 0.5511188667624718
HIT@20: 0.5675425990556354
NDCG@1: 0.30065694929172654
NDCG@5: 0.35245430172649717
NDCG@10: 0.3800927515452011
NDCG@15: 0.393551025351554
NDCG@20: 0.40303361334805504
MAP@1: 0.30065694929172654
MAP@5: 0.36188895732110676
MAP@10: 0.35599902065440026
MAP@15: 0.35046528222129264
MAP@20: 0.3451824532161969
$$$$$$ mfc $$$$$$
HIT@1: 0.09022787928556765
HIT@5: 0.20314103880106754
HIT@10: 0.2972695545062615
HIT@15: 0.42773557791008004
HIT@20: 0.5321289262985013
NDCG@1: 0.09022787928556765
NDCG@5: 0.10890309533975585
NDCG@10: 0.13991627481542507
NDCG@15: 0.17986647316691948
NDCG@20: 0.21574893450018512
MAP@1: 0.09022787928556765
MAP@5: 0.1257521841282876
MAP@10: 0.13313203331128293
MAP@15: 0.137504155017371
MAP@20: 0.1360612534787237

observation: the GNN without social edges <-> GNN full

result for 100k interactions, user CSP, first generate the train/test, then mask the train user/item in testset:
$$$$$$ gnn $$$$$$
HIT@1: 0.3684536721413077
HIT@5: 0.40842888131391386
HIT@10: 0.41834521227145954
HIT@15: 0.42113418035326927
HIT@20: 0.4220638363805392
NDCG@1: 0.3684536721413077
NDCG@5: 0.3790875522907137
NDCG@10: 0.3869377630068727
NDCG@15: 0.3889766068366397
NDCG@20: 0.3893633301610874
MAP@1: 0.3684536721413077
MAP@5: 0.37970423165650935
MAP@10: 0.3759462385822007
MAP@15: 0.3746157328082712
MAP@20: 0.37441603562047465
$$$$$$ pop $$$$$$
HIT@1: 0.31360396653238304
HIT@5: 0.25030472058671627
HIT@10: 0.2502043767615506
HIT@15: 0.2502043767615506
HIT@20: 0.2502043767615506
$$$$$$ mfn $$$$$$
HIT@1: 0.3532692903625658
HIT@5: 0.39107530213820885
HIT@10: 0.40904865199876045
HIT@15: 0.4198946389835761
HIT@20: 0.4217539510381159
NDCG@1: 0.3532692903625658
NDCG@5: 0.35672056981883904
NDCG@10: 0.366373228577383
NDCG@15: 0.3727862738214699
NDCG@20: 0.37439445140165195
MAP@1: 0.3532692903625658
MAP@5: 0.36380151843817793
MAP@10: 0.36059141986512616
MAP@15: 0.355289130991962
MAP@20: 0.35374710910828194
$$$$$$ mfc $$$$$$
HIT@1: 0.0077471335605825845
HIT@5: 0.03842578246048962
HIT@10: 0.19832661915091415
HIT@15: 0.3966532383018283
HIT@20: 0.4217539510381159
NDCG@1: 0.0077471335605825845
NDCG@5: 0.015457793965272656
NDCG@10: 0.06438639570917135
NDCG@15: 0.12515884062981955
NDCG@20: 0.13514275467218953
MAP@1: 0.0077471335605825845
MAP@5: 0.018212650208311813
MAP@10: 0.03704452300494622
MAP@15: 0.052811397243311006
MAP@20: 0.053912979493515825

item CSP:
$$$$$$ gnn $$$$$$
HIT@1: 0.08260408379740122
HIT@5: 0.20697427738000532
HIT@10: 0.24197825510474674
HIT@15: 0.24635375232033943
HIT@20: 0.24635375232033943
NDCG@1: 0.08260408379740122
NDCG@5: 0.13857546524723763
NDCG@10: 0.1554120039366199
NDCG@15: 0.1580545285911231
NDCG@20: 0.15810530946016338
MAP@1: 0.08260408379740122
MAP@5: 0.12555724977164914
MAP@10: 0.12707800038604675
MAP@15: 0.12656980890679323
MAP@20: 0.12654992028308598
$$$$$$ pop $$$$$$
HIT@1: 0.0
HIT@5: 0.0
HIT@10: 0.0
HIT@15: 0.0
HIT@20: 0.0
$$$$$$ mfn $$$$$$
HIT@1: 0.02532484752055158
HIT@5: 0.11190665605940069
HIT@10: 0.20392468841156192
HIT@15: 0.2462211614956245
HIT@20: 0.24635375232033943
NDCG@1: 0.02532484752055158
NDCG@5: 0.058480702454557214
NDCG@10: 0.09079606385854824
NDCG@15: 0.1081716025658536
NDCG@20: 0.10863400391667508
MAP@1: 0.02532484752055158
MAP@5: 0.05338051356846107
MAP@10: 0.06327301415424592
MAP@15: 0.06483676135094857
MAP@20: 0.0646855385268904
$$$$$$ mfc $$$$$$
HIT@1: 0.01617608061522143
HIT@5: 0.03699284009546539
HIT@10: 0.17966056748872977
HIT@15: 0.24595597984619463
HIT@20: 0.24635375232033943
NDCG@1: 0.01617608061522143
NDCG@5: 0.02245534541768104
NDCG@10: 0.06710332701945342
NDCG@15: 0.09122188052973833
NDCG@20: 0.09162275925455665
MAP@1: 0.01617608061522143
MAP@5: 0.023747569168213557
MAP@10: 0.04063809685163437
MAP@15: 0.045865588662592105
MAP@20: 0.04582674500682953


Future TODO:

1. For sparsity exp, move the user/item > u/i interactions selector inside data_loader. Then you can have all items for usparsity and all users for isparsity
2. For MF models we should fed the original edge_index without social_edges, and original node ids -> need refactor, or we can do the sparcity exp with GNN_without_social since there was not much difference