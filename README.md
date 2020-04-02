# Graph NN with StellarGraph

## Comparison

### \*\*Results will be different depeding on paramet settings\*\*

### Supervised

-   Almost every method can be applied
-   Parameter Setting:
    -   layers = [32, 32]
    -   activation = 'elu'
    -   dropout = 0.5
-   model specific parametrs are preserved

#### Dataset

-   Cora: Muliti-classification
-   Twitter: Bincary-classification

#### Node Classification

|            | Data     | F1_micro | F1_macro |
| ---------- | -------- | -------- | -------- |
| GCN        | Cora     | 0.844    | 0.836    |
|            | Twitter  | 0.929    | 0.790    |
|            | CiteSeer | 0.739    | 0.65     |
|            | PubMed   | 0.875    | 0.87     |
| GraphSAGE  | Cora     | 0.849    | 0.83     |
|            | Twitter  | 0.917    | 0.788    |
|            | CiteSeer | 0.731    | 0.632    |
|            | PubMed   | 0.869    | 0.867    |
| GAT        | Cora     | 0.838    | 0.827    |
|            | Twitter  | 0.916    | 0.716    |
|            | CiteSeer | 0.729    | 0.666    |
|            | PubMed   | 0.857    | 0.852    |
| ClusterGCN | Cora     | 0.837    | 0.825    |
|            | Twitter  | OOM      | OOM      |
|            | CiteSeer | 0.739    | 0.684    |
|            | PubMed   | 0.864    | 0.86     |

#### Link Prediction

|           | Data | F1_micro | F1_macro |
| --------- | ---- | -------- | -------- |
| GCN       | Cora | 0.711    | 0.697    |
| GraphSAGE | Cora | 0.691    | 0.681    |
| GAT       | Cora | 0.731    | 0.718    |

### Unsupervised

-   A few method can be applied:
    -   having own embedding startegy

#### Node Embedding

-   Every result is that of logistic regression after embedding

|                        | Data | Fl_micro | F1_macro | NC    |
| ---------------------- | ---- | -------- | -------- | ----- |
| Node2Vec(Word2Vec)     | Cora | 0.818    | 0.808    | 0.236 |
| - Word2Vec             |      |          |          |       |
| WatchYourStep          | Cora | 0.7      | 0.688    | 0.754 |
| Metapath2Vec           | Cora |          |          |       |
| - Word2Vec / meta-path |      |          |          |       |

