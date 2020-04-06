# Graph NN with StellarGraph

## Table of Contents

- [Intro](#intro)
- [Embedding 모형](#embedding-모형)
  
     - [Node2Vec](#node2vec)
     - [Metapath2Vec](#metapath2vec)
     - [Attri2Vec](#attri2vec)
     - [DGI(deep graph infomax)](#dgideep-graph-infomax)
- [GCN기반 모형](#gcn기반-모형)
     - [GCN](#gcn)
     - [RGCN](#rgcn)
     - [SGC](#sgc)
     - [Cluster-GCN](#cluster-gcn)
- [GraphSAGE 기반 모형](#GraphSAGE-기반-모형)  
     - [GraphSAGE](#GraphSAGE)  
     - [HinSAGE](#hinsage)
- [GAT기반 모형](#gat기반-모형)
  - [Watch Your Step](#watch-your-step)
- [Comparison](#comparison)
  - [Supervised](#supervised)  
       - [Dataset](#dataset)  
       - [Node Classification](#node-classification)  
       - [Link Prediction](#link-prediction)  
      
  - [Unsupervised](#unsupervised)
       - [Node Embedding](#node-embedding)
- [Reference](#Reference)

______

## Intro

-   GNN에 대해서 처음 접하면서 관련된 정보를 찾던 중 stellargraph라는 패키지를 알게되었습니다.
-   기본적인 구현이 생각보다 폭넓게 되어있다고 생각되어 관련논문을 읽고 어떻게 구현되어있는지를 알아보고자 하였습니다.
-   처음 접하는 만큼 논문을 읽으면서 완전히 깊이 이해하기 보다 큰그림을 이해하려고 하였습니다.
-   평가 및 비교는 실험데이터를 가지고 하였으며 각 모형마다 최적화는 하지 않았습니다.
    -   이로 인해 논문에서 주장한 결과와 상이하게 나올 수 있습니다.
-   Cora 파일에 설명이 있고 나머지는 비교를 위한 파일들로 구성되어 있습니다.

## stellargraph 사용법

-   stellargraph 객체를 만드는 방법
-   python API만 다루며 networkx관련하여는 추후 필요시 추가

### python API

1.  nodes, edges, features에 관한 dataframe을 준비

    -   node와 feature는 한 dataframe에 표현 가능
        -   **이 때 node는 feature의 인덱스칼럼이 됨**
    -   여러 타입의 node/edge를 함께 사용할 수 있음(개별 dataframe으로 준비)

    -   이 때 edge의 column은 source와 target으로 분류(수정 가능)

```python
# node
all_features = pd.read_table('../datasets/twitter/users_hate_all.content', 
                             header=None, index_col=0) # index칼럼이 node명이 됨

# edge
edges = pd.read_table('../datasets/twitter/users.edges', header=None, sep=' ')
edges.columns = ['source', 'target'] # should be follow this column name
```

2.  stellargraph 객체 생성
    -   node와 edge정보만 있으면 생성할 수 있음

```python
import stellargraph as sg
G = sg.StellarGraph(edges=edges, nodes=all_features)
```

```python
# 여러 노드 정보 활용가능
StellarGraph(nodes={"foo": foo_nodes, "bar": bar_nodes}, edges)
```

```python
# 여러 edge정보와도 결합가능
StellarGraph(
            nodes={"foo": foo_nodes, "bar": bar_nodes},
            edges={"h": horizontal_edges, "v": vertical_edges, "d": diagonal_edges})
```

3.  추가적으로 생성시 directed 여부 등을 설정할 수 있음
    -   directed그래프 지원 모델
        -   GraphSAGE
        -   GAT

## Embedding 모형

-   임베딩 모형은 그래프 구조에서 노드와 노드사이의 관계를 어떻게 latent space로 매핑할지에 대해 다룹니다.
-   임베딩한 결과를 바탕으로 classification문제 등으로 확장할 수 있습니다.

#### [Node2Vec](https://github.com/young31/GNN-StellarGraph/tree/master/1_1. Node2Vec)

-   가장 처음 접한 GNN기반 임베딩모형입니다.
-   이웃 노드를 찾는 과정에서 DFS/BFS를 일반화하는 식을 만들어 탐색합니다.

#### [Metapath2Vec](https://github.com/young31/GNN-StellarGraph/tree/master/1_2. Metapath2Vec)

-   heterogeneous networks를 다루기 위해 고안된 방법입니다.
-   서로 다른 타입(class)의 노드들이 있는 경우 활용합니다.

#### [Attri2Vec](https://github.com/young31/GNN-StellarGraph/tree/master/1_3. Attri2Vec)

-   그래프의 구조뿐만 아니라 그 특성(attribute)까지 파악하여 임베딩하는 방법입니다.
-   이전의 결과들을 결합하여 성능을 향상시켰습니다.

#### [DGI(deep graph infomax)](https://github.com/young31/GNN-StellarGraph/tree/master/1_4.DGI)

-   가장  최근에 나온 모형인 만큼 성능면에서 뛰어난 결과를 보여줬습니다.
-   구현 단계에서 bagging이나 boosting등에 활용과 비슷하여 새로웠습니다.

## GCN기반 모형

-   보다 분석의 초점을 맞춘 모형들입니다.
-   GCN의 개념이 처음 나오고 다양하게 변형되고 있는것 같습니다.
-   주요 목적으로는 node classification(NC), link prediction(lp)가 있습니다.

#### [GCN](https://github.com/young31/GNN-StellarGraph/tree/master/2_1. GCN)

-   가장 기본적이고 처음 접한 모형입니다.
-   spectral decomposition 방식으로 접근합니다.
-   초기 버전임에도 상당히 괜찮은 결과를 보여줍니다.
-   이 후 논문들은 이 모형을 기반으로 보완/발전해 나가는 모습을 보여줍니다.

#### RGCN

-   관계형 데이터의 그래프구조를 표현하기 위한 모형입니다.
-   그래프에서의 관계형 구조를 제대로 이해하지 못해 제대로 구현하지 못하였습니다.

#### [SGC](https://github.com/young31/GNN-StellarGraph/tree/master/2_3. SGC)

-   GCN의 simplify 버전입니다.
-   GCN의 힘이 non-linear형식이 아니라 local-averaging으로 부터 온다고 가정하여 식을 간단히합니다.
-   계산량은 줄이면서 성능은 크게 떨어지지 않는 효과를 기대하며 사용합니다.

#### [Cluster-GCN](https://github.com/young31/GNN-StellarGraph/tree/master/2_4. Cluster-GCN)

-   GCN방식을 clustering을 적용하여 이웃탐색 과정 없이 훈련하도록 디자합니다.
-   이로 인해 시간적/공간적으로 효율성을 높인다고 합니다.
-   그래프 clustering방식으로 하면 더 성능 향상을 볼 수 있는데 관련한 metis에 대해서 잘 알지 못하여 실험해보지 못하였습니다.

## GraphSAGE 기반 모형

#### [GraphSAGE](https://github.com/young31/GNN-StellarGraph/tree/master/3_1. GraphSAGE)

-   GCN에서 영감을 받아 발전한 형태입니다.
-   GCN의 transductive한 방식을 inductive한 방식으로 발전시켰습니다.
-   방향그래프에도 적용할 수 있습니다.

#### HinSAGE

-   패키지에서 구현한 변형 모형입니다.
-   heterogeneous network에도 적용할 수 있도록 개량한 버전입니다.

## GAT기반 모형

#### [GAT](https://github.com/young31/GNN-StellarGraph/tree/master/4_1. GAT)

-   attention 기술을 적용시켜 발전시킨 모형입니다.
-   행렬에 대한 decomposition을 하지 않기 때문에 상당히 빠르게 작동합니다. 
-   마찬가지로 방향그래프에 사용할 수 있습니다.
-   활용도가 가장 높다고 생각합니다.

#### [Watch Your Step](https://github.com/young31/GNN-StellarGraph/tree/master/4_2. WatchYourStep)

-   어텐션 매커니즘을 적용한 방식입니다.
-   임베딩을 위한 모형임에도 위의 이유로 GAT기반 모형에 분류하였습니다.
-   파라미터 선택에 민감한 것을 극복하고자 디자인되었습니다.

## Comparison

### Supervised

-   Almost every method can be applied
-   Parameter Setting:
    -   layers = [32, 32]
    -   activation = 'elu'
    -   dropout = 0.5
-   model specific parametrs are preserved

#### Dataset

**관계형 자료나 복수의 type을 갖는 graph는 포함시키지 않음**

-   Cora
    -   nodes: 2,708
    -   edges: 5,278
-   Twitter(local)
    -   nodes: 100,386
    -   edges: 2,194,979
-   Citeseer
    -   nodes: 3,312
    -   edges: 4,715
-   Pubmed
    -   nodes: 19,717
    -   edges: 44,338

#### Node Classification

-   GCN의 성능이 상대적으로 높아서 놀라움
-   실험 세팅이 편향되지 않았나 추측함
-   속도면에서는 GAT모형이 상당히 빠르게 나타남

|            | Data     | F1_micro  | F1_macro  |
| ---------- | -------- | --------- | --------- |
| GCN        | Cora     | 0.844     | **0.836** |
|            | Twitter  | **0.929** | **0.790** |
|            | CiteSeer | **0.739** | 0.65      |
|            | PubMed   | **0.875** | **0.87**  |
| GraphSAGE  | Cora     | **0.849** | 0.83      |
|            | Twitter  | 0.917     | 0.788     |
|            | CiteSeer | 0.731     | 0.632     |
|            | PubMed   | 0.869     | 0.867     |
| GAT        | Cora     | 0.838     | 0.827     |
|            | Twitter  | 0.916     | 0.716     |
|            | CiteSeer | 0.729     | 0.666     |
|            | PubMed   | 0.857     | 0.852     |
| ClusterGCN | Cora     | 0.837     | 0.825     |
|            | Twitter  | OOM       | OOM       |
|            | CiteSeer | **0.739** | **0.684** |
|            | PubMed   | 0.864     | 0.86      |

#### Link Prediction

|           | Data | F1_micro  | F1_macro  |
| --------- | ---- | --------- | --------- |
| GCN       | Cora | 0.711     | 0.697     |
| GraphSAGE | Cora | 0.691     | 0.681     |
| GAT       | Cora | **0.731** | **0.718** |

### Unsupervised

-   A few methods can be applied:
    -   having own embedding strategy

#### Node Embedding

-   Every result is that of logistic regression after embedding

|               | Data | Fl_micro  | F1_macro  | NC_iteself |
| ------------- | ---- | --------- | --------- | ---------- |
| Node2Vec      | Cora | 0.818     | 0.808     | 0.236      |
| - Word2Vec    |      |           |           |            |
| WatchYourStep | Cora | 0.7       | 0.688     | **0.754**  |
| DGI           | Cora | **0.845** | **0.831** | 0.300      |

## Reference

[[1]](https://arxiv.org/abs/1609.02907) SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS(GCN)

[[2]](https://arxiv.org/abs/1706.02216) Inductive Representation Learning on Large Graphs(GraphSAGE)

[[3]](https://arxiv.org/abs/1607.00653) node2vec: Scalable Feature Learning for Networks

[[4]](https://arxiv.org/abs/1710.10903) GRAPH ATTENTION NETWORKS

[[5]](https://arxiv.org/abs/1710.09599) Watch Your Step Learning Node Embeddings via Graph Attention

[[6]](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf) metapath2vec: Scalable Representation Learning for Heterogeneous Networks

[[7]](https://arxiv.org/abs/1703.06103) Modeling Relational Data with Graph Convolutional Networks

[[8]](https://arxiv.org/abs/1902.07153) Simplifying Graph Convolutional Networks

[[9]](https://arxiv.org/abs/1905.07953) Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks

[[10]](https://arxiv.org/abs/1901.04095) Attributed Network Embedding via Subspace Discovery

[[11]](https://arxiv.org/abs/1809.10341)  Deep Graph Infomax
