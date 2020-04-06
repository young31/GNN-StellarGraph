# SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS

### Breif Review

-   convolution을 graph에 적용한 사례(이것보다 빨리 사용한 사례가 있는지는 확인 못함)
-   spectral decomposition 방식을 사용한 그래프 분석 방법
    -   이 방법으로 semi-supervised 하게 접근한다.
-   이 후 graphsage 논문에서 transductive하다고 지적받는다.
    -   그럼에도 획기적인 방법이긴 했던 것 같다.

#### Description from StellarGraph

```markdown
- The GCN algorithm supports representation learning and node classification for homogeneous graphs. There are versions of the graph convolutional layer that support both sparse and dense adjacency matrices.
```

### Code Review

-   NC(node-classification)/ LP(link-prediction) 방식 모두 사용 가능

#### Node Classification

-   GCN류의 방식들은 특별한 generator를 사용함
    -   이 함수를 통해서 특정 node를 포함한 subgraph 구조를 만들 수 있음

```python
from stellargraph.mapper import FullBatchNodeGenerator

generator = FullBatchNodeGenerator(G, method="gcn")

tr_flow = generator.flow(tr_target.index, tr_target) # node_id, target
val_flow = generator.flow(val_target.index, val_target)
```

-   GCN을 적용한 구조를 만들기 위해서 해당 함수를 사용함
    -   사용법은 sklearn wrapper같은 형식

```python
from stellargraph.layer import GCN

gcn = GCN(
    layer_sizes=[16, 16], activations=["relu", "relu"], generator=generator, dropout=0.5
)
```

-   keras를 사용하여 함수구현하듯이 위에서 받은 함수를 연결해서 사용할 수 있음

```python
nc_inp, nc_out = gcn.build()
nc_layer = layers.Dense(8, activation='relu')(nc_out)
nc_layer = layers.Dense(tr_target.shape[1], activation='softmax')(nc_layer)

nc_model = Model(inputs=nc_inp, outputs=nc_layer)
nc_model.compile(
    optimizer=optimizers.Adam(lr=1e-2),
    loss='categorical_crossentropy'
)
```

-   graph구조를 사용하지 않은 방법(logistic_reg)과 비교하여 acc-score에서 확연한 향상이 있음
    -   0.814 vs 0.69

-   시각화 한 결과도 양호

![gcn_tsne](./asset/GCN_tsne.png)

#### Link Prediction

-   lp에서는 edge-splitter를 사용하여 몇 연결을 삭제한 결과를 얻음

```python
edge_splitter_test = EdgeSplitter(G)

# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G, and obtain the
# reduced graph G_test with the sampled links removed:
G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
    p=0.1, method="global", keep_connected=True
)
```

-   nodegenerator 대신 linkgenerator를 사용함

```python
from stellargraph.mapper import FullBatchLinkGenerator

train_gen = FullBatchLinkGenerator(G_train, method="gcn")
train_flow = train_gen.flow(edge_ids_train, edge_labels_train)
```

-   마찬가지로 모형을 받아서 layer를 추가해서 사용
    -   이 때는 LinkEmbedding 객체를 받아서 layer로 사용
    -   반드시 reshape층을 가져야 함

```python
from stellargraph.layer import LinkEmbedding

lp_layer = LinkEmbedding(activation='relu', method='ip')(out) # ip: inner product
lp_layer = layers.Reshape((-1,))(lp_layer)

lp_model = Model(inp, lp_layer)
```