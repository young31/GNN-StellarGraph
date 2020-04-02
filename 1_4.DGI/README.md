# Deep Graph Infomax

### Breif Review

-   현실에 더 적합한 unsupervised 방식으로 embedding작업을 수행하기 위해서 고안됨
-   GCN 및 Deep Infomax 방식을 차용하였음
-   negative sampling과 patch representation을 통해서 해당 목적을 달성함
-   2가지 learing startegy
    -   transductive learning:
        -   GCN방식을 차용하여 encoder를 구성함
    -   inductive learing
        -   GraphSAGE방식을 차용하여 encoder를 구성함

#### Description from StellarGraph

```markdown
- Deep Graph Infomax trains unsupervised GNNs to maximize the shared information between node level and graph level features.
```

### Code Review

-   코드의 구성이 ensemble방식을 짜던 코드와 비슷하게 구성되어 있음
-   embedding을 위한 코드지만 다른 목적의 코드와 비슷한 흐름을 가짐
-   generator를 구성해 줌
    -   corrupted generator:  this returns shuffled node features along with the regular node features and we train our model to discriminate between the two
    -   fullbatch 형식만 지원한다고 하니 해당 방식을 사용하는 다른 모형으로 확장 가능

```python
from stellargraph.mapper import FullBatchNodeGenerator, CorruptedGenerator

generator = FullBatchNodeGenerator(G)
corrupted_gen = CorruptedGenerator(generator) 
```

-   위와 같이 DGI 를 위한 방식으로 wrapping해주는 느낌

```python
from stellargraph.layer import GCN, DeepGraphInfomax, GAT

# gcn방식도 적용 가능
gat = GAT(layer_sizes=[256, 128], activations=[layers.PReLU(), layers.PReLU()], generator=generator)

dgi = DeepGraphInfomax(gat)
x_inp, x_out = dgi.in_out_tensors()
```

-   embedding task는 새로 모델을 만들어줘야함

```python
x_emb_in, x_emb_out = dgi.embedding_model()
emb_model = Model(inputs=x_emb_in, outputs=x_emb_out)

embedding = emb_model.predict(generator.flow(G.nodes()))
```

-   모형 예측에서 상당한 성능을 보여줌

![tsne](./asset/tsne.png)

