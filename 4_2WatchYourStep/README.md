# Watch Your Step: Learning Node Embeddings via Graph Attention

### Breif Review

-   attention mechanism을 임베딩에 응용
-   DeepWalk, GloVe등의 방식에 영감을 받음
-   이전 방식들이 hyper-parameter 세팅에 따라 성능이 크게 달라지는 것을 개선
    -   length of random-walk나 embedding 파라미터 등에 의해 영향받는 것 개선
    -   위 parameter들을 trainable parameter로 대체하여 해결

-   사용한 방법으로는 expectation사용, attention parameter를 loss함수에 사용하는 등을 응용

#### Description from StellarGraph

```markdown
- The Watch Your Step algorithm computes node embeddings by using adjacency powers to simulate expected random walks.
```

### Code Review

-   Adjacency 방식이 구현되어 있음

```python
from stellargraph.mapper import AdjacencyPowerGenerator

# it can be run fast via SVD
generator = AdjacencyPowerGenerator(G)

```

-   논문에서 언급한 것 처럼 attention에 regularizer 적용

```python
from stellargraph.layer import WatchYourStep

wys = WatchYourStep(generator=generator, embedding_dimension=128, attention_regularizer=regularizers.l2(0.5)) # use l2 norm regularization
nc_inp, nc_out = wys.build()
nc_model = Model(inputs=nc_inp, outputs=nc_out)
```

~~생각보다 결과가 GAT만큼 좋게 나오지 않아서 검토할 필요가 있음~~
