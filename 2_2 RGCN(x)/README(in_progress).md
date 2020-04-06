# Modeling Relational Data with Graph Convolutional Networks

### Breif Review

-   관계형 데이터를 처리하기 위해 고안된 방법
-   basis-decomposition과 block-diagnal-decomposition을 사용하여 효율성을 높임
    -   basis: weight sharing
    -   block: sparsity constraints
-   크게 보아서는 AE구조
    -   NC문제에서는 임베딩(encoder)만 사용ㅎ
    -   LP문제에서는 RGCN(encoder) + DistMult(decoder)

#### Description from StellarGraph

```markdown
- The RGCN algorithm performs semi-supervised learning for node representation and node classification on knowledge graphs. RGCN extends GCN to directed graphs with multiple edge types and works with both sparse and dense adjacency matrices.
```

### Code Review

