# DGL sparse benchmark

**Github repository: [https://github.com/czkkkkkk/dgl_sparse_benchmark](https://github.com/czkkkkkk/dgl_sparse_benchmark)**

## Setup

Datasets

|          | # Nodes   | # Edges     | Avg. degree | # Features |
| -------- | --------- | ----------- | ----------- | ---------- |
| Cora     | 2,708     | 10,556      | 3.90        | 1433       |
| Arxiv    | 169,343   | 1,335,586   | 7.88        | 128        |
| Products | 2,449,029 | 126,167,309 | 51.5        | 100        |

Models

|       | Parameters                      |
| ----- | ------------------------------- |
| GCN   | Hidden: 16; Layers: 2.          |
| GAT   | Hidden: 8; Heads: 8; Layers: 2. |
| APPNP | Hidden: 64; num_hops: 10.       |

## torchScript Runnable Test

| Model | Cora | ogbn-arxiv | ogbn-products |
| :---: | :--: | ------------- | ---------- |
|  GCN  |  Y  | Y         | Y          |
|  GAT  |  Y  | Y         | OOM        |
| appnp |  Y  | Y         | OOM        |
| sign  |  Y  | Y         | OOM        |

## Epoch time and memory consumption

Each cell is in form of "Epoch time(ms)/Epoch time with torchScript(ms)".

| Model | Cora | ogbn-arxiv | ogbn-products |
| :---: | :--: | ------------- | ---------- |
|  GCN  |  1.63/1.68  | 18.04/15.74      | 1007/1015         |
|  GAT  |  4.41/4.26  | 69.8/70.3        | OOM        |
| appnp |  4.22/3.18  | 91.2/86.0         | OOM        |
| sign  |  9.21/8.89  | 67.8/66.8         | OOM        |

