# DGL sparse benchmark

**Github repository: [https://github.com/czkkkkkk/dgl_sparse_benchmark](https://github.com/czkkkkkk/dgl_sparse_benchmark)**

## Setup

Datasets

|          |  # Nodes  |   # Edges   | Avg. degree | # Features |
| -------- | :-------: | :---------: | :---------: | :--------: |
| Cora     |   2,708   |   10,556   |    3.90    |    1433    |
| Arxiv    |  169,343  |  1,335,586  |    7.88    |    128    |
| Products | 2,449,029 | 126,167,309 |    51.5    |    100    |

Models

|       | Parameters                      |
| ----- | ------------------------------- |
| GCN   | Hidden: 16; Layers: 2.          |
| GAT   | Hidden: 8; Heads: 8; Layers: 2. |
| APPNP | Hidden: 64; num_hops: 10.       |

## torchScript Runnable Test

| Model | Cora | ogbn-arxiv | ogbn-products |
| :---: | :--: | :--------: | :-----------: |
|  GCN  |  Y  |     Y     |       Y       |
|  GAT  |  Y  |     Y     |      OOM      |
| appnp |  Y  |     Y     |      OOM      |
| sign |  Y  |     Y     |      OOM      |

## Epoch time and memory consumption

Each cell is in form of "Epoch time(ms)/Epoch time with torchScript(ms)".

| Model |   Cora   | ogbn-arxiv | ogbn-products |
| :---: | :-------: | :---------: | :-----------: |
|  GCN  | 1.63/1.68 | 18.04/15.74 |   1007/1015   |
|  GAT  | 4.41/4.26 |  69.8/70.3  |      OOM      |
| appnp | 4.22/3.18 |  91.2/86.0  |      OOM      |
| sign | 9.21/8.89 |  67.8/66.8  |      OOM      |

### Appnp fused kernel

torchScript automatically fuses three element-wise kernel in appnp model.

```python
Z = (1 - self.alpha) * dglsp.spmm(A_drop, Z) + self.alpha * Z_0
```

Before fuse

|Name |   Start   |  Duration  |  GPU	Context |
| :-------: | :--------: | :--------: | :------------: |
|    void dgl::aten::cuda::SpMMCooKernel    | 0.0111332s | 12.320 μs | GPU 0	Stream 7 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor`<float>`>, at::detail::Array<char *, (int)2>> | 0.0111735s | 3.968 μs | GPU 0	Stream 7 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor`<float>`>, at::detail::Array<char *, (int)2>> | 0.0111959s | 3.967 μs | GPU 0	Stream 7 |
|void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add`<float>`, at::detail::Array<char *, (int)3>>| 0.0112133s | 4.160 μs | GPU 0	Stream 7 |
|void at::native::`<unnam>`::edfused_dropout_kernel_vec<float, float, unsigned int, (int)1, (int)4, bool> | 0.0112699s | 6.273 μs | GPU 0	Stream 7 |

After fuse

|                           Name                           |   Start   |  Duration  |  GPU	Context  |
| :------------------------------------------------------: | :--------: | :--------: | :------------: |
|           void dgl::aten::cuda::SpMMCooKernel           | 0.0109067s | 12.736 μs | GPU 0	Stream 7 |
|                   CudaCodeGen::kernel1                   | 0.0109722s | 4.160 μs | GPU 0	Stream 7 |
| void at::native::`<unnamed>`::fused_dropout_kernel_vec | 0.0110076s | 6.304 μs | GPU 0	Stream 7 |

## Model IR

GCN

``` python
def forward(self,
    A_norm: __torch__.dgl.sparse.sparse_matrix.SparseMatrix,
    X: Tensor) -> Tensor:
  W1 = self.W1
  X0 = __torch__.dgl.sparse.matmul.spmm(A_norm, (W1).forward(X, ), )
  X1 = __torch__.torch.nn.functional.relu(X0, False, )
  W2 = self.W2
  X2 = __torch__.dgl.sparse.matmul.spmm(A_norm, (W2).forward(X1, ), )
  return X2
```

GAT

``` python
def forward(self,
    A_hat: __torch__.dgl.sparse.sparse_matrix.SparseMatrix,
    X: Tensor) -> Tensor:
  in_conv = self.in_conv
  _0 = __torch__.torch.nn.functional.elu((in_conv).forward(A_hat, X, ), 1., False, )
  Z = torch.flatten(_0, 1)
  out_conv = self.out_conv
  Z0 = torch.mean((out_conv).forward(A_hat, Z, ), [-1])
  return Z0
```

appnp

``` python
def forward(self,
    A_hat: __torch__.dgl.sparse.sparse_matrix.SparseMatrix,
    X: Tensor) -> Tensor:
  _0 = __torch__.dgl.sparse.sparse_matrix.val_like
  f_theta = self.f_theta
  _1 = (f_theta).forward(X, )
  num_hops = self.num_hops
  Z = _1
  for _2 in range(num_hops):
    A_dropout = self.A_dropout
    _3 = (A_dropout).forward((A_hat).__val_getter(), )
    A_drop = _0(A_hat, _3, )
    alpha = self.alpha
    _4 = torch.sub(1, alpha)
    _5 = __torch__.dgl.sparse.matmul.spmm(A_drop, Z, )
    _6 = torch.mul(_5, _4)
    alpha0 = self.alpha
    Z0 = torch.add(_6, torch.mul(_1, alpha0))
    Z = Z0
  return Z
```

sign
``` python
def forward(self,
    X_sign: Tensor) -> Tensor:
  theta = self.theta
  results = (theta).forward(X_sign, )
  Z = __torch__.torch.nn.functional.relu(results, False, )
  omega = self.omega
  return (omega).forward(Z, )
```