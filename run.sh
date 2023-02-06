
# models=("gcn" "gat" "appnp" "sign")
# models=("appnp" "sign")
models=("gcn")

for model in ${models[@]};
do
    echo run $model
    python $model/train_sparse_compile.py --dataset cora > log/${model}_cora.log 2>&1
    python $model/train_sparse_compile.py --dataset ogbn-products > log/${model}_products.log 2>&1
    python $model/train_sparse_compile.py --dataset ogbn-arxiv > log/${model}_arxiv.log 2>&1
    
done