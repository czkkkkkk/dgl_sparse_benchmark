models=("gat")
dataset=cora
for model in ${models[@]};
do
    echo run $model
    name=${model}_${dataset}_$(date +%H_%M_%S)
    # original model
    python $model/train_sparse_compile.py --dataset ${dataset} > log/epoch_time/${name}.log 2>&1 
    # compiled model
    python $model/train_sparse_compile.py --dataset ${dataset} --compile True > log/epoch_time/${name}_compile.log 2>&1 
done