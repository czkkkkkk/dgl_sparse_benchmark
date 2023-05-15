
export PROF_TARGET_PASS="NVPROF"
export PROF_TARGET_SESSION="2"
export PROF_TARGET_RANGE="2"
export PROF_EARLY_EXIT=true
export PYTHONPATH="/workspace2/python_profiler/:$PYTHONPATH"

# # models=("gcn" "gat" "appnp")
# models=("gcn")

# dataset=cora
# for model in ${models[@]};
# do
#     echo run $model
#     name=nsys_${model}_${dataset}_$(date +%H_%M_%S)
#     # original model
#     nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx -o log/nsys/${name} --stats true --force-overwrite true -c cudaProfilerApi --kill none python $model/train_sparse_compile.py --dataset ${dataset} --profile > log/nsys/${name}.log 2>&1 
#     # compiled model
#     nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx -o log/nsys/${name}_compile --stats true --force-overwrite true -c cudaProfilerApi --kill none python $model/train_sparse_compile.py --dataset ${dataset} --compile --profile > log/nsys/${name}_compile.log 2>&1 
# done


##########################cuda-graph#####################################
models=("gat")

for model in ${models[@]};
do
    echo run $model
    name=nsys_${model}_$(date +%H_%M_%S)
    logdir=log/cuda_graph/${name}
    nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx -o $logdir --stats true --force-overwrite true -c cudaProfilerApi --kill none python cuda_graph/gat/train.py > $logdir.log 2>&1 
done