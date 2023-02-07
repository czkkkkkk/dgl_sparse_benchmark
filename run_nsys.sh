
export PROF_TARGET_PASS="NVPROF"
export PROF_TARGET_SESSION="10"
export PROF_EARLY_EXIT=true
export PYTHONPATH="/workspace2/python_profiler/:$PYTHONPATH"

models=("appnp")

for model in ${models[@]};
do
    echo run $model
    nsys profile -o log/nsys_${model}_cora_compile --stats true --force-overwrite true -c cudaProfilerApi --kill none python $model/train_sparse_compile.py --dataset cora
    # nsys profile -o log/nsys_${model}_arxiv --stats true --force-overwrite true -c cudaProfilerApi --kill none python $model/train_sparse_compile.py --dataset ogbn-arxiv
done
