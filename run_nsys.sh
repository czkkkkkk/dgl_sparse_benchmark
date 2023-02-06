
export PROF_TARGET_PASS="NVPROF"
export PROF_TARGET_SESSION="10"
export PROF_EARLY_EXIT=true
export PYTHONPATH="/workspace2/python_profiler/:$PYTHONPATH"
nsys profile -o log/nsys_gcn_cora --stats true --force-overwrite true -c cudaProfilerApi --kill none python gcn/train_sparse_compile.py
