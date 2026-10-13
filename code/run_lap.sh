#!/bin/bash
# Usage: sbatch ... run_lap.sh <model> <group> [<group> ...]   (groups: pub test full inj)
# One unit per configuration -> lap/<model>/<name>.json; finished units are skipped on requeue.
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
cd "$(dirname "$0")"
MODEL=$1; shift
NP=${SLURM_CPUS_ON_NODE:-4}
LIST=$(for g in "$@"; do python3 lap_run.py list $g; done)
NTOT=$(echo "$LIST" | wc -l)
LAP=$(python3 -c 'import paths; print(paths.LAPDIR)')
mkdir -p $LAP/$MODEL
echo "model $MODEL groups $*: $NTOT units on $NP cores, $(hostname), $(date)"
echo "$LIST" | xargs -P $NP -I{} sh -c "[ -f $LAP/$MODEL/{}.json ] || python3 -u lap_run.py $MODEL {} >> $LAP/$MODEL/log_{}.txt 2>&1"
N=$(echo "$LIST" | while read n; do [ -f $LAP/$MODEL/$n.json ] && echo $n; done | wc -l)
echo "finished $N of $NTOT, $(date)"
[ "$N" -eq "$NTOT" ] && echo LAP_DONE || echo LAP_INCOMPLETE
