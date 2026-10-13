#!/bin/bash
# Seeded replicate of the full-sample forward model at one lambda_C.
# Usage: run_full_seed.sh <lambda_C> <seed>
# Identical configuration to the FULL_out runs (repeat=300, 2 BORG samples, regressors frozen at
# ii=19); the ONLY difference is the seed, so the spread between replicates is the forward-model
# realization noise on the bound at fixed lambda_C.
set -euo pipefail
export PATH=/usr/local/shared/python/3.9.7/bin:$PATH
export FULLSAMPLE=1
cd /users/hdesmond/V2
mkdir -p Partial FULL_out

LAM=$1
SEED=$2
REPEAT=300
NP=${SLURM_NTASKS:-28}
TAG="s${SEED}_lam${LAM}"

if [ -f "FULL_out/Signals_warp_FULL_${TAG}.dat" ]; then
    echo "=== $TAG already present, refusing to overwrite ==="; exit 0
fi

# Fail closed: clear any stale rank chunks for THIS tag before starting.
rm -f Partial/out_warp_FULL_${TAG}_*.dat

echo "=== FULL replicate, lambda_C = $LAM Mpc, seed = $SEED, repeat=$REPEAT, np=$NP  ($(date)) ==="
# NB: the seed goes in the environment, never as argv[4] -- that slot is the producer's
# "first N test points" debug switch and a seed there truncates the sample to SEED galaxies.
export REPLICATE_SEED=$SEED
mpirun -n $NP -x REPLICATE_SEED -x FULLSAMPLE python3 Calc_Proxies_warp_FULL_seed.py $LAM $TAG $REPEAT

# The producer concatenates and then removes its own rank chunks, so validate the concatenated
# output rather than counting chunks.
if [ ! -f "Signals_warp_FULL_${TAG}.dat" ]; then
    echo "=== $TAG FAILED: no concatenated output ==="; exit 1
fi
NROW=$(wc -l < "Signals_warp_FULL_${TAG}.dat")
if [ "$NROW" -ne 4345 ]; then
    echo "=== $TAG FAILED: $NROW rows, expected 4345 ==="; exit 1
fi
NCOL=$(head -1 "Signals_warp_FULL_${TAG}.dat" | wc -w)
if [ "$NCOL" -ne 60 ]; then
    echo "=== $TAG FAILED: $NCOL columns, expected 60 ==="; exit 1
fi
mv "Signals_warp_FULL_${TAG}.dat" FULL_out/
echo "=== $TAG DONE, ${NROW}x${NCOL} ($(date)) ==="
