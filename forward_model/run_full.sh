#!/bin/bash
# Full-sample forward model at one lambda_C, for the signed long-range likelihood.
# Same configuration as the P99 diagnostic (2 BORG samples, regressors frozen at ii=19)
# but over ALL galaxies and with repeat=300. lambda_C = 8 is run in the same configuration
# so the ratio of bounds is internally consistent.
export PATH=/usr/local/shared/python/3.9.7/bin:$PATH
export FULLSAMPLE=1
cd /users/hdesmond/V2
mkdir -p Partial FULL_out

LAM=$1
REPEAT=300
NP=${SLURM_NTASKS:-28}
TAG="lam${LAM}"

if [ -f "FULL_out/Signals_warp_FULL_${TAG}.dat" ]; then
    echo "=== $TAG already done, nothing to do ==="; exit 0
fi

echo "=== FULL sample, lambda_C = $LAM Mpc, repeat=$REPEAT, np=$NP  ($(date)) ==="
mpirun -n $NP python3 Calc_Proxies_warp_FULL.py $LAM $TAG $REPEAT

if [ -f "Signals_warp_FULL_${TAG}.dat" ]; then
    mv "Signals_warp_FULL_${TAG}.dat" FULL_out/
    echo "=== $TAG DONE ($(date)) ==="
    wc -l FULL_out/Signals_warp_FULL_${TAG}.dat
else
    echo "=== $TAG FAILED ==="; exit 1
fi
