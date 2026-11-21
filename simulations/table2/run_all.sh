#!/bin/sh
# Reproduce the Donoho-Johnstone illustration (M = 1000 per signal).
# Runtime: about 30 min on 2 cores. Requires Python >= 3.10, numpy >= 2.0, scipy, matplotlib.
cd "$(dirname "$0")"; mkdir -p res out
set -e
for pair in "Blocks 20261104" "Bumps 20261105" "Doppler 20261106" "HeaviSine 20261107"; do
  set -- $pair
  python3 -W ignore run_mc.py        $1 1000 5 sd $2 res/main_$1.npz   # 16 rules, same noise
  python3 -W ignore run_caca_mmle.py $1 1000 $2 res/mmle_$1.npz        # Ca/Ca, MMLE tau_j
  python3 -W ignore run_hs_wide.py   $1 1000 $2 res/hswide_$1.npz      # horseshoe, wide grid
done
python3 make_outputs.py main   # summary, LaTeX table rows (out/)
python3 fig_amse.py            # figure (out/)
