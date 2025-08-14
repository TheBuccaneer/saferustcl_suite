#!/usr/bin/env bash
# ---- GPU ----------------------------------------------------------------
sudo nvidia-smi -pm 1                 # Persistenzmodus
sudo nvidia-smi -lgc 1695,1695        # feste SM-Clocks (GPU-Base/Boost anpassen)
sudo nvidia-smi -pl 350               # konstantes Power-Cap

# ---- CPU ----------------------------------------------------------------
sudo cpupower frequency-set -g performance   # fixer CPU-Takt

# ---- Benchmark ----------------------------------------------------------
taskset -c 0-31 "$@"                # auf die 32 physischen Kerne pinnen
