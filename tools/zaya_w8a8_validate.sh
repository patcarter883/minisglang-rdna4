#!/usr/bin/env bash
# W8A8 ZAYA validation: coherence (eager + graph) + decode TPOT A/B (w8a8 vs old dequant->Triton).
# Runs the whole battery inside ONE container under ONE lease. CPU build is separate (already done).
set -uo pipefail

source /app/.venv/bin/activate
pip install -q msgpack pyzmq prompt_toolkit accelerate 2>&1 | tail -1
export PYTHONPATH=/engine/python:/engine
export HF_HUB_OFFLINE=1

echo "================ 1) COHERENCE EAGER (w8a8) ================"
MINISGL_MOE_SCATTER=0 python /engine/tools/zaya_coherence_smoke.py
echo "EAGER_EXIT=$?"

echo "================ 2) COHERENCE GRAPH (w8a8, scatter off) ================"
MINISGL_MOE_SCATTER=0 python /engine/tools/zaya_graph_smoke.py
echo "GRAPH_EXIT=$?"

echo "================ 3) PERF A/B decode TPOT (M=1) ================"
echo "---- 3a) NEW w8a8 kernel path ----"
MINISGL_MOE_SCATTER=0 MINISGL_ZAYA_OLDMOE=0 python /engine/tools/zaya_tpot_ab.py
echo "AB_NEW_EXIT=$?"
echo "---- 3b) OLD dequant->Triton path ----"
MINISGL_MOE_SCATTER=0 MINISGL_ZAYA_OLDMOE=1 python /engine/tools/zaya_tpot_ab.py
echo "AB_OLD_EXIT=$?"
echo "================ DONE ================"
