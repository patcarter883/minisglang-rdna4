#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "########## STEP 1: lossless gate + new propose timing ##########"
bash tools/run_mla_spec_graph.sh
echo "########## STEP 2: tok/s sweep EAGER (GRAPH_SPEC=0) ##########"
MODEL=QuantTrio/GLM-4.7-Flash-AWQ GRAPH_SPEC=0 TAG=onfix_eager CONFIGS="none:0:0 eagle3:6:1" bash tools/run_spec_len_sweep.sh
echo "########## STEP 3: tok/s sweep GRAPH (GRAPH_SPEC=8) ##########"
MODEL=QuantTrio/GLM-4.7-Flash-AWQ GRAPH_SPEC=8 TAG=onfix_graph CONFIGS="eagle3:6:1" bash tools/run_spec_len_sweep.sh
echo "########## VALIDATION DONE ##########"
