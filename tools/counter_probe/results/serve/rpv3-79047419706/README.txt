The small rocprofv3 artifacts from run 79047419706, kept in git.

The 249 MB busy_kernel_trace.csv (516,125 KERNEL_DISPATCH rows) is NOT committed — it lives at
  /home/pat/.cache/minisgl-perf/rpv3-busy-79047419706/busy_kernel_trace.csv
and every number derived from it is already in ../busy-79047419706.json. Regenerate with
  MODE=prof gpu-lease -n 2 -- bash tools/counter_probe/scorecard/serve/serve_busy_trace.sh
