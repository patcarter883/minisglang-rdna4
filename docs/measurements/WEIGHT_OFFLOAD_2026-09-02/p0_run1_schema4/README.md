# P0 run 1 (schema 4) — archived, superseded by ../p0.json

Run 1 of the P0 llama.cpp baseline, 2026-09-02T14:40Z. It is a complete, valid
measurement (status `ok`, 5/5 valid reps in both legs, GPU engagement
`confirmed`) and is kept as an INDEPENDENT REPEAT of the canonical run.

It is archived rather than cited because of one gap, not one error: this
llama.cpp build (lemonade rocm-nightly `b1319`) emits no `print_info:` /
`load_tensors:` lines, so run 1's same-packing I/O denominator
(`fraction_of_this_checkpoints_expert_bytes_from_storage`) came back **null**
and the primary storage-vs-compute discriminator could not be formed. Schema 5
fixes that by reading the geometry from the GGUF tensor table instead of the
server log; the canonical run in `../p0.json` was taken with that build of the
script.

Every measured field in run 1 stands. Compare the two runs' medians as a
reproducibility check — see `../P0_LLAMACPP_BASELINE.md`.
