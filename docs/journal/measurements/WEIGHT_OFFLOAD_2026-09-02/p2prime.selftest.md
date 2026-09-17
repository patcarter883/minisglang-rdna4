# P2prime — selftest **PASS**

58/58 checks passed · 2026-09-02T22:51:11.817+00:00

> no GPU was touched: no device was set, no HIP allocation was made, and the shared library was only inspected for symbols

| check | ok | detail |
|---|---|---|
| `defaults_parse` | ✅ | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] |
| `miss_counts_default_is_full_range` | ✅ | the exact binomial hit-rate expectation requires every m from 0..top_k |
| `miss_counts_override` | ✅ |  |
| `rejects[block_m not a multiple of 8]` | ✅ | --block-m must be a multiple of 8 and <= 64 (the core's contract) |
| `rejects[min(block_m, tokens) > 16]` | ✅ | min(block_m, tokens) > 16: this decode probe instantiates the MMAX {8, 16} ladder only. Raising it is a one-line change in p2prime_kernels.hip, but silently cla |
| `rejects[thrash below the Infinity Cache]` | ✅ | --thrash-mb below 64 cannot evict this card's Infinity Cache; a smaller value would let a host-resident expert be served from on-die cache and report a miss as  |
| `rejects[hit rate out of [0,1]]` | ✅ | --hit-rates must lie in [0, 1] |
| `layout_component_sizes` | ✅ | {"w13": 1572864, "s13": 49152, "z13": 12288, "w2": 786432, "s2": 24576, "z2": 6144} |
| `layout_offsets_aligned` | ✅ | {"w13": 0, "s13": 1572864, "z13": 1622016, "w2": 1634304, "s2": 2420736, "z2": 2445312} |
| `layout_offsets_disjoint_and_ordered` | ✅ |  |
| `granule_stride_covers_payload` | ✅ | stride=2453504 payload=2451456 |
| `layout_rejects_bad_inter` | ✅ |  |
| `distinct_route_is_distinct` | ✅ |  |
| `route_block_count` | ✅ | n_blocks=10 |
| `route_sti_length` | ✅ |  |
| `route_padding_marker_is_ge_num_valid` | ✅ | the core treats offs >= num_valid_tokens as padding; a smaller marker would make a pad row compute against a real token |
| `route_covers_every_routed_slot` | ✅ |  |
| `route_ntp_matches_blocks` | ✅ |  |
| `route_rejects_short_assignment` | ✅ |  |
| `collision_route_groups_by_expert` | ✅ |  |
| `mask_achieves[0]` | ✅ | host_experts=0 achieved=0 |
| `mask_achieves[1]` | ✅ | host_experts=1 achieved=1 |
| `mask_achieves[2]` | ✅ | host_experts=2 achieved=2 |
| `mask_achieves[3]` | ✅ | host_experts=3 achieved=3 |
| `mask_achieves[4]` | ✅ | host_experts=4 achieved=4 |
| `mask_achieves[5]` | ✅ | host_experts=5 achieved=5 |
| `mask_achieves[6]` | ✅ | host_experts=6 achieved=6 |
| `mask_achieves[7]` | ✅ | host_experts=7 achieved=7 |
| `mask_achieves[8]` | ✅ | host_experts=8 achieved=8 |
| `mask_achieves[9]` | ✅ | host_experts=9 achieved=9 |
| `mask_achieves[10]` | ✅ | host_experts=10 achieved=10 |
| `mask_rejects_overflow` | ✅ |  |
| `assembly_picks_exactly_the_host_blocks` | ✅ | 3 of 10 blocks came from the host reference |
| `assembly_differs_from_both_refs` | ✅ |  |
| `stats_median_is_robust` | ✅ | {"n": 5, "min": 1.0, "max": 100.0, "median": 3.0, "mean": 22.0, "p10": 1.4, "p25": 2.0, "p75": 4.0, "p90": 61.60000000000001, "iqr": 2.0, "rel_spread": 20.06666666666667, "stdev": 39.01281840626232} |
| `stats_empty_is_safe` | ✅ |  |
| `fit_linear_classifies_LINEAR` | ✅ | cliff_index=0.09999999999999999 |
| `fit_linear_recovers_slope` | ✅ |  |
| `fit_linear_W_is_about_one` | ✅ | 1.000 |
| `fit_cliff_classifies_CLIFFED` | ✅ | cliff_index=1.0 |
| `fit_cliff_W_is_infinite` | ✅ |  |
| `fit_partial_classifies_PARTIAL` | ✅ | cliff_index=0.44999999999999996 |
| `fit_flat_is_INDETERMINATE_not_a_shape` | ✅ | a curve that does not separate must refuse to name a shape |
| `fit_needs_two_points` | ✅ |  |
| `binomial_is_exact_over_full_sweep` | ✅ |  |
| `linear_curve_ties_per_expert_to_layer_granular` | ✅ | on a LINEAR curve, per-expert and layer-granular placement are worth exactly the same at equal byte budget -- the gain comes from the curve's CONVEXITY, so this is the correct null and it is asserted  |
| `cliff_curve_makes_per_expert_LOSE` | ✅ | ["0.775", "0.550", "0.342", "0.277", "0.314", "0.586"] |
| `hit_rate_analysis_handles_partial_sweep` | ✅ |  |
| `decide_linear` | ✅ |  |
| `decide_cliff` | ✅ |  |
| `decide_indeterminate` | ✅ |  |
| `layer_total_sums_the_arms` | ✅ | {"6": 11.4, "3": 8.7, "7": 12.3, "0": 1.5, "10": 15.0, "8": 13.2, "4": 9.6, "9": 14.1, "1": 6.9, "5": 10.5, "2": 7.8} |
| `layer_total_names_a_shape` | ✅ |  |
| `layer_total_handles_no_overlap` | ✅ |  |
| `json_shape_valid` | ✅ |  |
| `md_renders` | ✅ | 3338 chars |
| `validate_rejects_incomplete` | ✅ |  |
| `so_exports_every_required_symbol` | ✅ | /home/pat/code/minisgl-rdna4-offload/tools/offload/_build/p2prime_kernels.so |

