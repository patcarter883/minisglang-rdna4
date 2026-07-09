# minisgl serving metrics (Prometheus + Grafana)

The minisgl serve exposes Prometheus metrics at **`http://<host>:1919/metrics`** (the serve's
`EXPOSE` port). Metrics are emitted in the Prometheus text exposition format by hand — the lean
image (`minisgl-rdna4:lean`) does **not** ship `prometheus_client`, so no dependency is added.
Implementation: `python/minisgl/server/metrics.py` (frontend counters/histograms) +
`SchedulerMetrics` in `python/minisgl/scheduler/scheduler.py` (spec/occupancy), threaded to the
frontend by piggybacking a `StatsMsg` on the existing scheduler → detokenizer → frontend ZMQ path
(no new socket).

## Metric list

Throughput (counters): `minisgl_generation_tokens_total`, `minisgl_prompt_tokens_total`.
Requests (counters/gauge): `minisgl_requests_total`, `minisgl_requests_success_total`,
`minisgl_requests_aborted_total`, `minisgl_requests_inflight`.
Latency (histograms): `minisgl_ttft_seconds`, `minisgl_tpot_seconds`,
`minisgl_request_latency_seconds`.
Spec-decode (counters + derived gauges): `minisgl_spec_draft_tokens_total`,
`minisgl_spec_accepted_tokens_total`, `minisgl_spec_emitted_tokens_total`,
`minisgl_spec_steps_total`, `minisgl_spec_mean_accept_len` (= 1 + accepted/steps),
`minisgl_spec_accept_rate` (= accepted/drafted).
Occupancy (gauges): `minisgl_running_requests`, `minisgl_waiting_requests`,
`minisgl_kv_pool_used_tokens`, `minisgl_kv_pool_total_tokens`, `minisgl_gdn_state_used_slots`,
`minisgl_gdn_state_total_slots`.

Every series carries a `model_name="<model-path>"` label (mirrors vLLM; drives the Grafana
`$model` selector). Spec/occupancy series are summed across DP replicas in the frontend.

Metrics are ON by default. `MINISGL_METRICS=0` disables the scheduler snapshot push;
`MINISGL_METRICS_INTERVAL` (seconds, default 0.5) throttles it.

## Prometheus wiring (already in place)

`vllm-gfx1201-prometheus` scrapes via the config at
`/home/pat/code/vllm-gfx1201/monitoring/prometheus.yml`, which already contains a `minisgl` job:

```yaml
  - job_name: minisgl
    metrics_path: /metrics
    static_configs:
      - targets: ['host.docker.internal:1919']
```

A verbatim copy is vendored here as `prometheus-scrape-reference.yml`. The running Prometheus
already has the target loaded (`curl -s localhost:9090/api/v1/targets` shows job `minisgl`); it
reads `down` until a serve built with this code exposes `/metrics`. If the config is edited,
reload **non-destructively** (the container runs with `--web.enable-lifecycle`):

```
curl -X POST localhost:9090/-/reload
```

Do **not** restart the prometheus container (it would drop the vllm scrape / TSDB continuity).

## Grafana dashboard

`grafana-minisgl-dashboard.json` (title: "minisgl Serving + Spec Decode") is importable into the
running Grafana (`localhost:3000`) against the `vllm-prometheus` datasource. It covers throughput,
mean accept-len + acceptance rate (DFlash/DDTree headline), TTFT/TPOT/e2e p50/p95/p99, request
rate/error rate, and KV-pool + GDN-state utilization. It is also provisioned in
`vllm-gfx1201/monitoring/grafana/dashboards-minisgl/` so it auto-loads there.
