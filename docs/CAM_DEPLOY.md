# Deploying a CAM serve

One command stands up the CAM (Canonical Associative Memory) serve — the 35B model + the `/cam/*`
memory API — on two gfx1201 cards.

## Prerequisites

- Two RDNA4 (gfx1201) cards, both free — the 35B serve is TP=2 and takes both.
- The serve image `minisgl-rdna4:lean` built (`docker compose build`), and the HF model cached
  (`MINISGL_MODEL`, default `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`, under `$HF_HOME`).
- `docker` and `curl` on `$PATH`.

If the box runs a card arbiter (a `gpu-lease` binary on `$PATH`, as on a machine where several jobs
share the same GPUs), `cam-serve up` routes the boot through it so the launch queues instead of
colliding. That is optional — with no arbiter installed the script runs `docker compose` directly
and everything below is identical.

## Lifecycle

```bash
scripts/cam-serve up        # boot on both cards, wait until healthy, print the URL (~60-90s)
scripts/cam-serve status    # serve health + store stats
scripts/cam-serve logs      # follow serve logs
scripts/cam-serve down      # stop the serve and free the cards
scripts/cam-serve restart   # down then up
```

`up` boots the compose `cam` profile with the whitened-GTE semantic key on, polls `/health`, and
prints `READY at http://127.0.0.1:1919`. `down` stops the container and frees the cards. Both
address the job by its compose project name, `lease-minisgl-<CAM_SERVE_NAME>` (the same name the
optional arbiter would use, so the two paths are interchangeable) — so stop it with
`scripts/cam-serve down` rather than a bare `docker compose down`.

## Configuration (env, all optional)

| Env | Default | Meaning |
|---|---|---|
| `MINISGL_HOST_PORT` | `1919` | host port |
| `MINISGL_CAM_GTE_KEY` | `1` | whitened-GTE semantic key (paraphrase-robust recall) |
| `MINISGL_CAM_API_TOKEN` | — | single admin token (any namespace); unset = open (dev) |
| `MINISGL_CAM_TOKENS` | — | per-tenant `{token: "<ns>"｜["<ns>",..]｜"*"}` — see [CAM_API.md](CAM_API.md) |
| `MINISGL_CAM_MAX_FACTS` | `0` | per-namespace fact cap (LRU evict; 0 = unlimited) |
| `MINISGL_CAM_MAX_NAMESPACES` | `0` | tenant-namespace cap (429 past it; 0 = unlimited) |
| `CAM_SERVE_NAME` | `cam` | instance name (run several serves with distinct names) |

```bash
# a locked-down multi-tenant serve:
MINISGL_CAM_TOKENS='{"acme-tok":"acme","admin-tok":"*"}' MINISGL_CAM_MAX_NAMESPACES=50 scripts/cam-serve up
```

Then talk to it with the [`CAMClient`](../python/minisgl/cam/client.py) or `curl` (see
[CAM_API.md](CAM_API.md)).

## Troubleshooting

- **Boot fails: `RuntimeError: fused gemm1+silu needs the 'wmma' kernel; got id 7`** — a MoE fusion
  incompatibility with the 35B AWQ path (being fixed / rebuilt separately). Work around with
  `MINISGL_MOE_FUSED_SILU=0 scripts/cam-serve up`.
- **Boot fails on the cards / `up` hangs before the container starts** — something else is holding
  the GPUs (`rocm-smi` shows what). The 35B serve needs both cards to itself. On a box running a
  card arbiter, `up` waits for them by design and `gpu-status` names the current holder.
- **Timed out waiting for health** — `scripts/cam-serve logs` to see the boot; the 35B TP=2 takes ~60-90s.
- **Persistence**: the store autosaves atomically to `./.cam_store/store.pt` (+ `.bak`); it reloads on
  boot. `POST /cam/save` forces a snapshot.
