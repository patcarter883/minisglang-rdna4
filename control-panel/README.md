# GPU Serving Control Panel

A small web UI to start/stop the predefined `docker compose` profiles in **minisgl-rdna4**
and **vllm-gfx1201**, and to create/launch **custom** compose services — from your laptop or
anything on the local network.

It is a thin host-side orchestrator: it shells out to `docker compose` and the shared
`gpu-lease` arbiter exactly as you would by hand. GPU profiles are auto-wrapped in
`gpu-lease -n <cards> --detach --name <slug> -- docker compose --profile <p> up -d`, so the
panel coordinates the two gfx1201 cards through the same flock as every other agent. CPU-only
profiles (e.g. vLLM `monitoring`) are started without a lease.

## Run

```bash
cd /home/pat/code/minisgl-rdna4/control-panel
./run.sh                      # binds 0.0.0.0:7070 (see config.json)
```

Then open:

- **Locally:** http://localhost:7070
- **Over the LAN:** http://10.50.1.51:7070  (this box's LAN IP; use whichever `hostname -I`/`ip addr` reports)

To keep it running after you log out:

```bash
setsid nohup ./run.sh > logs/server.out 2>&1 < /dev/null &
# or, as a persistent service:  systemd --user, tmux, etc.
```

## What it does

- **Running services** — everything started through the panel, with live container state, a
  **Logs** button (`docker compose logs`), and a **Stop** button (`docker compose down`, which
  also releases the GPU lease because the lease is bound to container lifetime via `--detach`).
- **Predefined profiles** — auto-discovered from both compose files, grouped by profile.
  Each card shows the services, ports, whether it needs a GPU and how many cards (inferred from
  `--tensor-parallel-size` / `-dp` / `MINISGL_TP`), and an expandable list of every
  `${VAR:-default}` you can override (model id, ports, memory ratio, tool parser, …). Set a
  value, pick the card count, hit **Start**.
- **Custom compose service** — paste/edit a compose YAML (optionally seeded from an existing
  profile via the *load template…* menu), name it, tick *lease GPU* if it needs cards, and
  **Create & start**. The file is validated with `docker compose config`, saved under
  `custom/`, and launched. It then appears in *Running services* with the same Logs/Stop
  controls.

## Configuration — `config.json`

```jsonc
{
  "host": "0.0.0.0",      // 0.0.0.0 = reachable on the LAN; 127.0.0.1 = local only
  "port": 7070,
  "token": "",            // set a non-empty string to require ?token=... on every /api call
  "gpu_lease":  "/home/pat/code/gpu-lease/bin/gpu-lease",
  "gpu_status": "/home/pat/code/gpu-lease/bin/gpu-status",
  "repos": [
    { "id": "minisgl", "name": "minisglang (rdna4)", "dir": ".../minisgl-rdna4",  "compose": ".../docker-compose.yml" },
    { "id": "vllm",    "name": "vLLM (gfx1201)",     "dir": ".../vllm-gfx1201",   "compose": ".../docker-compose.yml" }
  ]
}
```

Add another stack by appending to `repos` (any `docker-compose.yml` with `profiles:` works).

## Profile defaults (safe launch settings)

The panel applies per-profile env-var defaults to every service it launches, so you never
inherit an unsafe compose default by accident. They are defined in `config.json` under
`profile_defaults`, keyed by `<repo>::<profile>`:

```jsonc
"profile_defaults": {
  "minisgl::serve": { "MINISGL_EXTRA_ARGS": "--max-running-requests 32" }
}
```

- In the UI these show as a `◆ panel defaults:` line on the profile card and are **pre-filled
  (and editable)** in the env list — an explicit value you type overrides the default.
- They are **enforced server-side at launch too** (even for direct `/api/up` calls), so a bare
  launch is always safe. To use a compose default instead, type the value you want.
- Add defaults for any profile by adding a `"<repo>::<profile>": { ENV: "value" }` entry and
  `systemctl --user restart gpu-control`.

Shipped default: `minisgl::serve` caps `--max-running-requests` at 32 (see the OOM note below).

## Serve gotchas (from an end-to-end test)

- **minisgl `serve` OOMs on 16 GB cards with the compose defaults.** The engine defaults
  `--max-running-requests` to **256**, and the GDN recurrent-state cache pre-allocates
  `256 × fp32` state slots (~7.6 GB/card) on top of the 35B weights → `CUDA out of memory`.
  Because the profile has `restart: unless-stopped`, it then crash-loops while holding both
  cards. **Fix via the Model card's env overrides:** set `MINISGL_EXTRA_ARGS` to
  `--max-running-requests 32` (single-user is fine at 32; validated: model loads, serves, and
  answers on `:1919/v1/chat/completions`). `MINISGL_SSM_BF16=1` halves the state cache too.
- This is a serving-config matter, not a panel issue — a hand-typed `docker compose --profile
  serve up` with the same defaults OOMs identically.

## Security

The panel executes `docker` and `gpu-lease` with **no authentication by default** and, with
`host: 0.0.0.0`, is reachable by anything on your LAN — effectively remote shell over your GPUs.
Only run it on a trusted network. To lock it down, set `"token"` in `config.json`; the UI then
requires `?token=<value>` in the URL and the API rejects calls without it. For local-only use,
set `"host": "127.0.0.1"`.

## Notes / conventions this honors

- **One lease per (repo, profile)**, named `<repo>-<profile>` (custom: `custom-<name>`), so
  `up` and `down` target the same `COMPOSE_PROJECT_NAME`/`LEASE_NAME` and containers resolve.
- **cwd = the repo dir** for every compose call, so the relative volume mounts in the compose
  files (`./monitoring`, `./test`, `./.hf-cache`, `.:/engine`) resolve correctly.
- It does **not** modify the tracked `docker-compose.yml` files. Overrides are passed as
  environment variables at launch (the compose files already read them as `${VAR:-default}`).
- Card count is a guess you can override per-launch. `-n` is **how many** cards, never which —
  the arbiter assigns the lowest free card(s), same as the CLI.
