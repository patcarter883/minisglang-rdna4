# Serving control panel

A small web UI and JSON API that starts, stops and tails the `docker compose` **profiles** you
already have — from a browser, without typing the compose invocation each time.

Point it at one or more compose files in `config.json`. It parses each one, groups the services by
`profiles:`, and renders a card per profile showing the services, the published ports, whether the
profile needs a GPU, and every `${VAR:-default}` the file reads. Fill in the overrides you want, hit
**Start**, and the panel runs the equivalent `docker compose --profile <p> up -d` for you, with a
stable project name so **Logs** and **Stop** keep working afterwards. You can also paste a compose
file of your own, validate it, and launch it the same way.

It is a thin orchestrator: stdlib + PyYAML, no database, no daemon. Everything it does, it does by
shelling out to `docker compose` in the stack's own directory.

## Security — read this before changing `host`

The panel is a remote-code-execution surface by design. Four specific facts, all in `server.py`:

- **Unauthenticated by default.** `token` defaults to `""` (absent or empty), and `_auth_ok()` then
  returns true for every request. The UI page itself (`GET /`) is served *before* the auth check, so
  it is always reachable regardless.
- **It executes compose YAML you POST to it** (`/api/custom` writes the body to `custom/<name>.yml`
  and runs `docker compose up -d` on it). A compose file can bind-mount `/`, add capabilities and
  run privileged, so **API access is equivalent to root on the host**, not merely "can restart a
  server".
- **`host` falls back to `0.0.0.0`** if the key is missing from `config.json` — i.e. reachable from
  anything that can route to the machine. `config.example.json` ships `127.0.0.1` instead.
- **Any website you have open can drive it (CSRF).** JSON replies carry
  `Access-Control-Allow-Origin: *`, and `_body_json()` parses the request body without looking at
  `Content-Type`. A cross-site `fetch()` with a default `text/plain` body is a CORS *simple* request:
  it is never preflighted, so it reaches the handler and is executed, and the `*` header additionally
  lets the attacking page read the reply. Binding to `127.0.0.1` does **not** help here — the browser
  is on the same host.

What to actually do:

| | |
|---|---|
| Keep `"host": "127.0.0.1"` | Reach it from another machine with an SSH tunnel (`ssh -L 7070:127.0.0.1:7070 <host>`), not by binding `0.0.0.0`. |
| Set `"token"` | A non-empty token is checked on every `/api/*` call, as `?token=<value>` or an `X-Token` header. Because it must be in the URL (or in a header, which forces a CORS preflight that this server does not answer), it also shuts the CSRF path. |
| Know the token caveat | **The bundled UI does not forward the token.** `api()` in `static/index.html` fetches `/api/...` with no query string, so with a token set the page still loads but every API call returns 401. Until the UI is taught to pass it, a token means API-only use. |

Treat "can open the panel" as "is an administrator of this box".

## Requirements

`python3` **3.9 or newer** with **PyYAML**, and the `docker` CLI with the **compose** plugin, run as
a user that can talk to the Docker daemon. No GPU-specific software is needed on the panel's side —
it only launches containers.

## Run

```bash
cd <repo>/control-panel
cp config.example.json config.json     # then edit: repos, hf_home, host, token
./run.sh                               # binds host:port from config.json
```

Open `http://127.0.0.1:7070`. `run.sh` `cd`s to its own directory first, so relative paths in
`config.json` resolve against `control-panel/` — prefer absolute ones.

To keep it running after you log out, use whatever supervisor you already have (a systemd user unit,
tmux, …). The simplest version:

```bash
mkdir -p logs && setsid nohup ./run.sh > logs/server.log 2>&1 < /dev/null &
```

(`logs/` is not in the repo — the panel creates it at startup, so make it first if you redirect into
it before the first run. `.log` is gitignored; another extension would not be.)

The panel reads `config.json` **once, at startup** — restart it after editing the file.

## What the UI gives you

- **Running services** — everything started through the panel, with live container state from
  `docker ps`, a **Logs** button (`docker compose logs --tail`) and a **Stop** button
  (`docker compose down`). It also surfaces *orphans*: a GPU-named compose project that is running
  but that `state.json` lost track of (a second start of the same profile, a restart that landed
  after a `down`). Without that, a running container would be invisible and unstoppable from the UI;
  stopping an orphan force-removes its containers by name.
- **Predefined profiles** — one card per profile discovered in each configured compose file. The
  card lists the services and images, the ports with their `${VAR:-default}`s resolved, a GPU/CPU
  tag, a model picker (when the profile reads a var ending in `MODEL` or `MODEL_ID` — plain `MODEL`,
  `VLLM_MODEL_ID`, … but never `*_MAX_MODEL_LEN` — populated from the HF cache and `~/models`),
  the promoted controls from `profile_controls`, and an expandable list of every other
  overridable env var. Values you type are passed as environment variables to the compose call —
  the tracked compose file is never modified.
- **Custom compose service** — paste or edit a compose file (optionally seeded from an existing
  profile via *load template…*), name it, tick the GPU box if it needs the cards, and **Create &
  start**. It is checked with `docker compose config` before launch, saved under `custom/`, and then
  appears in *Running services* with the same Logs/Stop controls.

## GPU profiles, and the optional arbiter hook

**By default the panel just runs `docker compose`** — for GPU profiles exactly as for CPU ones. It
exports `COMPOSE_PROJECT_NAME` = `LEASE_NAME` = `lease-<name>` for a GPU launch, which is only a
naming convention: it is what makes Stop, Logs and orphan recovery find the project again. Nothing
schedules or serialises the cards, and the card-count selector has no effect — how many cards a
service uses is whatever its own config says (`--tensor-parallel-size`, `TP`, …).

There is also a hook for boxes where several people contend for the same cards and an external
arbiter hands out exclusive leases. No such tool is bundled with this project and none is needed. If
the binaries named by the optional `gpu_lease` / `gpu_status` config keys resolve on `PATH` (or as
absolute paths), the panel wraps GPU launches in the first, binds the lease to container lifetime so
**Stop** releases it, and shows the second's output in the header. `/api/state` reports `have_lease`
and `have_gpu_status` so a client can hide the affordances when neither exists.

## Configuration — `config.json`

`config.json` is **not tracked** (the repo's `.gitignore` blanket-ignores `*.json`); the tracked
template is [`config.example.json`](config.example.json). Keys prefixed `_comment` are ignored.

| key | meaning |
|---|---|
| `host`, `port` | Bind address. **Absent `host` means `0.0.0.0`** — see Security. |
| `token` | Non-empty ⇒ `/api/*` requires `?token=…` or an `X-Token` header. |
| `repos[]` | `{id, name, dir, compose}`. `dir` is the cwd for every compose call on that stack, so the file's relative volume mounts resolve; `compose` is the file parsed for profiles. |
| `profile_defaults` | Env defaults per `"<repo>::<profile>"` — see below. |
| `profile_controls` | Env vars promoted to real UI controls per `"<repo>::<profile>"`: `{key, label, type: select\|number\|text, options, placeholder}`. |
| `gpu_lease`, `gpu_status` | Optional external arbiter binaries; bare name or absolute path. Omitted from the example — see "GPU profiles, and the optional arbiter hook" above. |
| `hf_home` | Absolute path to the HF cache the model picker lists from. No `~`/`$VAR` expansion; falls back to `$HF_HOME`. |
| `custom_dir` | Where UI-created YAML is written. Defaults to `control-panel/custom/`. |

Add another stack by appending to `repos` — any `docker-compose.yml` that uses `profiles:` works.

### `profile_defaults`

Defaults the panel applies to every launch of a profile, keyed by `<repo>::<profile>`. They are
pre-filled (and editable) in the UI, shown on the card as a `◆ panel defaults:` line, **and enforced
server-side in `do_up()`**, so a direct `/api/up` call launches identically to a UI click. An
explicit non-empty value you type wins; an empty string is inert at launch and only pre-fills the
control.

A worked `minisgl::serve` block for a two-card box pins `MODEL`, `TP: 2`, `DP: 1`, `EP: 0`,
`CONC: 2` and the `MINISGL_*` host knobs, and deliberately leaves `SERVED_NAME`, `SPEC`, `SPEC_K`,
`CTX`, `ATTN`, `MEM_RATIO`, `GRAPH_BS`, `EXTRA_ARGS` and `DRAFT` empty. `CONC` is the one worth
understanding: the compose service defaults it to 4 and it becomes `--max-running-requests` in
[`tools/serve.sh`](../tools/serve.sh), which sizes admission and graph capture — a property of how
much VRAM the cards have, i.e. exactly the kind of thing that belongs in `profile_defaults`.

### Leave the per-model knobs EMPTY

Anything pinned in `profile_defaults` **overrides `tools/serve.sh`'s per-model table**, which is
where this repo keeps its *measured* defaults — the served alias, the spec algorithm, the memory
ratio, the draft checkpoint. A value that is right for one model becomes wrong for every other one
the moment you change `MODEL` in the UI without also clearing it.

Two that bite in practice:

| key | if you pin it | cost |
|---|---|---|
| `SERVED_NAME` | to one model's alias | every checkpoint then advertises that alias in `/v1/models`, whatever it actually is. The UI field's own placeholder reads "auto (base model name)" for this reason. |
| `SPEC` | to any one algorithm | every model loses its measured `spec_default`. Pinning `none` on a checkpoint whose own default is `mtp` cost a measured 105.4 → 79.3 tok/s — **33% of a daily-driver serve's throughput**, silently. |

The rule: `profile_defaults` is for things that are genuinely a property of the **box** or of how you
like to launch — `TP`, `DP`, `EP`, `CONC`, the `MINISGL_*` box knobs. It is not for anything
`serve.sh`'s table has an opinion about. If `serve.sh` derives it per model, leave it empty and let
it. `config.example.json` carries `_comment_*` keys saying the same thing at the point of use.

## HTTP API

Every `/api/*` route requires the token when one is set; `GET /` does not.

| | route | body / query |
|---|---|---|
| GET | `/` | the UI |
| GET | `/api/state` | discovered profiles, tracked + orphaned services, `docker compose ls`, `have_lease`, `have_gpu_status` |
| GET | `/api/gpu` | arbiter status text (empty when there is none) |
| GET | `/api/models` | model ids from the HF cache and `~/models`, cached 60 s |
| GET | `/api/job` | `?id=<job>` → status, command, last 8 KB of its log |
| GET | `/api/logs` | `?key=<key>&tail=200` → `docker compose logs` for a tracked service |
| POST | `/api/up` | `{repo, profile, needs_gpu, cards, name, env}` |
| POST | `/api/down` | `{key}` |
| POST | `/api/custom` | `{name, yaml, profile, needs_gpu, cards, workdir, env}` |
| POST | `/api/custom-delete` | `{name}` |

A *key* is `<repo>::<profile>`, `custom::<name>` or `orphan::<project>`. Launches are asynchronous:
the POST returns a job id, and the job's output is streamed to `logs/<job>.log`.

## Files

| path | tracked? | what |
|---|---|---|
| `server.py`, `static/index.html`, `run.sh` | yes | the panel |
| `config.example.json` | yes | config template |
| `config.json` | **no** | your real config (`*.json` is gitignored) |
| `state.json` | **no** | runtime: what the panel has started, and the env it used, so `down`/`logs` can reproduce it |
| `logs/` | **no** | runtime: one `<job>.log` per launch/stop. The dir itself is not in the repo — the panel creates it. Only `*.log` inside it is gitignored. |
| `custom/` | not ignored | YAML created through the UI. Runtime state in practice — don't commit it unless you mean to. |

Deleting `state.json` makes the panel forget what it started; running containers then show up
through orphan recovery instead, and can still be stopped.

## Conventions it honours

- **One project per (repo, profile)**, named `<repo>-<profile>` (custom: `custom-<name>`), prefixed
  `lease-` for any GPU job whether or not an arbiter ran. `up`, `down` and `logs` must compute the
  same name — if they disagree, `docker compose down` exits 0 with only a warning, the UI shows a
  green tick, and the container is left running and unreachable.
- **cwd = the stack's own directory** for every compose call, so relative mounts in the compose file
  (`./monitoring`, `.:/engine`, …) resolve the way they do when you run it by hand.
- **It never edits your compose files.** Overrides are environment variables at launch; the files
  already read them as `${VAR:-default}`.
- **Card count is a guess you can override per launch**, inferred from `--tensor-parallel-size`,
  `--tp`, `-dp`, `--data-parallel-size` or `MINISGL_TP` in the service command, clamped to 1–2.
