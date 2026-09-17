#!/usr/bin/env python3
"""Web control panel for docker-compose serving profiles.

A dependency-light (stdlib + PyYAML) web UI + JSON API to start/stop the predefined
docker-compose profiles in the repos listed in config.json, and to create/launch custom
compose services.

GPU arbitration is OPTIONAL. If a `gpu-lease` binary is found on PATH (or at the path named
in config.json), GPU profiles are wrapped in it so several agents can share a box's cards.
If it is absent -- the normal case for a single-user machine -- `docker compose` is run
directly and the panel behaves identically otherwise.

Run:  python3 server.py            (binds per config.json; 127.0.0.1:7070 by default)
"""
import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())
CUSTOM_DIR = Path(CONFIG.get("custom_dir") or (HERE / "custom"))
LOG_DIR = HERE / "logs"
STATE_FILE = HERE / "state.json"
CUSTOM_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Resolve the optional GPU arbiter ONCE at startup. shutil.which() accepts a bare name (looked up
# on PATH) or an absolute path (checked for existence + executability), so config.json may carry
# either. None means "not installed on this box" -- every GPU launch then runs docker compose
# directly. This is what makes the panel portable off the box it was written for.
GPU_LEASE = shutil.which(CONFIG.get("gpu_lease", "gpu-lease"))
GPU_STATUS = shutil.which(CONFIG.get("gpu_status", "gpu-status"))
HAVE_LEASE = GPU_LEASE is not None
TOKEN = CONFIG.get("token", "") or ""
# Env-var defaults the panel applies to services it launches, keyed by "<repo>::<profile>".
# Surfaced in the UI (pre-filled + editable) and enforced at launch — see do_up().
PROFILE_DEFAULTS = CONFIG.get("profile_defaults", {}) or {}
PROFILE_CONTROLS = CONFIG.get("profile_controls", {}) or {}   # knobs promoted to real UI controls

ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")
SLUG_RE = re.compile(r"[^a-z0-9]+")
# The env var that selects the served model: MINISGL_MODEL / VLLM_MODEL_ID / ZAYA_MODEL_ID.
# Deliberately excludes *_MAX_MODEL_LEN (ends in MODEL_LEN, not MODEL/MODEL_ID).
MODEL_VAR_RE = re.compile(r"(^|_)MODEL(_ID)?$")   # plain MODEL too: minisgl's serve service
                                                  # takes MODEL=<alias|hf-id> (tools/serve.sh)

# Where the panel looks for locally cached HF models to populate the model picker. config.json wins,
# then $HF_HOME, then huggingface_hub's own default (~/.cache/huggingface) — a per-user path, so the
# panel finds the cache on any machine instead of a directory that exists only on the author's box.
HF_HOME = Path(CONFIG.get("hf_home") or os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
_models_cache = {"t": 0.0, "v": []}

_state_lock = threading.Lock()
_jobs = {}          # job_id -> {status, cmd, log, returncode, started, finished, kind, key}
_jobs_lock = threading.Lock()
_job_seq = [0]


# --------------------------------------------------------------------------- state
def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_state(state):
    with _state_lock:
        STATE_FILE.write_text(json.dumps(state, indent=2))


def slug(*parts):
    s = "-".join(str(p) for p in parts if p)
    return SLUG_RE.sub("-", s.lower()).strip("-")


# --------------------------------------------------------------------------- discovery
def _iter_env_vars(node, acc):
    """Collect ${VAR:-default} references anywhere under a service subtree."""
    if isinstance(node, dict):
        for v in node.values():
            _iter_env_vars(v, acc)
    elif isinstance(node, list):
        for v in node:
            _iter_env_vars(v, acc)
    elif isinstance(node, str):
        for m in ENV_RE.finditer(node):
            name, default = m.group(1), m.group(2)
            if name not in acc or (acc[name] == "" and default):
                acc[name] = default if default is not None else ""


def _needs_gpu(svc):
    devs = svc.get("devices") or []
    for d in devs:
        if "/dev/kfd" in str(d) or "/dev/dri" in str(d):
            return True
    return False


def _detect_cards(svc):
    """Guess the card count from the service command (tp / dp / -n hints)."""
    cmd = svc.get("command") or []
    text = " ".join(str(c) for c in cmd) if isinstance(cmd, list) else str(cmd)
    best = 1
    for pat in (r"tensor-parallel-size[= ](\d+)", r"--tp[= ](\d+)", r"-dp[= ](\d+)",
                r"data-parallel-size[= ](\d+)", r"MINISGL_TP[^0-9]*(\d+)"):
        for m in re.finditer(pat, text):
            best = max(best, int(m.group(1)))
    # minisgl serve defaults --tp 2 via ${MINISGL_TP:-2}
    for m in ENV_RE.finditer(text):
        if m.group(1) in ("MINISGL_TP",) and m.group(2):
            best = max(best, int(re.sub(r"\D", "", m.group(2)) or "1"))
    return max(1, min(2, best))


def _resolve_defaults(s):
    """Render ${VAR:-default} -> default for display (leaves bare ${VAR} as-is)."""
    return ENV_RE.sub(lambda m: m.group(2) if m.group(2) is not None else m.group(0), str(s))


def _ports(svc):
    out = []
    for p in svc.get("ports") or []:
        out.append(_resolve_defaults(p))
    return out


def discover_repo(repo):
    """Parse one compose file and group its services by profile."""
    path = Path(repo["compose"])
    if not path.exists():
        return {"id": repo["id"], "name": repo["name"], "error": f"missing {path}", "profiles": []}
    try:
        doc = yaml.safe_load(path.read_text())
    except Exception as e:
        return {"id": repo["id"], "name": repo["name"], "error": str(e), "profiles": []}

    services = doc.get("services") or {}
    profiles = {}  # profile_name -> {services, ports, env, needs_gpu, cards}
    for sname, svc in services.items():
        if not isinstance(svc, dict):
            continue
        svc_profiles = svc.get("profiles") or ["_default"]
        env_acc = {}
        _iter_env_vars(svc, env_acc)
        gpu = _needs_gpu(svc)
        cards = _detect_cards(svc) if gpu else 0
        for prof in svc_profiles:
            p = profiles.setdefault(prof, {
                "profile": prof, "services": [], "ports": [], "env": {},
                "needs_gpu": False, "cards": 1, "images": [],
                "controls": PROFILE_CONTROLS.get(f"{repo['id']}::{prof}", []),
            })
            p["services"].append(sname)
            p["ports"].extend(_ports(svc))
            p["env"].update(env_acc)
            if svc.get("image"):
                p["images"].append(str(svc["image"]))
            if gpu:
                p["needs_gpu"] = True
                p["cards"] = max(p["cards"], cards)

    prof_list = []
    for prof, info in sorted(profiles.items()):
        info["ports"] = sorted(set(info["ports"]))
        info["images"] = sorted(set(info["images"]))
        # env vars sorted with a helpful subset first
        info["env"] = dict(sorted(info["env"].items()))
        model_var = next((k for k in info["env"] if MODEL_VAR_RE.search(k)), None)
        info["model_var"] = model_var
        info["model_default"] = info["env"].get(model_var, "") if model_var else ""
        info["defaults"] = PROFILE_DEFAULTS.get(f"{repo['id']}::{prof}", {})
        prof_list.append(info)
    return {"id": repo["id"], "name": repo["name"], "dir": repo["dir"],
            "compose": str(path), "profiles": prof_list}


# --------------------------------------------------------------------------- docker introspection
def sh(cmd, cwd=None, env=None, timeout=30):
    try:
        r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except FileNotFoundError as e:
        return 127, "", str(e)


def compose_ls():
    rc, out, _ = sh(["docker", "compose", "ls", "-a", "--format", "json"])
    if rc != 0 or not out.strip():
        return []
    try:
        return json.loads(out)
    except Exception:
        return []


def docker_ps():
    rc, out, _ = sh(["docker", "ps", "-a", "--format", "{{json .}}"])
    if rc != 0:
        return []
    rows = []
    for line in out.splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def gpu_status_text():
    """Arbiter status, or '' when no arbiter is installed (the UI hides the panel then)."""
    if not GPU_STATUS:
        return ""
    rc, out, err = sh([GPU_STATUS])
    return out if rc == 0 else (out + err)


def list_models():
    """Model ids/paths that a serve profile can be pointed at: cached HF repos + loose
    snapshot dirs + local model dirs. Cached 60s so the 5s UI poll stays cheap."""
    now = time.time()
    if _models_cache["v"] and now - _models_cache["t"] < 60:
        return _models_cache["v"]
    ids = set()
    hub = HF_HOME / "hub"
    if hub.is_dir():
        for d in hub.iterdir():
            if d.is_dir() and d.name.startswith("models--"):
                ids.add("/".join(d.name[len("models--"):].split("--")))
    # loose snapshot dirs directly under the HF cache root (e.g. Laguna-XS.2-RXF) -> container path
    if HF_HOME.is_dir():
        for d in HF_HOME.iterdir():
            if d.is_dir() and d.name not in ("hub", "modules", "datasets") and (d / "config.json").exists():
                ids.add(f"/root/.cache/huggingface/{d.name}")
    # local checkpoint dirs mounted into containers (ZAYA etc. -> /models)
    mdir = Path.home() / "models"
    if mdir.is_dir():
        for d in mdir.iterdir():
            if d.is_dir() and (d / "config.json").exists():
                ids.add(f"/models/{d.name}")
    v = sorted(ids, key=str.lower)
    _models_cache.update(t=now, v=v)
    return v


# --------------------------------------------------------------------------- job runner
def new_job(cmd, cwd, env, kind, key):
    with _jobs_lock:
        _job_seq[0] += 1
        jid = f"job{_job_seq[0]}"
    log_path = LOG_DIR / f"{jid}.log"
    job = {"id": jid, "status": "running", "cmd": cmd, "log": str(log_path),
           "returncode": None, "started": time.time(), "finished": None,
           "kind": kind, "key": key}
    with _jobs_lock:
        _jobs[jid] = job

    def run():
        with open(log_path, "w") as lf:
            lf.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n\n")
            lf.flush()
            try:
                p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=lf,
                                     stderr=subprocess.STDOUT, text=True)
                rc = p.wait()
            except Exception as e:
                lf.write(f"\n[control-panel] launch error: {e}\n")
                rc = 1
        job["returncode"] = rc
        job["status"] = "done" if rc == 0 else "error"
        job["finished"] = time.time()

    threading.Thread(target=run, daemon=True).start()
    return job


def job_view(job):
    log = ""
    try:
        log = Path(job["log"]).read_text()[-8000:]
    except Exception:
        pass
    return {k: job[k] for k in ("id", "status", "returncode", "kind", "key", "started", "finished")} | {
        "cmd": " ".join(shlex.quote(c) for c in job["cmd"]), "log": log}


# --------------------------------------------------------------------------- actions
def repo_by_id(rid):
    for r in CONFIG["repos"]:
        if r["id"] == rid:
            return r
    return None


def _project(rec):
    """The real docker-compose project name.

    The 'lease-' prefix is the PANEL's naming convention for any GPU job, not evidence that a
    lease ran. When gpu-lease is installed it reproduces the same name itself
    (LEASE_NAME/COMPOSE_PROJECT_NAME = lease-<name>); when it is absent do_up() exports that
    identical pair by hand. Both must agree, because Stop/Logs look a job up by this name --
    if an unleased GPU job used the bare name, `docker compose -p lease-<name> down` would exit
    0 with only a warning, the UI would show a green tick, and the container would be left
    running and unreachable. CPU jobs use the bare name."""
    p = rec.get("project")
    if p:
        return p
    return ("lease-" + rec["name"]) if rec.get("needs_gpu") else rec["name"]


def do_up(repo, profile, needs_gpu, cards, name, env_overrides, extra_args=None):
    # Panel defaults form the base; an explicit non-empty user value overrides one. This holds
    # even for direct API calls, so nothing launched through the panel inherits an unsafe compose
    # default (e.g. minisgl serve's max-running-requests=256 → GDN-state OOM).
    merged = dict(PROFILE_DEFAULTS.get(f"{repo['id']}::{profile}", {}))
    merged.update({k: v for k, v in (env_overrides or {}).items() if v not in (None, "")})
    env_overrides = merged
    env = dict(os.environ)
    env.update({k: str(v) for k, v in (env_overrides or {}).items() if v not in (None, "")})
    compose_cmd = ["docker", "compose", "--profile", profile, "up", "-d"]
    if extra_args:
        compose_cmd += extra_args
    # Three cases, and all three MUST end up with the project name _project() will compute.
    if needs_gpu and HAVE_LEASE:
        # Arbitrated: gpu-lease exports COMPOSE_PROJECT_NAME=LEASE_NAME=lease-<name> for us.
        cmd = [GPU_LEASE, "-n", str(cards), "--detach", "--name", name, "--"] + compose_cmd
    elif needs_gpu:
        # GPU profile, no arbiter installed: run compose directly, but reproduce the exact pair
        # gpu-lease would have exported, so Stop/Logs and orphan recovery keep working.
        env.setdefault("COMPOSE_PROJECT_NAME", f"lease-{name}")
        env.setdefault("LEASE_NAME", f"lease-{name}")
        cmd = compose_cmd
    else:
        # CPU-only profile (e.g. monitoring): no lease, but keep a stable project name.
        env.setdefault("COMPOSE_PROJECT_NAME", name)
        env.setdefault("LEASE_NAME", name)
        cmd = compose_cmd
    key = f"{repo['id']}::{profile}"
    project = f"lease-{name}" if needs_gpu else name
    job = new_job(cmd, cwd=repo["dir"], env=env, kind="up", key=key)

    st = load_state()
    st[key] = {"repo": repo["id"], "profile": profile, "name": name, "project": project,
               "needs_gpu": needs_gpu, "cards": cards, "compose": repo["compose"],
               "dir": repo["dir"], "custom": False, "started": time.time(), "job": job["id"],
               "env": {k: v for k, v in (env_overrides or {}).items() if v not in (None, "")}}
    save_state(st)
    return job


def do_down(key):
    st = load_state()
    rec = st.get(key)
    if not rec:
        # Orphan / lost-track path (see build_state orphan recovery): no state record, but the UI
        # surfaced a running `lease-*` project. Stop it by force-removing its live containers —
        # `docker rm -f` releases the gpu-lease on container exit, and this is robust to compose/state
        # desync (we don't have the compose file/env for an untracked project anyway).
        if key.startswith("orphan::"):
            proj = key[len("orphan::"):]
            _by_name, projects, _ = running_index()
            names = [r.get("Names") for r in projects.get(proj, []) if r.get("Names")]
            if not names:
                return None, "orphan project has no running containers (already stopped?)"
            job = new_job(["docker", "rm", "-f", *names], cwd=str(HERE),
                          env=dict(os.environ), kind="down", key=key)
            return job, None
        return None, "no tracked service for that key"
    proj = _project(rec)
    env = dict(os.environ)
    env["COMPOSE_PROJECT_NAME"] = proj
    env["LEASE_NAME"] = proj
    # preserve the env used at up so container_name / port templates resolve identically
    for k, v in (rec.get("env") or {}).items():
        env[k] = str(v)
    if rec.get("custom"):
        cmd = ["docker", "compose", "-p", proj, "-f", rec["compose"]]
        if rec.get("profile"):
            cmd += ["--profile", rec["profile"]]
        cmd += ["down"]
        cwd = rec.get("dir") or str(CUSTOM_DIR)
    else:
        cmd = ["docker", "compose", "-p", proj, "--profile", rec["profile"], "down"]
        cwd = rec["dir"]
    job = new_job(cmd, cwd=cwd, env=env, kind="down", key=key)
    del st[key]
    save_state(st)
    return job, None


def do_logs(key, tail=200):
    st = load_state()
    rec = st.get(key)
    if not rec:
        return "no tracked service for that key (it may have been started outside the panel)"
    proj = _project(rec)
    env = dict(os.environ)
    env["COMPOSE_PROJECT_NAME"] = proj
    env["LEASE_NAME"] = proj
    for k, v in (rec.get("env") or {}).items():
        env[k] = str(v)
    if rec.get("custom"):
        cmd = ["docker", "compose", "-p", proj, "-f", rec["compose"]]
        if rec.get("profile"):
            cmd += ["--profile", rec["profile"]]
        cmd += ["logs", "--no-color", "--tail", str(tail)]
        cwd = rec.get("dir") or str(CUSTOM_DIR)
    else:
        cmd = ["docker", "compose", "-p", proj, "--profile", rec["profile"], "logs", "--no-color", "--tail", str(tail)]
        cwd = rec["dir"]
    rc, out, err = sh(cmd, cwd=cwd, env=env, timeout=25)
    return out + (("\n" + err) if err else "")


def do_custom(name, yaml_text, profile, needs_gpu, cards, workdir, env_overrides):
    name_slug = slug(name) or f"custom-{int(time.time())}"
    # validate it parses
    try:
        yaml.safe_load(yaml_text)
    except Exception as e:
        return None, f"invalid YAML: {e}"
    fpath = CUSTOM_DIR / f"{name_slug}.yml"
    fpath.write_text(yaml_text)
    # validate via docker compose config
    rc, out, err = sh(["docker", "compose", "-f", str(fpath), "config"], timeout=25)
    if rc != 0:
        return None, f"docker compose rejected the file:\n{err or out}"

    cwd = workdir or str(CUSTOM_DIR)
    env = dict(os.environ)
    env.update({k: str(v) for k, v in (env_overrides or {}).items() if v not in (None, "")})
    lease_name = slug("custom", name_slug)
    compose_cmd = ["docker", "compose", "-f", str(fpath)]
    if profile:
        compose_cmd += ["--profile", profile]
    compose_cmd += ["up", "-d"]
    # Same three-way split as do_up() — see the note on _project() for why the unleased GPU path
    # must still reproduce the 'lease-' prefixed pair.
    if needs_gpu and HAVE_LEASE:
        cmd = [GPU_LEASE, "-n", str(cards), "--detach", "--name", lease_name, "--"] + compose_cmd
    elif needs_gpu:
        env.setdefault("COMPOSE_PROJECT_NAME", f"lease-{lease_name}")
        env.setdefault("LEASE_NAME", f"lease-{lease_name}")
        cmd = compose_cmd
    else:
        env.setdefault("COMPOSE_PROJECT_NAME", lease_name)
        env.setdefault("LEASE_NAME", lease_name)
        cmd = compose_cmd
    key = f"custom::{name_slug}"
    project = f"lease-{lease_name}" if needs_gpu else lease_name
    job = new_job(cmd, cwd=cwd, env=env, kind="up", key=key)
    st = load_state()
    st[key] = {"repo": "custom", "profile": profile or "", "name": lease_name,
               "project": project, "needs_gpu": needs_gpu, "cards": cards, "compose": str(fpath),
               "dir": cwd, "custom": True, "started": time.time(), "job": job["id"],
               "env": {k: v for k, v in (env_overrides or {}).items() if v not in (None, "")}}
    save_state(st)
    return job, None


def list_customs():
    out = []
    for f in sorted(CUSTOM_DIR.glob("*.yml")):
        out.append({"name": f.stem, "path": str(f), "text": f.read_text()})
    return out


# --------------------------------------------------------------------------- aggregate state
def running_index():
    """Map container-name -> ps row, and compose project -> status."""
    ps = docker_ps()
    by_name = {row.get("Names", ""): row for row in ps}
    projects = {}
    for row in ps:
        labels = row.get("Labels", "")
        # docker ps Labels is a comma string k=v; find compose project
        m = re.search(r"com\.docker\.compose\.project=([^,]+)", labels)
        if m:
            projects.setdefault(m.group(1), []).append(row)
    return by_name, projects, ps


def build_state():
    by_name, projects, ps = running_index()
    st = load_state()
    repos = [discover_repo(r) for r in CONFIG["repos"]]

    # annotate each tracked service with live status
    tracked = []
    tracked_projects = set()
    for key, rec in st.items():
        proj = _project(rec)
        tracked_projects.add(proj)
        rows = projects.get(proj, [])
        # also match by container-name prefix (compose names <project>-<svc>-N or explicit)
        if not rows:
            rows = [r for n, r in by_name.items() if n.startswith(proj)]
        statuses = [r.get("State", r.get("Status", "")) for r in rows]
        up = any("running" in s.lower() or "up" in s.lower() for s in statuses)
        tracked.append({**rec, "key": key, "containers": [
            {"name": r.get("Names"), "state": r.get("State"), "status": r.get("Status"),
             "ports": r.get("Ports")} for r in rows], "up": up})

    # SELF-HEALING orphan recovery: a running `lease-*` compose project that state.json lost track of
    # — e.g. a 2nd start of the same profile overwrote the record, or a queued restart came up AFTER a
    # `down` deleted the key. Without this the panel shows "no running services" while a container (and
    # its 2-card lease) is actually up and unstoppable from the UI. Reflect docker reality: surface any
    # running lease-* project not already tracked, with a synthetic `orphan::<project>` key that
    # do_down() force-stops. (Only projects with a RUNNING container; exited leftovers are ignored.)
    for proj, rows in projects.items():
        if proj in tracked_projects or not proj.startswith("lease-"):
            continue
        live = [r for r in rows
                if "running" in (r.get("State", "") or "").lower()
                or "up" in (r.get("Status", "") or "").lower()]
        if not live:
            continue
        tracked.append({
            "key": f"orphan::{proj}", "orphan": True, "project": proj,
            "name": proj[len("lease-"):], "repo": "?", "profile": "",
            "needs_gpu": True, "custom": False, "started": 0, "up": True,
            "containers": [{"name": r.get("Names"), "state": r.get("State"),
                            "status": r.get("Status"), "ports": r.get("Ports")} for r in live],
        })

    return {
        "repos": repos,
        "tracked": sorted(tracked, key=lambda t: t.get("started", 0), reverse=True),
        "compose_ls": compose_ls(),
        "gpu_status": gpu_status_text(),
        # Lets the UI hide the arbiter affordances (lease pills, card-count selector) on a box
        # that has no gpu-lease installed, rather than rendering permanently empty controls.
        "have_lease": HAVE_LEASE,
        "have_gpu_status": GPU_STATUS is not None,
        "customs": list_customs(),
        "now": time.time(),
    }


# --------------------------------------------------------------------------- HTTP
INDEX_HTML = (HERE / "static" / "index.html")


class Handler(BaseHTTPRequestHandler):
    server_version = "gpu-control/1.0"

    def log_message(self, *a):
        pass  # quiet

    # ---- helpers
    def _auth_ok(self):
        if not TOKEN:
            return True
        q = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(q)
        if params.get("token", [""])[0] == TOKEN:
            return True
        if self.headers.get("X-Token", "") == TOKEN:
            return True
        return False

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text, code=200, ctype="text/plain; charset=utf-8"):
        body = text.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body_json(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode())
        except Exception:
            return {}

    # ---- routes
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            return self._send_text(INDEX_HTML.read_text(), ctype="text/html; charset=utf-8")
        if not self._auth_ok():
            return self._send_json({"error": "unauthorized"}, 401)
        if path == "/api/state":
            return self._send_json(build_state())
        if path == "/api/gpu":
            return self._send_text(gpu_status_text())
        if path == "/api/models":
            return self._send_json({"models": list_models()})
        if path == "/api/job":
            jid = urllib.parse.parse_qs(parsed.query).get("id", [""])[0]
            with _jobs_lock:
                job = _jobs.get(jid)
            if not job:
                return self._send_json({"error": "no such job"}, 404)
            return self._send_json(job_view(job))
        if path == "/api/logs":
            q = urllib.parse.parse_qs(parsed.query)
            key = q.get("key", [""])[0]
            tail = int(q.get("tail", ["200"])[0])
            return self._send_text(do_logs(key, tail))
        return self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if not self._auth_ok():
            return self._send_json({"error": "unauthorized"}, 401)
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        data = self._body_json()

        if path == "/api/up":
            repo = repo_by_id(data.get("repo"))
            if not repo:
                return self._send_json({"error": "unknown repo"}, 400)
            profile = data.get("profile")
            needs_gpu = bool(data.get("needs_gpu", True))
            cards = int(data.get("cards", 1))
            name = data.get("name") or slug(repo["id"], profile)
            job = do_up(repo, profile, needs_gpu, cards, name, data.get("env") or {})
            return self._send_json({"job": job["id"]})

        if path == "/api/down":
            job, err = do_down(data.get("key"))
            if err:
                return self._send_json({"error": err}, 400)
            return self._send_json({"job": job["id"]})

        if path == "/api/custom":
            job, err = do_custom(
                data.get("name", ""), data.get("yaml", ""), data.get("profile", ""),
                bool(data.get("needs_gpu", False)), int(data.get("cards", 1)),
                data.get("workdir", ""), data.get("env") or {})
            if err:
                return self._send_json({"error": err}, 400)
            return self._send_json({"job": job["id"]})

        if path == "/api/custom-delete":
            name = slug(data.get("name", ""))
            f = CUSTOM_DIR / f"{name}.yml"
            if f.exists():
                f.unlink()
                return self._send_json({"ok": True})
            return self._send_json({"error": "not found"}, 404)

        return self._send_json({"error": "not found"}, 404)


def main():
    host = CONFIG.get("host", "0.0.0.0")
    port = int(CONFIG.get("port", 7070))
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"gpu-control panel on http://{host}:{port}  (Ctrl-C to stop)")
    if TOKEN:
        print(f"  auth token required: append ?token={TOKEN}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
