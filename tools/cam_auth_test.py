"""Unit test for CAM multi-tenant auth: token->namespace registry, principal resolution, and per-namespace
authorization. Pure logic (no serve). Run: PYTHONPATH=python python tools/cam_auth_test.py"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))
from fastapi import HTTPException  # noqa: E402
from minisgl.server import cam_api as A  # noqa: E402

ok = tot = 0


def check(name, cond):
    global ok, tot
    tot += 1; ok += bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def setenv(**kw):
    for k in ("MINISGL_CAM_API_TOKEN", "MINISGL_CAM_TOKENS"):
        os.environ.pop(k, None)
    for k, v in kw.items():
        os.environ[k] = v


def principal(auth):
    return A._cam_principal(authorization=auth)


def raises401(auth):
    try:
        principal(auth); return False
    except HTTPException as e:
        return e.status_code == 401


# --- no tokens -> open (dev) ----------------------------------------------------------------------
setenv()
check("no tokens configured -> open (admin)", principal(None) == A._ADMIN)
check("open: any namespace authorized", A._authorized(principal(None), "anything"))

# --- legacy single token -> admin (backward compat) ----------------------------------------------
setenv(MINISGL_CAM_API_TOKEN="legacy")
check("legacy token resolves to admin", principal("Bearer legacy") == A._ADMIN)
check("legacy admin -> any namespace", A._authorized(principal("Bearer legacy"), "tenantX"))
check("missing token -> 401", raises401(None))
check("wrong token -> 401", raises401("Bearer nope"))

# --- scoped multi-tenant registry ----------------------------------------------------------------
setenv(MINISGL_CAM_TOKENS='{"tokA": "acme", "tokB": ["b1", "b2"], "root": "*"}')
pA, pB, pRoot = principal("Bearer tokA"), principal("Bearer tokB"), principal("Bearer root")
check("scoped token A -> its namespace only", A._authorized(pA, "acme") and not A._authorized(pA, "b1"))
check("scoped token B -> its list", A._authorized(pB, "b1") and A._authorized(pB, "b2") and not A._authorized(pB, "acme"))
check("admin '*' token -> any namespace", pRoot == A._ADMIN and A._authorized(pRoot, "whatever"))
check("default namespace needs explicit grant", not A._authorized(pA, None))   # A only has 'acme'
check("unknown token still 401 under registry", raises401("Bearer ghost"))

# --- _require_ns raises 403 on cross-tenant ------------------------------------------------------
def raises403(p, ns):
    try:
        A._require_ns(p, ns); return False
    except HTTPException as e:
        return e.status_code == 403
check("cross-tenant namespace -> 403", raises403(pA, "b1"))
check("own namespace -> no raise", not raises403(pA, "acme"))

# --- both legacy + scoped set (admin + tenants) --------------------------------------------------
setenv(MINISGL_CAM_API_TOKEN="admin", MINISGL_CAM_TOKENS='{"t1": "n1"}')
check("legacy+scoped: admin is admin", principal("Bearer admin") == A._ADMIN)
check("legacy+scoped: t1 scoped to n1", A._authorized(principal("Bearer t1"), "n1") and not A._authorized(principal("Bearer t1"), "n2"))

print(f"\nCAM AUTH: {ok}/{tot}")
sys.exit(0 if ok == tot else 1)
