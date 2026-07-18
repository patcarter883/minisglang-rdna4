"""Exercise every CAMClient method against a LIVE CAM serve — proves the SDK's endpoints, headers, and
response shapes match the real API. Zero-dep (stdlib only). Usage: BASE=http://127.0.0.1:PORT python tools/cam_client_smoke.py"""
import importlib.util
import os
import sys

# Load client.py STANDALONE (not via the minisgl package) to prove it's genuinely zero-dep — no torch,
# no package __init__ side effects. This is exactly how an external user with only stdlib would vendor it.
_p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python", "minisgl", "cam", "client.py"))
_spec = importlib.util.spec_from_file_location("cam_client", _p)
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
CAMClient, CAMError = _m.CAMClient, _m.CAMError

BASE = os.environ.get("BASE", "http://127.0.0.1:1919")
TOKEN = os.environ.get("CAM_TOKEN") or None
ok = tot = 0


def check(name, cond, extra=""):
    global ok, tot
    tot += 1; ok += bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + str(extra)) if extra else ''}")


cam = CAMClient(BASE, token=TOKEN, namespace="sdk_smoke")
cam.drop_namespace("sdk_smoke")   # clean slate

# --- write + read ---------------------------------------------------------------------------------
r = cam.remember("Wolfgang Amadeus Mozart", "Salzburg", prompt="birthplace", relation="birthplace")
check("remember returns stored/mode", r.get("stored") is True and r.get("mode_served") == "pointer", r)
cam.remember("Wolfgang Amadeus Mozart", "1756", prompt="birth year", relation="birth year")

lk = cam.lookup("the composer Mozart", relation="where born")   # paraphrase subject + relation
check("lookup (paraphrase) delivers birthplace", lk.get("delivered") and lk.get("object") == "Salzburg", lk)
lk2 = cam.lookup("Mozart", relation="birth year")
check("lookup multi-fact delivers birth year", lk2.get("object") == "1756", lk2)

a = cam.ask("Where was Mozart born?", "Wolfgang Amadeus Mozart", relation="birthplace")
check("ask delivers exact object", a.get("delivered") and a.get("object") == "Salzburg", {"object": a.get("object")})

tm = cam.lookup_text("Tell me where Mozart was born.")
check("lookup_text (transparent) returns matches list", isinstance(tm, list) and any(m.get("object") == "Salzburg" for m in tm), tm)

# --- introspection --------------------------------------------------------------------------------
f = cam.facts()
check("facts lists stored subjects", isinstance(f, list) and len(f) >= 1, f)
s = cam.stats()
check("stats has index_size + health fields", "index_size" in s and "recovered_from_backup" in s, {k: s.get(k) for k in ("index_size", "dirty")})
au = cam.audit()
check("audit returns events", isinstance(au, list) and len(au) >= 1)

# --- store ops ------------------------------------------------------------------------------------
sv = cam.save()
check("save returns saved count", "saved" in sv, sv)
fr = cam.freeze()
check("freeze -> frozen true", fr.get("frozen") is True, fr)
cam.unfreeze()

# --- namespaces + scoping -------------------------------------------------------------------------
nss = cam.namespaces()
check("namespaces lists sdk_smoke", any(n.get("namespace") == "sdk_smoke" for n in nss))
other = cam.namespace("sdk_other")
other.remember("Marie Curie", "Warsaw", prompt="birthplace")
check("re-scoped client isolates namespace", not other.lookup("Wolfgang Amadeus Mozart").get("delivered"))

# --- forget + delete (multi-fact: forget by subject+relation, same as it was stored) --------------
check("forget (subject+relation) removes the fact", cam.forget("Wolfgang Amadeus Mozart", relation="birthplace") is True)
# the deleted OBJECT is gone (a birthplace query may now match the close birth-year sibling above tau —
# that's the semantic index, not a failed delete — so assert Salzburg specifically is unretrievable)
check("forgotten object (Salzburg) is gone", cam.lookup("Wolfgang Amadeus Mozart", relation="birthplace").get("object") != "Salzburg")
check("sibling fact still delivers after forget", cam.lookup("Wolfgang Amadeus Mozart", relation="birth year").get("object") == "1756")
cam.drop_namespace("sdk_smoke"); cam.drop_namespace("sdk_other")

# --- error surface: non-2xx must raise a typed CAMError with the status --------------------------
try:
    cam._req("GET", "/cam/definitely_not_a_route")   # unknown route -> 404
    check("bad route raises CAMError", False)
except CAMError as e:
    check("bad route raises CAMError(404)", e.status == 404, e.status)

print(f"\nCAM CLIENT SDK: {ok}/{tot}")
sys.exit(0 if ok == tot else 1)
