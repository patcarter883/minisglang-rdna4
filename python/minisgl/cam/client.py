"""Zero-dependency Python client for the CAM (Canonical Associative Memory) HTTP API.

Talks to a running minisgl CAM serve (`/cam/*`). Stdlib only (urllib) so it can be vendored or pip-
installed with no extra deps. Every call carries the Bearer token (if the serve sets
MINISGL_CAM_API_TOKEN) and the X-CAM-Namespace header, so one client instance is scoped to one
namespace; use `.namespace(other)` for a cheap re-scoped copy.

    from minisgl.cam.client import CAMClient
    cam = CAMClient("http://127.0.0.1:1919", token="secret", namespace="acme")
    cam.remember("Wolfgang Amadeus Mozart", "Salzburg", relation="birthplace")
    cam.ask("Where was Mozart born?", "Wolfgang Amadeus Mozart", relation="birthplace")["object"]  # 'Salzburg'
    cam.lookup("the composer Mozart", relation="where born")   # dry-run: {'delivered': True, 'object': 'Salzburg'}
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional


class CAMError(RuntimeError):
    """A CAM API call returned a non-2xx status. `.status` is the HTTP code, `.detail` the server message."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"CAM API {status}: {detail}")
        self.status = status
        self.detail = detail


class CAMClient:
    def __init__(self, base_url: str, token: Optional[str] = None, namespace: Optional[str] = None,
                 timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.ns = namespace
        self.timeout = timeout

    # ---- scoping -------------------------------------------------------------------------------------
    def namespace(self, ns: Optional[str]) -> "CAMClient":
        """A copy of this client scoped to a different namespace (same url/token/timeout)."""
        return CAMClient(self.base_url, self.token, ns, self.timeout)

    # ---- transport -----------------------------------------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        h = {"content-type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        if self.ns:
            h["X-CAM-Namespace"] = self.ns
        return h

    def _req(self, method: str, path: str, *, params: Dict[str, Any] = None,
             body: Dict[str, Any] = None) -> Any:
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read().decode()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            try:
                detail = json.loads(detail).get("detail", detail)
            except Exception:  # noqa: BLE001
                pass
            raise CAMError(e.code, detail) from None

    # ---- write ---------------------------------------------------------------------------------------
    def remember(self, subject: str, object: str, prompt: str = "", relation: str = None) -> dict:
        """Store subject->object (optionally under a `relation` for multi-fact). Returns
        {stored, base_p, mode_served, gate_reason}. `prompt` is the fact's relation prompt (used by the
        base-uncertainty write gate); it is not needed for delivery."""
        return self._req("POST", "/cam/remember",
                         body={"subject": subject, "object": object, "prompt": prompt, "relation": relation})

    def forget(self, subject: str, relation: str = None) -> bool:
        """Tombstone-delete a stored fact. Pass the same `relation` you stored it under for a multi-fact
        entity (the address is composed identically to remember/ask). Returns whether something was removed."""
        addr = f"{subject} {relation}".strip() if relation else subject
        return bool(self._req("DELETE", "/cam/facts/" + urllib.parse.quote(addr, safe="")).get("deleted"))

    # ---- read ----------------------------------------------------------------------------------------
    def ask(self, prompt: str, subject: str, relation: str = None, max_tokens: int = 32) -> dict:
        """Answer `prompt` with the stored fact for `subject` (optionally `relation`): the serve forces the
        exact stored object tokens, then the base continues. Returns {text, delivered, object, mode_served}
        — `object` is the exact delivered object (empty when nothing matched)."""
        return self._req("POST", "/cam/ask",
                         body={"prompt": prompt, "subject": subject, "relation": relation,
                               "max_tokens": max_tokens})

    def lookup(self, subject: str, relation: str = None) -> dict:
        """Dry-run: what /cam/ask WOULD deliver for `subject` (+`relation`), no generation, no mutation.
        Returns {delivered, subject, object}."""
        return self._req("GET", "/cam/lookup", params={"subject": subject, "relation": relation})

    def lookup_text(self, text: str) -> List[dict]:
        """Transparent-read dry-run: the stored facts `text` mentions (auto-RAG). Returns [{subject, object}]."""
        return (self._req("GET", "/cam/lookup", params={"text": text}) or {}).get("matches", [])

    def facts(self) -> List[dict]:
        """List stored facts in this namespace: [{subject, object}]."""
        return self._req("GET", "/cam/facts") or []

    def stats(self) -> dict:
        """Store health for this namespace (fact count, index crowding, persistence signals, ...)."""
        return self._req("GET", "/cam/stats") or {}

    def audit(self) -> List[dict]:
        """Recent write/forget/merge/evict events for this namespace (most recent last)."""
        return self._req("GET", "/cam/audit") or []

    # ---- store ops ----------------------------------------------------------------------------------
    def save(self) -> dict:
        """Force a durable snapshot now. Returns {saved: <#edits>}."""
        return self._req("POST", "/cam/save") or {}

    def reload(self) -> dict:
        """Re-read the store from disk (pick up another replica's writes). Returns {edits}."""
        return self._req("POST", "/cam/reload") or {}

    def undo(self) -> dict:
        """Undo the last write in this namespace."""
        return self._req("POST", "/cam/undo") or {}

    def freeze(self, frozen: bool = True) -> dict:
        """Freeze (or unfreeze) this namespace: ambient auto-write is refused; explicit remember still writes."""
        return self._req("POST", "/cam/freeze", params={"frozen": str(frozen).lower()}) or {}

    def unfreeze(self) -> dict:
        return self.freeze(False)

    # ---- namespaces ---------------------------------------------------------------------------------
    def namespaces(self) -> List[dict]:
        """Every namespace with its fact count + freeze state."""
        return self._req("GET", "/cam/namespaces") or []

    def drop_namespace(self, ns: str) -> dict:
        """Delete a namespace's entire store."""
        return self._req("DELETE", "/cam/namespaces/" + urllib.parse.quote(ns, safe="")) or {}
