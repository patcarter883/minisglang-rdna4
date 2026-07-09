"""Standalone FastAPI app for the CAM edit-plane (Task 1 of the serving integration).

This is the *server shell* proof for the ``/cam/*`` endpoints. The endpoint LOGIC (write gate +
router-gated seed-once decode) is already proven live via ``e2e_check.py``; this wires the exact
same ``cam_router`` into a real uvicorn app so the path is exercised over HTTP (curl), with ZERO
backend/ZMQ coupling.

It mounts ONLY ``minisgl.server.cam_api.cam_router``, driven by the co-located
``minisgl.cam.get_cam_runtime()`` singleton (a frozen HF Qwen3.5-4B base + CAMMemory). The full
``api_server`` path (which wires a ZMQ backend scheduler and needs a *served* model too) is the
production model-share task; this decouples the HTTP proof from it.

Run (in the lean image, on a GPU lease):

    MINISGL_CAM_CHECKPOINT=/ckpt CAM_NATIVE_GDN=1 \
      python /engine/python/minisgl/cam/serve_app.py            # serves 0.0.0.0:1919

then, from any shell that can reach it (same container/network):

    curl -s localhost:1919/health
    curl -s localhost:1919/cam/remember -H 'content-type: application/json' \
      -d '{"subject":"Oleg Kotov","prompt":"The mother tongue of Oleg Kotov is","object":"English"}'
    curl -s localhost:1919/cam/ask -H 'content-type: application/json' \
      -d '{"prompt":"The mother tongue of Oleg Kotov is","subject":"Oleg Kotov"}'
    curl -s localhost:1919/cam/facts
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

logger = logging.getLogger("minisgl.cam.serve_app")


@asynccontextmanager
async def _lifespan(app):
    # Warm the CAM runtime AT BOOT (co-located base ~8 GB) so the first /cam/* request isn't a cold
    # multi-second model load. get_cam_runtime() is a lazy singleton; calling it here primes it.
    from minisgl.cam import get_cam_runtime

    rt = get_cam_runtime()
    if rt is None:
        logger.warning("CAM runtime NOT loaded (check MINISGL_CAM_CHECKPOINT) — /cam/* will 503.")
    else:
        logger.info("CAM runtime warm: enabled=%s tap_layer=%s device=%s",
                    getattr(rt.memory, "enabled", None), getattr(rt.memory, "tap_layer", None),
                    getattr(rt, "device", None))
    yield


def build_app():
    from fastapi import FastAPI

    from minisgl.server.cam_api import cam_router

    app = FastAPI(title="minisgl CAM edit-plane (standalone)", lifespan=_lifespan)
    app.include_router(cam_router)

    @app.get("/health")
    async def health():
        from minisgl.cam import get_cam_runtime

        rt = get_cam_runtime()
        loaded = rt is not None and bool(getattr(rt.memory, "enabled", False))
        return {"cam_loaded": loaded}

    return app


app = build_app()


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    port = int(os.environ.get("CAM_PORT", "1919"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
