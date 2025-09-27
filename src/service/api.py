# src/service/api.py
from __future__ import annotations
from collections import deque
from contextlib import asynccontextmanager
import glob
import inspect

import io
import math
import os
import secrets
import tempfile
import threading
import time

import aiofiles
from fastapi.exceptions import RequestValidationError
import numpy as np

from .middleware.session import SessionMiddleware
from .runs import router as runs_router

from ..signal.detrend import fit_baseline_ransac
from ..signal.peaks import detect_peaks
from ..signal.period import estimate_dominant_frequency, frequency_to_period
from ..visualize import _fit_global_sine

from ..utils import load_config
os.environ.setdefault("MPLBACKEND", "Agg")

import re
import asyncio
import json
import shutil
import uuid
import hashlib
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import Depends, FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from sse_starlette.sse import EventSourceResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.staticfiles import StaticFiles
# from starlette.middleware.sessions import SessionMiddleware

import logging
log = logging.getLogger("isolation")

from google.cloud import storage
from datetime import timedelta
from fastapi import Body

from .storage import get_storage
from .storage.base import safe_join, content_type_from_name

# Pipeline generator (emits JobEvent)
from .pipeline import _sample_name_from_arr_path, iter_run_project, JobEvent


# --- DEBUG UTILS ---
import os, time, logging
try:
    import socket
    _HOSTNAME = socket.gethostname()
except Exception:  # pragma: no cover
    _HOSTNAME = os.getenv("HOSTNAME", "")

log_dbg = logging.getLogger("api_dbg")

def _dbg_headers(request, *, run_id: str | None = None, served_from: str | None = None, extra: dict | None = None):
    sid = ""
    try:
        sid = (_sid_from_request(request) or "")[:8]
    except Exception:
        pass
    hdrs = {
        "X-Revision": os.getenv("K_REVISION", ""),
        "X-Service": os.getenv("K_SERVICE", ""),
        "X-Hostname": _HOSTNAME,
        "X-Storage": STORAGE_KIND,
        "X-SID": sid,
    }
    if run_id:
        hdrs["X-Run-Id"] = run_id
    if served_from:
        hdrs["X-Served-From"] = served_from
    if extra:
        hdrs.update({k: str(v) for k, v in extra.items()})
    return hdrs

def _ms(t0: float) -> int:
    return int((time.time() - t0) * 1000)

# Set SSE_PING_SECONDS=0 in dev to disable heartbeats (reduces socket.send noise).
SSE_PING_SECONDS = int(os.getenv("SSE_PING_SECONDS", "25"))  # 0 → disables pings
SSE_LOG_SUBS = os.getenv("SSE_LOG_SUBS", "0") == "1"         # 1 → log +sub/-sub

logger = logging.getLogger("wavecalling.sse")

SESSION_SECRET = os.getenv("SESSION_SECRET")
if not SESSION_SECRET:
    raise RuntimeError("SESSION_SECRET is required for per-browser isolation")

# ---- Config ----
APP_ROOT = Path(__file__).resolve().parents[2]  # repo root
DEFAULT_CONFIG = Path(os.getenv("DEFAULT_CONFIG", "/app/configs/default.yaml")).resolve()

if not DEFAULT_CONFIG.exists():
    log.error("DEFAULT_CONFIG missing at %s", DEFAULT_CONFIG)

WEB_DIR = Path(os.getenv("WEB_DIR", "/app/web")).resolve()

"""
RUNS_ROOT = APP_ROOT / "runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
"""

RUNS_DIR = Path(os.getenv("RUNS_DIR", "/tmp/runs")).resolve()
RUNS_DIR.mkdir(parents=True, exist_ok=True)

STAGING_ROOT = Path(os.getenv("STAGING_ROOT", "/app/staging"))
STAGING_ROOT.mkdir(parents=True, exist_ok=True)

GCS_BUCKET = os.getenv("GCS_BUCKET")

STORAGE_KIND = os.getenv("STORAGE", "local").lower()
runs_storage = get_storage()

# SID_COOKIE = os.getenv("SESSION_COOKIE", "sid")
# SESSION_SAMESITE = os.getenv("SESSION_SAMESITE", "Lax")  # "Lax" if same-origin, "None" if cross-site
# SESSION_HTTPS_ONLY = os.getenv("SESSION_HTTPS_ONLY", "true").lower() == "true"

# COOKIE_NAME = SID_COOKIE

# _SAMESITE = {"None": "none", "Lax": "lax", "Strict": "strict"}
# COOKIE_SAMESITE = _SAMESITE.get(os.getenv("SESSION_SAMESITE", "Lax"), "lax")
# COOKIE_SECURE = os.getenv("SESSION_HTTPS_ONLY", "true").lower() == "true"
# Cookie config via env (works on Cloud Run)
COOKIE_NAME     = os.getenv("SESSION_COOKIE", "sid")
COOKIE_SAMESITE = (os.getenv("SESSION_SAMESITE", "lax") or "lax").lower()  # "none"|"lax"|"strict"
COOKIE_SECURE   = os.getenv("SESSION_HTTPS_ONLY", "true").lower() == "true"
COOKIE_DOMAIN   = os.getenv("SESSION_DOMAIN") or None                      # e.g. "waves.rohitmahesh.net"
COOKIE_PATH     = "/"
COOKIE_MAX_AGE  = int(os.getenv("SESSION_TTL_SECS", str(14 * 24 * 60 * 60)))  # 14 days

_RUN_ID_RE = re.compile(r"/(?:api/)?(?:runs|files)/(?P<rid>[A-Za-z0-9_-]+)")

def _issue_sid() -> str:
    return secrets.token_urlsafe(32)

def get_sid(request: Request, response: Response = None) -> str:
    """
    Return the browser SID. Does not set cookies; a middleware will attach it.
    """
    sid = getattr(request.state, "sid", None) or request.cookies.get(COOKIE_NAME)
    if sid:
        return sid
    sid = _issue_sid()
    request.state.sid = sid
    return sid

RUN_ID_SEG = r"(?P<run_id>[A-Za-z0-9._-]{6,64})"
_RUNS_RE  = re.compile(rf"^/api/runs/{RUN_ID_SEG}(?:/|$)", re.IGNORECASE)
_FILES_RE = re.compile(r"^/(files|runs)/(?P<run_id>[A-Za-z0-9._-]{6,64})")

r"""
_RUNS_RE  = re.compile(r"^/api/runs/(?P<run_id>[-A-Za-z0-9]+)$")
_FILES_RE = re.compile(r"^/runs/(?P<run_id>[-A-Za-z0-9]+)$")
"""
# -----------------------
# Persistence helpers
# -----------------------

def run_key(run_id: str, *parts: str) -> str:
    """
    Build the object key relative to the storage root.
    IMPORTANT:
      - If LocalStorage.root == RUNS_DIR ( FUSE mount), DO NOT prefix with "runs".
        Use "<run_id>/..." keys so files land under /app/runs/<run_id>/...
      - If STORAGE=gcs with a flat bucket, optionally
        set RUNS_KEY_PREFIX="runs" and we’ll store as "runs/<run_id>/...".
    """
    prefix = os.getenv("RUNS_KEY_PREFIX", "").strip("/")
    bits = ([prefix] if prefix else []) + [run_id, *parts]
    return safe_join(*bits)

def _sid_from_request(request: Request) -> str:
    return getattr(request.state, "sid", None) or get_sid(request)

def _stage_dir_for(request: Request) -> Path:
    sid = _sid_from_request(request)
    d = STAGING_ROOT / sid
    d.mkdir(parents=True, exist_ok=True)
    return d

def _list_staged(d: Path) -> list[dict]:
    out = []
    for p in sorted(d.glob("*")):
        if p.is_file():
            try:
                out.append({"name": p.name, "size": p.stat().st_size})
            except FileNotFoundError:
                pass
    return out

def _try_read_json_bytes(p: Path) -> bytes | None:
    """
    Read a JSON file safely. If the file is missing, empty, or mid-write
    (JSONDecodeError), return None so the caller can emit 204.
    """
    try:
        with open(p, "rb") as f:
            raw = f.read()
        raw = raw.strip()
        if not raw:
            return None
        obj = json.loads(raw)  # validate; will raise on partial writes
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _try_read_ndjson_as_overlay_bytes(p: Path, *, version: int = 0) -> bytes | None:
    """
    Read NDJSON safely and wrap as {"version": N, "tracks":[...]}.
    Skips truncated/malformed lines; if nothing usable, return None.
    """
    try:
        with open(p, "rb") as f:
            b = f.read()
    except FileNotFoundError:
        return None
    except OSError:
        return None

    lines = b.splitlines()
    tracks = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            tracks.append(json.loads(line))
        except Exception:
            # ignore partial trailing line / junk
            continue

    if not tracks:
        return None

    payload = {"version": int(version) if version is not None else 0, "tracks": tracks}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

def _run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id

def _events_path(run_id: str) -> Path:
    return _run_dir(run_id) / "events.ndjson"

def _run_json_path(run_id: str) -> Path:
    return _run_dir(run_id) / "run.json"

def _run_json_path(run_id: str) -> Path:
    """Persisted run metadata (used for restore)."""
    return _run_dir(run_id) / "run.json"

def _output_dir(run_id: str) -> Path:
    """Pipeline outputs root."""
    return _run_dir(run_id) / "output"

def _overlay_json_path(run_id: str) -> Path:
    """Final overlay (single JSON file)."""
    return _output_dir(run_id) / "overlay" / "tracks.json"

def _overlay_ndjson_path(run_id: str) -> Path:
    """Streaming overlay (NDJSON accumulating during run)."""
    return _output_dir(run_id) / "overlay" / "tracks.ndjson"

def _progress_json_path(run_id: str) -> Path:
    """Live progress snapshot JSON."""
    return _output_dir(run_id) / "progress.json"

def _run_json_candidates(run_id: str) -> list[Path]:
    """
    Try several plausible locations to be resilient to legacy/alternate layouts.
    """
    return [
        RUNS_DIR / run_id / "run.json",                 # current expected
        RUNS_DIR / "runs" / run_id / "run.json",        # nested "runs" folder
        Path("/app/runs") / run_id / "run.json",        # legacy image-local
    ]

def _write_json(path: Path, payload: dict) -> None:
    """Atomically write JSON to `path` using a unique temp file.

    Avoids races where multiple writers share a deterministic *.tmp name.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    # unique tmp in the same dir to keep os.replace atomic on the same filesystem
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)  # atomic on POSIX
    finally:
        # if replace failed, try to remove the temp file
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except OSError:
            pass

def _artifact_url(run_id: str, rel: str) -> str:
    # all front-end artifact links go through this gateway:
    return f"/files/{run_id}/{rel}"

# ---- ETag helpers ----
def _sha1_etag_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def _weak_file_etag(p: Path) -> str:
    st = p.stat()
    # weak etag from mtime+size (fast; stable across restarts)
    return f'W/"{int(st.st_mtime_ns)}-{st.st_size}"'

def _sanitize_for_json(obj):
    """Recursively replace NaN/Inf with None; convert numpy scalars to python."""
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(x) for x in obj]
    return obj

def _append_event(run_id: str, evt: JobEvent) -> None:
    """Append an event to NDJSON log (best-effort; small writes)."""
    p = _events_path(run_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(evt)) + "\n")

def _load_run_json(path: Path) -> Optional[dict]:
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return None

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- startup: load persisted runs from disk ----
    for d in RUNS_DIR.iterdir():
        if not d.is_dir():
            continue
        rj = _run_json_path(d.name)
        meta = _load_run_json(rj) or {}
        if not meta:
            continue
        run_id = meta.get("run_id") or d.name
        name = meta.get("name") or f"run-{run_id}"
        st = _RunState(
            run_id=run_id,
            name=name,
            base_dir=d,
            config_path=Path(meta.get("config_path", d / "config.yaml")),
            created_at_iso=meta.get("created_at"),
            status=meta.get("status", "DONE"),
            error=meta.get("error"),
        )
        _RUNS[run_id] = st

    # hand over control to the app
    yield

    # ---- shutdown cleanup (safe no-op if unused) ----
    for st in list(_RUNS.values()):
        if st.status in ("QUEUED", "RUNNING"):
            st.cancel_event.set()
            if st.worker_task:
                try:
                    await st.worker_task
                except Exception:
                    pass


class SPAStaticFiles(StaticFiles):
    """
    Serve files from WEB_DIR; if a path isn't found and it looks like a client-route
    (no dot in the last path segment), fall back to index.html so React Router can handle it.
    """
    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            # Only rewrite 404s that are likely client routes (/ui/something without a file extension)
            if exc.status_code == 404 and "." not in path.rsplit("/", 1)[-1]:
                return await super().get_response("index.html", scope)
            raise


# ---- FastAPI app ----
app = FastAPI(title="WaveCalling API", version="0.2.0", lifespan=lifespan)

# Expose the runs directory for downloading artifacts (CSV/plots/overlay)
SAMPLES_DIR = Path(os.getenv("SAMPLES_DIR", "/app/samples"))
if STORAGE_KIND == "local":
    if not any(SAMPLES_DIR.glob("*")):
        logger.warning("No sample files in %s. Run ./fetch_samples.sh to download examples.", SAMPLES_DIR)
    app.mount("/files", StaticFiles(directory=str(RUNS_DIR), html=False), name="files")
    logger.info("Mounted /files from %s (local mode)", RUNS_DIR)
else:
    logger.info("Skipping /files static mount (remote-storage mode)")

# --- Serve sample files (read-only) at /samples/<name>.csv ---
if SAMPLES_DIR.is_dir():
    if not any(SAMPLES_DIR.glob("*")):
        logger.warning("No sample files in %s. Run ./fetch_samples.sh to download examples.", SAMPLES_DIR)
    app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR), html=False), name="samples")
    logger.info("Mounted sample files from %s at /samples", SAMPLES_DIR)
else:
    logger.warning("SAMPLES_DIR not found: %s (sample buttons will 404)", SAMPLES_DIR)

app.mount("/ui", SPAStaticFiles(directory=str(WEB_DIR), html=True), name="web")

"""
SESSION_HTTPS_ONLY = os.getenv("SESSION_HTTPS_ONLY", "false").lower() in ("1","true","yes")
SESSION_SAMESITE   = os.getenv("SESSION_SAMESITE", "lax")
"""
"""
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    same_site=SESSION_SAMESITE,
    https_only=SESSION_HTTPS_ONLY,
    max_age=60 * 60 * 24 * 30,
)"""
"""
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    same_site="lax",   # sends cookie on top-level navigations
    https_only=True,   # cookie only over HTTPS
    max_age=60 * 60 * 24 * 30,  # 30 days
)
"""

app.include_router(runs_router)

allow_origins = os.getenv("ALLOW_ORIGINS", "").split(",") if os.getenv("ALLOW_ORIGINS") else []
if allow_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,           # required for cookies
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.middleware("http")
async def add_debug_headers(request, call_next):
    import os, socket
    resp = await call_next(request)
    resp.headers.setdefault("X-Revision", os.getenv("K_REVISION", ""))
    resp.headers.setdefault("X-Service", os.getenv("K_SERVICE", ""))
    try:
        resp.headers.setdefault("X-Hostname", socket.gethostname())
    except Exception:
        pass
    return resp

@app.middleware("http")
async def error_wrapper(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        # surface the message to the client while debugging
        return JSONResponse(status_code=500, content={"detail": str(e)})

async def _load_run_state(run_id: str) -> Optional[_RunState]:
    st = _RUNS.get(run_id)
    if st is not None:
        return st

    async with _RUNS_LOCK:
        st = _RUNS.get(run_id)
        if st is not None:
            return st

        # ---- try local run.json ----
        meta = None
        try:
            p = _run_json_path(run_id)
            with open(p, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            meta = None
        except Exception:
            logging.exception("reading local run.json failed for %s", run_id)
            meta = None

        # ---- fallback to storage ----
        if meta is None:
            try:
                meta = await _load_run_meta_from_storage(run_id)
            except Exception:
                logging.exception("restore_from_storage failed for %s", run_id)
                meta = None

        if not meta:
            return None

        # ---- construct state ----
        try:
            if hasattr(_RunState, "from_meta"):
                st = _RunState.from_meta(meta)
            else:
                st = _RunState.from_run_json(meta)  # supports dict-based variant
        except TypeError:
            # some codebases have from_run_json(run_id) instead
            st = _RunState.from_run_json(run_id)
        except Exception:
            logging.exception("constructing _RunState failed for %s", run_id)
            return None

        try:
            st.maybe_bump_overlay_version()
        except Exception:
            pass

        _RUNS[run_id] = st
        return st

def _set_sid_cookie(resp: Response, sid: str) -> None:
    # Always set (success and error) so the browser keeps a stable SID, with sliding TTL.
    resp.set_cookie(
        key=COOKIE_NAME,
        value=sid,
        httponly=True,
        secure=COOKIE_SECURE,       # Required if SameSite=None
        samesite=COOKIE_SAMESITE,   # "none"|"lax"|"strict"
        domain=COOKIE_DOMAIN,       # e.g. "waves.rohitmahesh.net"
        path=COOKIE_PATH,
        max_age=COOKIE_MAX_AGE,
    )

def _extract_run_id(path: str) -> str | None:
    m = _RUN_ID_RE.search(path)
    return m.group("rid") if m else None

@app.middleware("http")
async def isolation_guard(request: Request, call_next):
    minted = False
    sid = request.cookies.get(COOKIE_NAME)
    if not sid:
        sid = _issue_sid()
        minted = True
    request.state.sid = sid

    if request.method == "OPTIONS":
        resp = await call_next(request)
        if minted:
            kw = {}
            if "COOKIE_DOMAIN" in globals() and COOKIE_DOMAIN:
                kw["domain"] = COOKIE_DOMAIN
            resp.set_cookie(
                key=COOKIE_NAME, value=sid, httponly=True,
                secure=COOKIE_SECURE, samesite=COOKIE_SAMESITE, path=COOKIE_PATH, **kw
            )
        return resp

    try:
        path = request.url.path
        m = _RUNS_RE.match(path) or _FILES_RE.match(path)
        if m:
            run_id = m.group("run_id")
            st = await _load_run_state(run_id)

            if st is not None:
                _assert_can_view(st, sid)
            # else: meta not ready or path mismatch → allow route to decide (will 204)
        resp = await call_next(request)

    except HTTPException as e:
        resp = JSONResponse({"detail": e.detail}, status_code=e.status_code)

    if minted:
        kw = {}
        if "COOKIE_DOMAIN" in globals() and COOKIE_DOMAIN:
            kw["domain"] = COOKIE_DOMAIN
        resp.set_cookie(
            key=COOKIE_NAME, value=sid, httponly=True,
            secure=COOKIE_SECURE, samesite=COOKIE_SAMESITE, path=COOKIE_PATH, **kw
        )
    return resp

async def ensure_sid(request: Request) -> str:
    # centralize session creation/lookup; set cookie if needed
    return get_sid(request)

async def require_run_view(
    request: Request,
    run_id: str,                          # path param is available to dependencies
    sid: str = Depends(ensure_sid),
):
    st = await _load_run_state(run_id)               # ✅ always a _RunState
    if not st:
        raise HTTPException(404, "Run not found")
    _assert_can_view(st, sid)                        # ✅ pass the SID
    request.state.run = st

async def _persist_run_meta(st: _RunState) -> None:
    """
    Best-effort persistence of run metadata:
      1) write local run.json (atomic; handled by st._persist_run_json)
      2) mirror a compact-but-complete meta to canonical storage (run.json)

    Serialized per-run via st._meta_lock so concurrent events don't race.
    """
    import logging, json
    log = logging.getLogger(__name__)

    # Serialize per-run to prevent concurrent writers
    async with st._meta_lock:
        # 1) Local write (uses atomic _write_json inside _persist_run_json)
        try:
            await asyncio.to_thread(st._persist_run_json)
        except FileNotFoundError:
            # Run directory removed between events; nothing to persist.
            return
        except Exception:
            log.exception("persist_run_json failed", extra={"run_id": st.run_id})

        # 2) Canonical remote write (include dirs/paths for robust restore)
        try:
            meta = st.to_meta()
            meta.update({
                "input_dir": str(st.input_dir),
                "output_dir": str(st.output_dir),
                "plots_dir": str(st.plots_dir),
                "config_path": str(st.config_path),
            })
            key = run_key(st.run_id, "run.json")
            await runs_storage.put(
                key,
                json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
                content_type="application/json",
            )
        except Exception:
            # Do not break the pipeline on telemetry/mirroring problems
            log.exception("persist_run_meta remote put failed", extra={"run_id": st.run_id})
    
async def _load_run_meta_from_storage(run_id: str) -> dict | None:
    """
    Fetch run.json from canonical store (e.g., GCS).
    Tries both "<id>/run.json" and "runs/<id>/run.json" to be resilient.
    """
    keys = [
        run_key(run_id, "run.json"),               # respects RUNS_KEY_PREFIX (e.g., "" or "runs")
        safe_join("runs", run_id, "run.json"),     # compatibility path
    ]
    for key in keys:
        try:
            b = await runs_storage.get(key)  # returns bytes or None
            if b:
                return json.loads(b.decode("utf-8"))
        except Exception:
            continue
    return None

def _read_local_meta(run_id: str) -> dict | None:
    for p in _run_json_candidates(run_id):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            continue
        except json.JSONDecodeError:
            # mid-write; try other candidates / fallback to storage
            continue
        except Exception:
            logging.exception("reading %s failed", p)
            continue
    return None

async def _restore_from_meta(run_id: str) -> dict | None:
    """
    Keep existing implementation.
    This wrapper just calls it whether it's async or sync.
    """
    # If file defines a concrete `_restore_from_meta`, we assume it exists.
    # Otherwise implement the storage fetch here (e.g., GCS read) and return dict or None.
    try:
        fn = globals().get("_restore_from_meta_impl") or globals().get("_restore_from_meta")
        if fn is None:
            return None
        if inspect.iscoroutinefunction(fn):
            return await fn(run_id)
        # sync function
        return await asyncio.to_thread(fn, run_id)
    except Exception:
        logging.exception("restore_from_meta failed for %s", run_id)
        return None
# -----------------------
# In-memory run registry
# -----------------------

class RunInfo(BaseModel):
    run_id: str
    name: str
    created_at: str
    status: Literal["QUEUED","RUNNING","DONE","ERROR","CANCELLED"]
    error: Optional[str] = None
    input_dir: str
    output_dir: str
    plots_dir: str
    config_path: str
    owner_sid: Optional[str] = Field(default=None)
    is_public: bool = False

    class Config:
        extra = "ignore"  # tolerate future keys


class _RunState:
    """
    Internal state for each run: dirs, subscribers, status, and persistence.
    Persists run.json and events.ndjson so UI can list/replay after refresh.
    """

    # ------------ construction / restore ------------

    def __init__(
        self,
        run_id: str,
        name: str,
        base_dir: Path,
        config_path: Path,
        created_at_iso: Optional[str] = None,
        status: str = "QUEUED",
        error: Optional[str] = None,
        *,
        owner_sid: Optional[str] = None,
        is_public: bool = False,
    ):
        self.run_id = run_id
        self.name = name
        self.base_dir = Path(base_dir)
        self.input_dir = self.base_dir / "input"
        self.output_dir = self.base_dir / "output"
        self.plots_dir = self.base_dir / "plots"
        self.config_path = Path(config_path)

        # Ensure directories exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.created_at = (
            datetime.fromisoformat(created_at_iso.replace("Z", "+00:00"))
            if created_at_iso else datetime.now(timezone.utc)
        )
        self.status = status
        self.error: Optional[str] = error

        # Ownership / sharing
        self.owner_sid: Optional[str] = owner_sid
        self.is_public: bool = bool(is_public)

        # Worker / cancellation
        self.cancel_event: asyncio.Event = asyncio.Event()
        self.worker_task: Optional[asyncio.Task] = None

        # Live subscribers (SSE/websocket queues)
        self._subscribers: List[asyncio.Queue] = []
        self._sub_lock = asyncio.Lock()

        # Overlay versioning (bump when overlay files change)
        self.overlay_version: int = 0
        self._overlay_mtime_ns: int = 0

        # --- NEW: meta write dedupe/serialization ---
        self._meta_task: Optional[asyncio.Task] = None
        self._meta_lock: asyncio.Lock = asyncio.Lock()

        # --- NEW: throttles for remote mirroring ---
        self._last_artifacts_flush_ns: int = 0
        self._last_events_flush_ns: int = 0

        # Worker / cancellation  ✅ use threading.Event so worker thread can see it
        self.cancel_event: threading.Event = threading.Event()
        self.worker_task: Optional[asyncio.Task] = None

        self.ref = f"{id(self):x}"

        # Persist initial snapshot (local)
        self._persist_run_json()
    
    def cancel(self) -> None:
            """Request cooperative cancellation."""
            self.cancel_event.set()

    def reset_cancel(self) -> None:
        """Clear a previous cancellation request (for resume/retry)."""
        self.cancel_event.clear()

    def is_cancel_requested(self) -> bool:
        return self.cancel_event.is_set()

    async def wait_cancelled(self, timeout: float | None = None) -> bool:
        """Async helper to await cancel in coroutine code while using threading.Event under the hood."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.cancel_event.wait, timeout)

    @classmethod
    def from_run_json(cls, run_id: str) -> Optional["_RunState"]:
        p = _run_json_path(run_id)
        try:
            payload = _load_run_json(p)
        except FileNotFoundError:
            return None
        except Exception:
            return None

        output_dir = Path(payload.get("output_dir", ""))
        base_dir = output_dir.parent if output_dir.name == "output" else p.parent

        return cls(
            run_id=payload["run_id"],
            name=payload.get("name") or payload["run_id"],
            base_dir=base_dir,
            config_path=Path(payload.get("config_path") or (base_dir / "config.yaml")),
            created_at_iso=payload.get("created_at"),
            status=payload.get("status", "QUEUED"),
            error=payload.get("error"),
            owner_sid=payload.get("owner_sid"),
            is_public=bool(payload.get("is_public", False)),
        )

    # ------------ properties / meta ------------

    @property
    def created_at_iso(self) -> str:
        return self.created_at.isoformat().replace("+00:00", "Z")

    def to_meta(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "name": self.name,
            "created_at": self.created_at_iso,
            "status": self.status,
            "error": self.error,
            "owner_sid": self.owner_sid,
            "is_public": self.is_public,
            "allowed_sids": sorted(list(getattr(self, "allowed_sids", []) or [])),
        }

    @classmethod
    def from_meta(cls, meta: dict) -> "_RunState":
        runs_root = Path(os.getenv("RUNS_DIR", "/app/runs"))
        if meta.get("output_dir"):
            base_dir = Path(meta["output_dir"]).parent
        elif meta.get("input_dir"):
            base_dir = Path(meta["input_dir"]).parent
        else:
            base_dir = runs_root / meta["run_id"]

        st = cls(
            run_id=meta["run_id"],
            name=meta.get("name") or meta["run_id"],
            base_dir=base_dir,
            config_path=Path(meta.get("config_path") or (base_dir / "config.yaml")),
            created_at_iso=meta.get("created_at"),
            status=meta.get("status", "QUEUED"),
            error=meta.get("error"),
            owner_sid=meta.get("owner_sid"),
            is_public=bool(meta.get("is_public", False)),
        )
        try:
            st.allowed_sids = set(meta.get("allowed_sids") or [])
        except Exception:
            pass
        return st

    def info(self) -> RunInfo:
        return RunInfo(
            run_id=self.run_id,
            name=self.name,
            created_at=self.created_at_iso,
            status=self.status,
            error=self.error,
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir),
            plots_dir=str(self.plots_dir),
            config_path=str(self.config_path),
            owner_sid=self.owner_sid,
            is_public=self.is_public,
        )

    # ------------ subscribers / events ------------

    async def add_subscriber(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        async with self._sub_lock:
            self._subscribers.append(q)
        return q

    async def remove_subscriber(self, q: asyncio.Queue) -> None:
        async with self._sub_lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def subscribers_count(self) -> int:
        return len(self._subscribers)

    def _persist_run_json(self) -> None:
        payload = {
            "run_id": self.run_id,
            "name": self.name,
            "created_at": self.created_at_iso,
            "status": self.status,
            "error": self.error,
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "plots_dir": str(self.plots_dir),
            "config_path": str(self.config_path),
            "owner_sid": self.owner_sid,
            "is_public": self.is_public,
        }
        _write_json(_run_json_path(self.run_id), payload)

    def _persist_event(self, evt: JobEvent) -> None:
        try:
            _append_event(self.run_id, evt)
        except Exception:
            # Do not block on telemetry persistence
            pass

    # --- NEW: deduped scheduling of meta persistence ---
    def _schedule_persist_meta(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        t = self._meta_task
        if not t or t.done():
            # one task at a time per run
            self._meta_task = loop.create_task(_persist_run_meta(self))

    async def publish(self, evt: JobEvent) -> None:
        # 1) persist to local ndjson
        self._persist_event(evt)

        # 2) fan-out to live subscribers (non-blocking)
        try:
            async with self._sub_lock:
                subscribers = list(self._subscribers)
            for q in subscribers:
                try:
                    q.put_nowait(evt)
                except asyncio.QueueFull:
                    pass
        except RuntimeError:
            # event loop not running or shutting down — ignore
            pass

        # 3) mirror events + artifacts to remote storage periodically (when enabled)
        if STORAGE_KIND != "local":
            now = time.time_ns()

            # 3a) events.ndjson (helps cross-instance SSE)
            if (evt.phase in ("DONE", "ERROR", "CANCELLED")) or (now - self._last_events_flush_ns > 2_000_000_000):
                self._last_events_flush_ns = now
                try:
                    data = (_run_dir(self.run_id) / "events.ndjson").read_bytes()
                    asyncio.get_running_loop().create_task(
                        runs_storage.put(
                            run_key(self.run_id, "events.ndjson"),
                            data,
                            content_type="application/x-ndjson",
                        )
                    )
                except Exception:
                    pass

            # 3b) overlay/progress/base
            if (evt.phase in ("DONE", "ERROR", "CANCELLED")) or (now - self._last_artifacts_flush_ns > 2_000_000_000):
                self._last_artifacts_flush_ns = now
                try:
                    # overlay (JSON first, else NDJSON)
                    j = _overlay_json_path(self.run_id)
                    if j.exists():
                        data = j.read_bytes()
                        asyncio.get_running_loop().create_task(
                            runs_storage.put(
                                run_key(self.run_id, "output", "overlay", "tracks.json"),
                                data,
                                content_type="application/json",
                            )
                        )
                    nd = _overlay_ndjson_path(self.run_id)
                    if nd.exists():
                        data = nd.read_bytes()
                        asyncio.get_running_loop().create_task(
                            runs_storage.put(
                                run_key(self.run_id, "output", "overlay", "tracks.ndjson"),
                                data,
                                content_type="application/x-ndjson",
                            )
                        )

                    # progress
                    prog_path, _ = _resolve_progress_paths_for_state(self)
                    if prog_path.exists():
                        data = prog_path.read_bytes()
                        asyncio.get_running_loop().create_task(
                            runs_storage.put(
                                run_key(self.run_id, "output", "progress.json"),
                                data,
                                content_type="application/json",
                            )
                        )

                    base = _run_dir(self.run_id) / "output" / "base.png"
                    if base.exists():
                        data = base.read_bytes()
                        asyncio.get_running_loop().create_task(
                            runs_storage.put(
                                run_key(self.run_id, "output", "base.png"),
                                data,
                                content_type="image/png",
                            )
                        )
                except Exception:
                    pass

        # 4) persist run meta + notify global runs bus (deduped)
        self._schedule_persist_meta()
        try:
            asyncio.get_running_loop().create_task(_publish_runs_dirty())
        except RuntimeError:
            pass

    # ------------ status / control ------------

    def set_status(self, status: str, error: str | None = None) -> None:
        self.status = status
        if error is not None:
            self.error = error
        # Always persist locally (atomic writer should ensure this is safe)
        self._persist_run_json()
        # schedule background meta write + runs list notification
        self._schedule_persist_meta()
        try:
            asyncio.get_running_loop().create_task(_publish_runs_dirty())
        except RuntimeError:
            pass

    def cancel(self) -> None:
        self.cancel_event.set()

    # ------------ overlay + artifacts ------------

    def _overlay_candidates(self) -> List[Path]:
        overlay_dir = self.output_dir / "overlay"
        return [
            overlay_dir / "tracks.json",
            overlay_dir / "tracks.partial.json",
            overlay_dir / "tracks.ndjson",
        ]

    def maybe_bump_overlay_version(self) -> bool:
        """
        Detect if overlay output changed (by mtime) and bump version.
        Call this after steps that may update overlay artifacts.
        """
        cand = [p for p in self._overlay_candidates() if p.exists()]
        if not cand:
            return False
        try:
            newest_ns = max(p.stat().st_mtime_ns for p in cand)
        except FileNotFoundError:
            return False
        if newest_ns > self._overlay_mtime_ns:
            self._overlay_mtime_ns = newest_ns
            self.overlay_version += 1
            return True
        return False

    def _artifacts_map(self) -> Dict[str, bool]:
        """
        Coarse artifact presence flags used by /snapshot to let UI decide what to fetch.
        Keep keys stable across releases.
        """
        overlay_dir = self.output_dir / "overlay"
        debug_dir = self.output_dir / "debug"
        return {
            "overlay_tracks_json": (overlay_dir / "tracks.json").exists(),
            "overlay_tracks_partial": (overlay_dir / "tracks.partial.json").exists(),
            "overlay_tracks_ndjson": (overlay_dir / "tracks.ndjson").exists(),
            "debug_dir": debug_dir.exists(),
            "plots_any_png": any(self.plots_dir.glob("*.png")),
        }

    def snapshot(self, progress: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compact snapshot for conditional polling (ETag on server-side recommended).
        Mirrors frontend's SnapshotResponse.
        """
        return {
            "run_id": self.run_id,
            "status": self.status,
            "error": self.error,
            "overlay_version": self.overlay_version,
            "artifacts": self._artifacts_map(),
            "progress": progress or None,
        }


# Global registry (in-memory)
_RUNS: Dict[str, _RunState] = {}
_RUNS_LOCK = asyncio.Lock()  # single-flight for concurrent load
# -----------------------
# Global "runs" pub/sub (dashboard updates)
# -----------------------
_GLOBAL_SUBS: List[asyncio.Queue] = []
_GLOBAL_LOCK = asyncio.Lock()


async def _global_add_sub() -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue()
    async with _GLOBAL_LOCK:
        _GLOBAL_SUBS.append(q)
    return q

async def _global_remove_sub(q: asyncio.Queue) -> None:
    async with _GLOBAL_LOCK:
        if q in _GLOBAL_SUBS:
            _GLOBAL_SUBS.remove(q)

async def _publish_runs_dirty() -> None:
    async with _GLOBAL_LOCK:
        for q in list(_GLOBAL_SUBS):
            try:
                q.put_nowait({"type": "runs", "dirty": {"list": True}, "ts": _iso_now()})
            except asyncio.QueueFull:
                pass


def _read_input_object_bytes(obj_key: str) -> bytes:
    """
    Returns file bytes for an object used as pipeline input.
    - If obj_key startswith "samples/", read from SAMPLES_DIR in the container.
    - Else fetch from the configured GCS bucket using the lightweight client.
    Raises HTTPException with a clear 4xx if it can't read.
    """
    # 1) "samples/" → read from image's SAMPLES_DIR
    if obj_key.startswith("samples/"):
        samples_dir = os.getenv("SAMPLES_DIR", "/app/samples")
        src = Path(samples_dir) / obj_key.split("/", 1)[1]
        if not src.exists():
            raise HTTPException(status_code=404, detail=f"sample not found: {obj_key}")
        try:
            return src.read_bytes()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"failed to read sample {obj_key}: {e}")

    # 2) everything else → read from GCS bucket
    bucket = os.getenv("GCS_BUCKET")
    if not bucket:
        raise HTTPException(status_code=400, detail="GCS_BUCKET not set (required for /runs/from_gcs)")
    try:
        from google.cloud import storage as gcs_storage
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"google-cloud-storage not available: {e}")

    try:
        client = gcs_storage.Client()
        blob = client.bucket(bucket).blob(obj_key)
        if not blob.exists():
            raise HTTPException(status_code=404, detail=f"missing object: gs://{bucket}/{obj_key}")
        return blob.download_as_bytes()
    except HTTPException:
        raise
    except Exception as e:
        # Map transport errors to a clean 502/500 with context
        raise HTTPException(status_code=502, detail=f"GCS read failed for gs://{bucket}/{obj_key}: {e}")

# -----------------------
# Models / payloads
# -----------------------
class CreateRunResponse(BaseModel):
    run_id: str
    status: Literal["QUEUED","RUNNING","DONE","ERROR","CANCELLED"]
    info: RunInfo

class RunStatusResponse(BaseModel):
    info: RunInfo
    artifacts: Dict[str, str] = Field(
        default_factory=dict,
        description="Paths (URLs) to main artifacts once available",
    )

class StartUploadResp(BaseModel):
    upload_url: str          # resumable session URL (client PUTs the file here)
    object: str              # "uploads/<uuid>/<safe_name>"


# -----------------------
# Helpers
# -----------------------

async def _write_control(run_id: str, name: str, payload: dict) -> None:
    p = safe_join("runs", run_id, "control", name)
    await runs_storage.put(p, json.dumps(payload).encode("utf-8"), "application/json")

async def _cancel_requested(run_id: str) -> bool:
    p = safe_join("runs", run_id, "control", "cancel.json")
    return await runs_storage.exists(p)

async def _save_uploads(files, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    def _safe_name(name: str) -> str:
        base = Path(name or "upload.bin").name
        return re.sub(r"[^A-Za-z0-9._-]+", "_", base)

    saved_paths: list[Path] = []
    for uf in files or []:
        fname = _safe_name(getattr(uf, "filename", "") or "upload.bin")
        out_path = out_dir / fname

        async with aiofiles.open(out_path, "wb") as w:
            while True:
                chunk = await uf.read(64 * 1024)
                if not chunk:
                    break
                await w.write(chunk)
        await uf.close()
        saved_paths.append(out_path)

    return saved_paths

def _artifact_paths_by_state(state: _RunState) -> Dict[str, Path]:
    o = state.output_dir
    base = state.base_dir
    return {
        "tracks_csv": o / "metrics.csv",
        "waves_csv": o / "metrics_waves.csv",
        "overlay_json": o / "overlay" / "tracks.json",
        "overlay_json_partial": o / "overlay" / "tracks.partial.json",
        "overlay_ndjson": o / "overlay" / "tracks.ndjson",
        "run_json": base / "run.json",
        "events_ndjson": base / "events.ndjson",
        "progress_json": o / "progress.json",
        "plots_dir": base / "plots",
        "output_dir": o,
        "base_image": o / "base.png",
    }

def _resolve_progress_paths_for_state(st: _RunState) -> Tuple[Path, List[Path]]:
    """Resolve progress.json + candidate marker dirs based on a live state's config."""
    cfg = load_config(st.config_path)
    svc = (cfg.get("service") or {})
    resume = (svc.get("resume") or {})
    progress_rel = resume.get("progress_file", "progress.json")
    marker_rel = resume.get("marker_dir", "processed")

    out = Path(st.output_dir)
    progress_path = (out / progress_rel).resolve()
    marker_dir = (out / marker_rel).resolve()
    alt_marker_dir = (out / "output" / "processed").resolve()  # back-compat
    markers = [p for p in {marker_dir, alt_marker_dir} if p.exists()] or [marker_dir, alt_marker_dir]
    return progress_path, markers

def _resolve_progress_paths_for_disk(run_id: str) -> Tuple[Path, List[Path]]:
    """Resolve progress paths when the run isn't in memory (restart scenario)."""
    base = _run_dir(run_id)
    cfg_path = base / "config.yaml"
    out = base / "output"
    try:
        cfg = load_config(cfg_path)
        svc = (cfg.get("service") or {})
        resume = (svc.get("resume") or {})
        progress_rel = resume.get("progress_file", "progress.json")
        marker_rel = resume.get("marker_dir", "processed")
    except Exception:
        progress_rel = "progress.json"
        marker_rel = "processed"

    progress_path = (out / progress_rel).resolve()
    marker_dir = (out / marker_rel).resolve()
    alt_marker_dir = (out / "output" / "processed").resolve()
    markers = [p for p in {marker_dir, alt_marker_dir} if p.exists()] or [marker_dir, alt_marker_dir]
    return progress_path, markers

def _artifact_urls_by_state(state: _RunState) -> dict[str, str]:
    rid = state.run_id
    return {
        "tracks_csv":            _artifact_url(rid, "output/metrics.csv"),
        "waves_csv":             _artifact_url(rid, "output/metrics_waves.csv"),
        "overlay_json":          _artifact_url(rid, "output/overlay/tracks.json"),
        "overlay_json_partial":  _artifact_url(rid, "output/overlay/tracks.partial.json"),
        "overlay_ndjson":        _artifact_url(rid, "output/overlay/tracks.ndjson"),
        "run_json":              _artifact_url(rid, "run.json"),
        "events_ndjson":         _artifact_url(rid, "events.ndjson"),
        "progress_json":         _artifact_url(rid, "output/progress.json"),
        "plots_dir":             _artifact_url(rid, "plots"),
        "output_dir":            _artifact_url(rid, "output"),
        "base_image":            _artifact_url(rid, "output/base.png"),
    }

def _artifact_urls_by_id(run_id: str) -> dict[str, str]:
    # identical, but takes an id
    return {
        "tracks_csv":            _artifact_url(run_id, "output/metrics.csv"),
        "waves_csv":             _artifact_url(run_id, "output/metrics_waves.csv"),
        "overlay_json":          _artifact_url(run_id, "output/overlay/tracks.json"),
        "overlay_json_partial":  _artifact_url(run_id, "output/overlay/tracks.partial.json"),
        "overlay_ndjson":        _artifact_url(run_id, "output/overlay/tracks.ndjson"),
        "run_json":              _artifact_url(run_id, "run.json"),
        "events_ndjson":         _artifact_url(run_id, "events.ndjson"),
        "progress_json":         _artifact_url(run_id, "output/progress.json"),
        "plots_dir":             _artifact_url(run_id, "plots"),
        "output_dir":            _artifact_url(run_id, "output"),
        "base_image":            _artifact_url(run_id, "output/base.png"),
    }

def _pipeline_worker_proc(
    input_dir: str,
    config_path: str,
    output_dir: str,
    plots_dir: str,
    config_overrides: dict | None,
    verbose: bool,
    q,
):
    """
    Runs inside a separate process. It must not touch asyncio.
    Sends simple dict events back to the parent via `q`.
    """
    try:
        for evt in iter_run_project(  # existing generator
            input_dir=Path(input_dir),
            config_path=Path(config_path),
            output_dir=Path(output_dir),
            plots_out=Path(plots_dir),
            progress_cb=None,
            config_overrides=config_overrides,
            verbose=verbose,
            # NOTE: do not rely on cancel_cb in the child; parent will kill us on cancel
            cancel_cb=lambda: False,
        ):
            try:
                q.put(
                    {
                        "phase": getattr(evt, "phase", None),
                        "message": getattr(evt, "message", None),
                        "progress": getattr(evt, "progress", None),
                        "extra": getattr(evt, "extra", None),
                    },
                    timeout=1.0,
                )
            except Exception:
                # If the parent isn't reading fast enough, just drop (best-effort)
                pass
    except Exception as e:
        try:
            q.put({"phase": "ERROR", "message": str(e), "progress": 1.0}, timeout=1.0)
        except Exception:
            pass
    finally:
        # If the generator didn't emit a terminal event, close with DONE
        try:
            q.put({"phase": "DONE", "message": "Run finished", "progress": 1.0}, timeout=1.0)
        except Exception:
            pass

async def _run_pipeline(state: _RunState, config_overrides: Optional[dict], verbose: bool) -> None:
    """
    Run the pipeline in a child process and stream events back to this process.
    On cancel: terminate the child immediately and emit CANCELLED.
    """
    print(f"[PIPELINE] start run={state.run_id} st.ref={getattr(state,'ref','?')}")
    state.set_status("RUNNING")
    await state.publish(JobEvent(phase="INIT", message="pipeline starting", progress=0.0))

    import multiprocessing as mp
    from queue import Empty as QueueEmpty

    loop = asyncio.get_running_loop()
    terminal_seen = False
    terminal_phase: str | None = None
    terminal_message: str | None = None

    # Cross-instance cancel poller (preserves old behavior)
    async def _cancel_poller():
        while state.status in ("QUEUED", "RUNNING"):
            try:
                state._cancel_flag = await _cancel_requested(state.run_id)
            except Exception:
                state._cancel_flag = False
            await asyncio.sleep(2)

    state._cancel_flag = False
    asyncio.create_task(_cancel_poller())

    # Child process and IPC queue
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue(maxsize=1024)
    proc = ctx.Process(
        target=_pipeline_worker_proc,
        args=(
            str(state.input_dir),
            str(state.config_path),
            str(state.output_dir),
            str(state.plots_dir),
            config_overrides or None,
            bool(verbose),
            q,
        ),
        daemon=True,
    )
    proc.start()

    # Track the child lifetime in state.worker_task so resume/cancel logic can see it
    async def _join_proc():
        await asyncio.to_thread(proc.join)
    state.worker_task = asyncio.create_task(_join_proc())

    try:
        while True:
            # Hard cancel path: if user asked to cancel, kill child immediately
            if state.is_cancel_requested() or getattr(state, "_cancel_flag", False):
                if proc.is_alive():
                    proc.terminate()
                    try:
                        await asyncio.to_thread(proc.join, 5.0)
                    except Exception:
                        pass
                terminal_seen = True
                terminal_phase = "CANCELLED"
                terminal_message = "cancel observed"
                state.set_status("CANCELLED")
                await state.publish(JobEvent(phase="CANCELLED", message=terminal_message, progress=1.0))
                break

            # Pull next event with a small timeout so we can re-check cancel frequently
            try:
                item = await asyncio.to_thread(q.get, True, 0.25)
            except QueueEmpty:
                # No event yet; loop to re-check cancel/liveness
                if not proc.is_alive():
                    # Child exited without sending a terminal event (shouldn't happen due to DONE in finally)
                    terminal_seen = True
                    terminal_phase = "DONE"
                    terminal_message = terminal_message or "Run finished"
                    state.set_status("DONE")
                    await state.publish(JobEvent(phase="DONE", message=terminal_message, progress=1.0))
                    break
                continue

            phase = item.get("phase")
            message = item.get("message")
            progress = item.get("progress")
            extra = item.get("extra")
            evt = JobEvent(phase=phase, message=message, progress=progress, extra=extra)

            # Reflect status transitions immediately
            if phase in ("ERROR", "DONE", "CANCELLED"):
                terminal_seen = True
                terminal_phase = phase
                terminal_message = message or terminal_message
                state.set_status(phase, error=(message if phase == "ERROR" else None))

            await state.publish(evt)

            if phase in ("ERROR", "DONE", "CANCELLED"):
                break

        # Ensure child is gone
        if proc.is_alive():
            try:
                await asyncio.to_thread(proc.join, 2.0)
            except Exception:
                pass

    except Exception as e:
        # Any unexpected async-side error
        terminal_seen = True
        terminal_phase = "ERROR"
        terminal_message = str(e)
        state.set_status("ERROR", error=terminal_message)
        await state.publish(JobEvent(phase="ERROR", message=terminal_message, progress=1.0))

    finally:
        # Ensure exactly one terminal event was emitted
        if not terminal_seen:
            ph = "CANCELLED" if (state.is_cancel_requested() or getattr(state, "_cancel_flag", False)) else "ERROR"
            msg = terminal_message or ("cancel observed" if ph == "CANCELLED" else "Run finished")
            state.set_status(ph, error=(msg if ph == "ERROR" else None))
            await state.publish(JobEvent(phase=ph, message=msg, progress=1.0))

def _synthesize_info_from_dir(d: Path) -> Optional[RunInfo]:
    rj = _run_json_path(d.name)
    if not rj.exists():
        return None  # <-- only directories with run.json are real runs

    meta = _load_run_json(rj) or {}
    run_id = meta.get("run_id") or d.name
    name = meta.get("name") or f"run-{run_id}"
    created = meta.get("created_at") or datetime.fromtimestamp(
        d.stat().st_mtime, tz=timezone.utc
    ).isoformat().replace("+00:00", "Z")
    status = meta.get("status") or "DONE"
    input_dir = meta.get("input_dir") or str(d / "input")
    output_dir = meta.get("output_dir") or str(d / "output")
    plots_dir = meta.get("plots_dir") or str(d / "plots")
    config_path = meta.get("config_path") or str(d / "config.yaml")

    return RunInfo(
        run_id=run_id,
        name=name,
        created_at=created,
        status=status,
        error=meta.get("error"),
        input_dir=input_dir,
        output_dir=output_dir,
        plots_dir=plots_dir,
        config_path=config_path,
        owner_sid=meta.get("owner_sid"),
        is_public=bool(meta.get("is_public", False)),
    )


# -----------------------
# Endpoints
# -----------------------

WEB_DIR = Path(os.getenv("WEB_DIR", "/app/web"))

@app.get("/", include_in_schema=False)
async def root_redirect():
    # send bare root to the SPA
    return RedirectResponse(url="/ui/")

@app.get("/ui", include_in_schema=False)
async def ui_slash_redirect():
    return RedirectResponse(url="/ui/")

# SPA fallback for deep links like /ui/viewer/advanced
@app.get("/ui/{path:path}", include_in_schema=False)
async def ui_catchall(path: str):
    index = WEB_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    raise HTTPException(status_code=404, detail="UI bundle not found")

@app.post("/api/runs/_debug_echo")
async def runs_debug_echo(files: List[UploadFile] = File(...)):
    out = []
    for f in files:
        data = await f.read()
        out.append({"filename": f.filename, "bytes": len(data)})
    return {"ok": True, "received": out}


@app.get("/api/uploads")
async def list_uploads(request: Request, sid: str = Depends(get_sid)):
  d = _stage_dir_for(request)
  return {"files": _list_staged(d)}

@app.post("/api/uploads")
async def upload_files(request: Request, files: list[UploadFile] = File(...), append: bool = Query(False), sid: str = Depends(get_sid)):
  d = _stage_dir_for(request)
  if not append:
    # clear existing staged files
    for p in d.glob("*"):
      try: p.unlink()
      except Exception: pass

  saved = await _save_uploads(files, d)
  return {"files": _list_staged(d), "saved": [p.name for p in saved]}

@app.delete("/api/uploads")
async def clear_uploads(request: Request, sid: str = Depends(get_sid)):
  d = _stage_dir_for(request)
  count = 0
  for p in d.glob("*"):
    try:
      p.unlink(); count += 1
    except Exception:
      pass
  return {"cleared": count}

from pathlib import Path, PurePosixPath
from fastapi import HTTPException, Response
from fastapi.responses import FileResponse, RedirectResponse
import inspect
import socket

log = logging.getLogger("files_gateway")

@app.api_route("/files/{run_id}/{path:path}", methods=["GET", "HEAD"])
async def files_gateway(run_id: str, path: str, request: Request):
    import inspect
    # ---------- authn/authz ----------
    st = getattr(request.state, "run", None)
    if not st or getattr(st, "run_id", None) != run_id:
        st = await _load_run_state(run_id)
    if not st:
        hdrs = _dbg_headers(request, run_id=run_id, served_from="miss")
        raise HTTPException(status_code=404, detail="Run not found", headers=hdrs)
    _assert_can_view(st, _sid_from_request(request))

    # ---------- sanitize ----------
    rel = PurePosixPath(path)
    if rel.is_absolute() or ".." in rel.parts or str(rel).strip() == "":
        raise HTTPException(status_code=400, detail="bad path")

    mt = content_type_from_name(rel.name)
    dbg = {"X-Path": str(rel)}
    served_from = "none"

    # ---------- local-first ----------
    base = _run_dir(run_id).resolve()
    full = (base / rel).resolve()
    try:
        full.relative_to(base)
    except Exception:
        raise HTTPException(status_code=400, detail="bad path")

    if full.exists() and full.is_file():
        served_from = "local"
        if request.method == "HEAD":
            try:
                stt = full.stat()
            except FileNotFoundError:
                hdrs = _dbg_headers(request, run_id=run_id, served_from="miss", extra=dbg)
                raise HTTPException(status_code=404, detail="not found", headers=hdrs)
            headers = {
                **_dbg_headers(request, run_id=run_id, served_from=served_from, extra=dbg),
                "Content-Length": str(stt.st_size),
                'ETag': f'W/"{stt.st_mtime_ns:x}-{stt.st_size:x}"',
                "Cache-Control": "private, max-age=60",
            }
            log_dbg.info("FILES HEAD local run=%s path=%s bytes=%d", run_id, rel, stt.st_size)
            return Response(status_code=200, media_type=mt, headers=headers)
        headers = {
            **_dbg_headers(request, run_id=run_id, served_from=served_from, extra=dbg),
            "Cache-Control": "private, max-age=60",
        }
        log_dbg.info("FILES GET local run=%s path=%s", run_id, rel)
        return FileResponse(full, media_type=mt, headers=headers)

    # ---------- remote ----------
    if STORAGE_KIND != "local":
        key = run_key(run_id, *rel.parts)

        if request.method == "HEAD":
            # never redirect HEAD — answer server-side
            exists = None
            if hasattr(runs_storage, "exists"):
                fn = runs_storage.exists
                try:
                    exists = await fn(key) if inspect.iscoroutinefunction(fn) else fn(key)
                except Exception as e:
                    log_dbg.warning("FILES HEAD exists err key=%s %r", key, e)
            if exists:
                headers = _dbg_headers(request, run_id=run_id, served_from="remote-exists", extra=dbg)
                log_dbg.info("FILES HEAD remote-exists key=%s", key)
                return Response(status_code=200, media_type=mt, headers=headers)
            # last resort: small read
            if hasattr(runs_storage, "get"):
                fn = runs_storage.get
                data = await fn(key) if inspect.iscoroutinefunction(fn) else fn(key)
                if data is not None:
                    headers = {
                        **_dbg_headers(request, run_id=run_id, served_from="remote-proxy-head", extra=dbg),
                        "Content-Length": str(len(data)),
                        "Cache-Control": "private, max-age=60",
                    }
                    log_dbg.info("FILES HEAD remote-proxy-head key=%s", key)
                    return Response(status_code=200, media_type=mt, headers=headers)
            headers = _dbg_headers(request, run_id=run_id, served_from="miss", extra=dbg)
            log_dbg.info("FILES HEAD miss run=%s path=%s", run_id, rel)
            raise HTTPException(status_code=404, detail="not found", headers=headers)

        # GET: prefer signed URL; else proxy bytes
        signed = None
        signer = getattr(runs_storage, "get_signed_url", None) or getattr(runs_storage, "signed_url", None)
        if signer:
            try:
                if inspect.iscoroutinefunction(signer):
                    try:
                        signed = await signer(key, method="GET", ttl=300)
                    except TypeError:
                        signed = await signer(key, "GET", 300)
                else:
                    try:
                        signed = signer(key, method="GET", ttl=300)
                    except TypeError:
                        signed = signer(key, "GET", 300)
            except Exception as e:
                log_dbg.warning("FILES sign err key=%s %r", key, e)

        if signed:
            resp = RedirectResponse(signed, status_code=307)
            for k, v in _dbg_headers(request, run_id=run_id,
                                     served_from="remote-redirect", extra=dbg).items():
                resp.headers[k] = v
            log_dbg.info("FILES GET remote-redirect key=%s", key)
            return resp

        if hasattr(runs_storage, "get"):
            fn = runs_storage.get
            data = await fn(key) if inspect.iscoroutinefunction(fn) else fn(key)
            if data is not None:
                headers = {
                    **_dbg_headers(request, run_id=run_id, served_from="remote-proxy", extra=dbg),
                    "Cache-Control": "private, max-age=60",
                }
                log_dbg.info("FILES GET remote-proxy key=%s bytes=%d", key, len(data))
                return Response(content=data, media_type=mt, headers=headers)

    headers = _dbg_headers(request, run_id=run_id, served_from="miss", extra=dbg)
    log_dbg.info("FILES %s miss run=%s path=%s", request.method, run_id, rel)
    raise HTTPException(status_code=404, detail="not found", headers=headers)

@app.post("/api/runs", response_model=CreateRunResponse)
async def create_run(
    request: Request,
    files: Optional[list[UploadFile]] = File(None, description="Optional direct files; if omitted, use session staging"),
    run_name: Optional[str] = Form(None),
    config_overrides: Optional[str] = Form(None),
    verbose: bool = Form(False),
    sid: str = Depends(get_sid)
):
    sid = getattr(request.state, "sid", get_sid(request))
    run_id = uuid.uuid4().hex[:10]
    base_dir = _run_dir(run_id)
    (base_dir / "input").mkdir(parents=True, exist_ok=True)

    staged_used = False
    if files:
        # immediate mode (back-compat)
        saved = await _save_uploads(files, base_dir / "input")
    else:
        # use session staging
        stage_dir = _stage_dir_for(request)
        staged = list(stage_dir.glob("*"))
        if not staged:
            raise HTTPException(status_code=400, detail="No staged files. Upload first.")
        for p in staged:
            shutil.copy2(p, base_dir / "input" / p.name)
        staged_used = True
        saved = list((base_dir / "input").glob("*"))

    if not saved:
        raise HTTPException(status_code=400, detail="No files to run")

    # copy config (as before)
    if not DEFAULT_CONFIG.exists():
        raise HTTPException(status_code=500, detail=f"Default config not found: {DEFAULT_CONFIG}")
    config_path = base_dir / "config.yaml"
    shutil.copyfile(DEFAULT_CONFIG, config_path)

    overrides: Optional[dict] = None
    if config_overrides:
        try:
            overrides = json.loads(config_overrides)
            if not isinstance(overrides, dict):
                raise ValueError("config_overrides must be a JSON object")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid config_overrides JSON: {e}")

    state = _RunState(
        run_id=run_id,
        name=run_name or f"run-{run_id}",
        base_dir=base_dir,
        config_path=config_path,
        created_at_iso=_iso_now(),
        status="QUEUED",
        owner_sid=sid,
    )
    _RUNS[run_id] = state

    await _persist_run_meta(state)
    asyncio.create_task(_run_pipeline(state, overrides, verbose))
    await _publish_runs_dirty()

    # optional: clear staging after successful start
    if staged_used:
        try:
            for p in _stage_dir_for(request).glob("*"):
                p.unlink()
        except Exception:
            pass

    return CreateRunResponse(run_id=run_id, status=state.status, info=state.info())

@app.post("/api/uploads/start", response_model=StartUploadResp)
async def start_upload(
    file_name: str = Form(...),
    content_type: str = Form("application/octet-stream"),
    sid = Depends(get_sid)
):
    if not GCS_BUCKET:
        raise HTTPException(500, "GCS_BUCKET not configured")
    # sanitize
    safe = re.sub(r"[^A-Za-z0-9._-]+","_", Path(file_name).name) or "upload.bin"
    object_name = f"uploads/{uuid.uuid4().hex}/{safe}"

    def _make_session() -> str:
        client = storage.Client()
        blob = client.bucket(GCS_BUCKET).blob(object_name)
        # This returns a resumable-session URL the browser can PUT the file to.
        return blob.create_resumable_upload_session(content_type=content_type)
    upload_url = await asyncio.to_thread(_make_session)
    return StartUploadResp(upload_url=upload_url, object=object_name)

class RunFromGCSReq(BaseModel):
    objects: List[str]               # e.g. ["uploads/ab12/file.csv", "samples/ripple.csv"]
    run_name: Optional[str] = None
    config_overrides: Optional[dict] = None
    verbose: bool = False

log = logging.getLogger("api")

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def content_type_from_name(name: str) -> str:
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return {
        "csv": "text/csv",
        "tsv": "text/tab-separated-values",
        "xls": "application/vnd.ms-excel",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "npy": "application/octet-stream",
        "json": "application/json",
    }.get(ext, "application/octet-stream")

@app.post("/api/runs/from_gcs", response_model=CreateRunResponse)
async def create_run_from_gcs(request: Request, sid: str = Depends(ensure_sid)):
    """
    Create a run from objects that already exist in GCS (or configured runs_storage).
    - Writes inputs to the local working dir: RUNS_DIR/<run_id>/input/<name>
    - Also mirrors inputs to the canonical store: runs/<run_id>/input/<name>
    - Persists run.json to the canonical store so list/detail work across instances
    """
    def _safe_name(name: str) -> str:
        # basic traversal guard
        name = (name or "").strip().replace("\\", "/").split("/")[-1]
        return name or "input"

    try:
        payload = await request.json()
        objects = payload.get("objects") or []
        if not objects:
            raise HTTPException(status_code=400, detail="objects[] is required")

        # Ensure default config exists
        if not DEFAULT_CONFIG.exists():
            log.error("DEFAULT_CONFIG missing: %s", DEFAULT_CONFIG)
            raise HTTPException(status_code=500, detail=f"Default config not found: {DEFAULT_CONFIG}")

        # ---- Create run skeleton on local FS ----
        run_id = uuid.uuid4().hex[:10]
        base_dir = _run_dir(run_id)
        input_dir = base_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        # Copy default config into run folder
        config_path = base_dir / "config.yaml"
        shutil.copyfile(DEFAULT_CONFIG, config_path)

        # ---- Stage each object: local + canonical store ----
        for obj_key in objects:
            name = _safe_name(obj_key)
            try:
                # Read from the source/staging area (GCS, etc.)
                data = _read_input_object_bytes(obj_key)
            except Exception:
                log.exception("read_input_object failed for %s", obj_key)
                raise HTTPException(status_code=400, detail=f"cannot read staged object: {obj_key}")

            # 1) Local write so the pipeline can discover inputs
            try:
                dst_local = input_dir / name
                dst_local.parent.mkdir(parents=True, exist_ok=True)
                dst_local.write_bytes(data)
            except Exception:
                log.exception("failed to write local input %s", name)
                raise HTTPException(status_code=500, detail=f"failed to stage local input: {name}")

            # 2) Canonical store (so other instances can see inputs later)
            try:
                dst_key = run_key(run_id, "input", name)  # e.g. runs/<id>/input/<name>
                await runs_storage.put(dst_key, data, content_type=content_type_from_name(name))
            except Exception:
                log.exception("runs_storage.put failed for %s -> %s", obj_key, dst_key)
                # Keep the 502 semantics to signal upstream storage trouble
                raise HTTPException(status_code=502, detail="storage write failed")

        # ---- Create state & persist meta ----
        st = _RunState(
            run_id=run_id,
            name=payload.get("run_name") or f"run-{run_id}",
            base_dir=base_dir,
            config_path=config_path,
            created_at_iso=_iso_now(),
            status="QUEUED",
            owner_sid=sid,
            is_public=False,
        )
        # Scope access to the creator; to_meta() should serialize this as a list
        try:
            st.allowed_sids = {sid}
        except Exception:
            # if model uses a list, fall back
            st.allowed_sids = [sid]

        _RUNS[run_id] = st
        await _persist_run_meta(st)  # IMPORTANT: this should write run.json to GCS

        # ---- Start the pipeline ----
        config_overrides = payload.get("config_overrides")
        verbose = bool(payload.get("verbose") or False)
        try:
            asyncio.create_task(_run_pipeline(st, config_overrides, verbose))
        except Exception:
            log.exception("failed to schedule pipeline; trying thread executor")
            try:
                asyncio.get_running_loop().run_in_executor(
                    None, lambda: asyncio.run(_run_pipeline(st, config_overrides, verbose))
                )
            except Exception:
                log.exception("failed to schedule pipeline in executor")
                raise HTTPException(status_code=500, detail="failed to start pipeline")

        # Best-effort notify list subscribers
        try:
            await _publish_runs_dirty()
        except Exception:
            log.exception("_publish_runs_dirty failed")

        return CreateRunResponse(run_id=run_id, status=st.status, info=st.info())

    except HTTPException:
        raise
    except Exception:
        log.exception("create_run_from_gcs failed")
        raise HTTPException(status_code=500, detail="internal error starting run")

def _assert_can_view(st: _RunState, sid: str):
    if getattr(st, "is_public", False):
        return
    owner = getattr(st, "owner_sid", None)
    allowed = set(getattr(st, "allowed_sids", []) or [])
    if sid and owner and sid == owner:
        return
    if sid in allowed:
        return
    log.warning("ACL deny run=%s sid=%s owner=%s allowed=%s", st.run_id, sid, owner, list(allowed))
    raise HTTPException(status_code=403, detail="guard error")

@app.get("/api/samples/{name}/signed")
def signed_sample(name: str):
    client = storage.Client()
    blob = client.bucket(GCS_BUCKET).blob(f"samples/{Path(name).name}")
    url = blob.generate_signed_url(version="v4", expiration=timedelta(minutes=15), method="GET")
    return {"url": url}

RUN_ID_RE = re.compile(r"^[a-f0-9]{6,32}$", re.I)  # optional, for extra filtering

@app.get("/api/runs")
async def list_runs(request: Request, sid: str = Depends(ensure_sid)):
    t0 = time.time()
    try:
        runs: list[dict] = []

        # 1) try storage listing
        try:
            prefixes = [os.getenv("RUNS_KEY_PREFIX") or "", "runs"]
            seen: set[str] = set()
            for pref in prefixes:
                pref = (pref.rstrip("/") + "/") if pref and not pref.endswith("/") else pref
                for key in await runs_storage.list(prefix=pref):
                    if not key.endswith("/run.json"):
                        continue
                    run_id = key.split("/")[-2]  # ".../<id>/run.json"
                    if run_id in seen:
                        continue
                    seen.add(run_id)
                    meta = await _load_run_meta_from_storage(run_id)
                    if not meta:
                        continue

                    owner = meta.get("owner_sid")
                    allowed = set(meta.get("allowed_sids") or [])
                    if meta.get("is_public") or (sid and (sid == owner or sid in allowed)):
                        runs.append({
                            "run_id": run_id,
                            "status": meta.get("status", "UNKNOWN"),
                            "name": meta.get("name") or meta.get("run_name"),
                            "created_at": meta.get("created_at"),
                        })
        except Exception:
            # non-fatal; fall back to local
            pass

        # 2) fallback to local if storage listing failed/empty
        if not runs:
            try:
                base = RUNS_DIR
                for child in base.iterdir():
                    p = child / "run.json"
                    if not p.is_file():
                        continue
                    try:
                        meta = json.loads(p.read_text(encoding="utf-8"))
                        owner = meta.get("owner_sid")
                        allowed = set(meta.get("allowed_sids") or [])
                        if meta.get("is_public") or (sid and (sid == owner or sid in allowed)):
                            runs.append({
                                "run_id": child.name,
                                "status": meta.get("status", "UNKNOWN"),
                                "name": meta.get("name") or meta.get("run_name"),
                                "created_at": meta.get("created_at"),
                            })
                    except Exception:
                        continue
            except Exception:
                pass

        # sort newest first
        runs.sort(key=lambda r: (r.get("created_at") or 0), reverse=True)

        hdrs = _dbg_headers(request, extra={"X-Runs-Count": len(runs), "X-Elapsed-ms": _ms(t0)})
        log_dbg.info("LIST_RUNS ok count=%d sid=%s ms=%d",
                     len(runs), (sid[:8] if sid else ""), _ms(t0))
        return JSONResponse(runs, headers=hdrs)

    except Exception as e:
        hdrs = _dbg_headers(request, extra={
            "X-Elapsed-ms": _ms(t0),
            "X-Dbg-Why": f"err:{type(e).__name__}"
        })
        log_dbg.exception("LIST_RUNS fail sid=%s ms=%d", (sid[:8] if sid else ""), _ms(t0))
        raise HTTPException(status_code=500, detail="list_runs error", headers=hdrs)

@app.get("/api/runs/{run_id}")
async def get_run_status(run_id: str, request: Request):
    t0 = time.time()
    st = await _load_run_state(run_id)
    if not st:
        hdrs = _dbg_headers(request, run_id=run_id, extra={"X-Dbg-Why":"no-run","X-Elapsed-ms":_ms(t0)})
        log_dbg.info("GET_RUN 404 run=%s", run_id)
        raise HTTPException(status_code=404, detail="Run not found", headers=hdrs)
    try:
        _assert_can_view(st, _sid_from_request(request))
    except HTTPException as he:
        he.headers = {**(he.headers or {}), **_dbg_headers(request, run_id=run_id, extra={"X-Dbg-Why":"forbidden","X-Elapsed-ms":_ms(t0)})}
        log_dbg.info("GET_RUN 403 run=%s", run_id)
        raise
    payload = st.to_status_response() if hasattr(st, "to_status_response") else {
        "run_id": st.run_id, "status": st.status, "error": st.error
    }
    hdrs = _dbg_headers(request, run_id=run_id, extra={"X-Elapsed-ms": _ms(t0)})
    log_dbg.info("GET_RUN 200 run=%s status=%s ms=%d", run_id, payload.get("status"), _ms(t0))
    return JSONResponse(payload, headers=hdrs)

@app.get("/api/runs/{run_id}", dependencies=[Depends(require_run_view)])
async def get_run_info_api(run_id: str):
    st = _RUNS.get(run_id)
    if st:
        return {"info": st.info(), "artifacts": _artifact_urls_by_state(st)}

    # local disk fallback
    rj = _run_json_path(run_id)
    if rj.exists():
        info = _synthesize_info_from_dir(_run_dir(run_id))
        if info:
            return {"info": info, "artifacts": _artifact_urls_by_id(run_id)}

    # GCS fallback
    try:
        data = await runs_storage.get(safe_join("runs", run_id, "run.json"))
        if data:
            meta = json.loads(data)
            info = RunInfo(
                run_id=meta.get("run_id", run_id),
                name=meta.get("name", f"run-{run_id}"),
                created_at=meta.get("created_at", _iso_now()),
                status=meta.get("status", "DONE"),
                error=meta.get("error"),
                input_dir=meta.get("input_dir", ""),
                output_dir=meta.get("output_dir", ""),
                plots_dir=meta.get("plots_dir"),
                config_path=meta.get("config_path", ""),
            )
            return {"info": info, "artifacts": _artifact_urls_by_id(run_id)}
    except Exception:
        pass

    raise HTTPException(404, "run not found")

@app.get("/api/runs/{run_id}", response_model=RunStatusResponse, dependencies=[Depends(require_run_view)])
async def get_run(run_id: str):
    """Run status + best-effort artifact URLs."""
    st = _RUNS.get(run_id)
    if st:
        return RunStatusResponse(info=st.info(), artifacts=_artifact_urls_by_state(st))

    # Fallback to disk-only (e.g., after API restart)
    d = _run_dir(run_id)
    if not d.exists():
        raise HTTPException(status_code=404, detail="Unknown run_id")
    info = _synthesize_info_from_dir(d)
    if not info:
        raise HTTPException(status_code=404, detail="Unknown run_id")
    return RunStatusResponse(info=info, artifacts=_artifact_urls_by_id(run_id))


@app.get("/api/runs/{run_id}/artifacts")
async def get_run_artifacts(run_id: str):
    """Raw artifact URL map (even if files not yet available)."""
    st = _RUNS.get(run_id)
    if st:
        return JSONResponse(_artifact_urls_by_state(st))
    # disk fallback
    d = _run_dir(run_id)
    if not d.exists():
        raise HTTPException(status_code=404, detail="Unknown run_id")
    return JSONResponse(_artifact_urls_by_id(run_id))


@app.get("/api/runs/{run_id}/events", dependencies=[Depends(require_run_view)])
async def stream_events(run_id: str, replay: int = Query(0)):
    st = _RUNS.get(run_id)

    async def gen():
        # ---- initial replay (local first, then storage) ----
        data: bytes | None = None
        try:
            local = _events_path(run_id)
            data = local.read_bytes() if local.exists() else None
        except Exception:
            data = None

        if data is None and STORAGE_KIND != "local":
            try:
                data = await runs_storage.get(run_key(run_id, "events.ndjson"))
            except Exception:
                data = None

        if data:
            lines = data.splitlines()
            for line in lines[-max(0, replay):]:
                if line.strip():
                    yield {"event": "message", "data": line.decode("utf-8", errors="ignore")}

        # ---- live streaming ----
        if st:
            # Same-instance: subscribe to in-memory queue
            q = await st.add_subscriber()
            try:
                while True:
                    evt = await q.get()
                    yield {"event": "message", "data": json.dumps(asdict(evt))}
            finally:
                await st.remove_subscriber(q)
        else:
            # Cross-instance: tail the persisted NDJSON in object storage
            last_len = len(data) if data else 0
            while True:
                blob: bytes | None = None
                try:
                    blob = await runs_storage.get(run_key(run_id, "events.ndjson"))
                except Exception:
                    blob = None

                if blob and len(blob) > last_len:
                    chunk = blob[last_len:]
                    for line in chunk.splitlines():
                        if line.strip():
                            try:
                                yield {"event": "message", "data": line.decode("utf-8", errors="ignore")}
                            except Exception:
                                pass
                    last_len = len(blob)

                # modest poll (EventSourceResponse will also send keepalives if ping>0)
                await asyncio.sleep(1.0)

    # Helpful for proxies that buffer
    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return EventSourceResponse(gen(), ping=(SSE_PING_SECONDS or None), headers=headers)

@app.get("/api/runs/events")
async def stream_runs_events():
    q = await _global_add_sub()
    async def gen():
        try:
            # initial nudge so clients can immediately refresh once
            yield {"event": "message", "data": json.dumps({"type": "runs", "dirty": {"list": True}})}
            while True:
                msg = await q.get()
                yield {"event": "message", "data": json.dumps(msg)}
        except asyncio.CancelledError:
            return
        finally:
             await _global_remove_sub(q)
    return EventSourceResponse(gen(), ping=25)


@app.get("/api/runs/{run_id}/image", dependencies=[Depends(require_run_view)])
async def get_base_image(run_id: str, request: Request):
    url = f"/files/{run_id}/output/base.png"
    hdrs = _dbg_headers(request, run_id=run_id, served_from="image-redirect")
    log_dbg.info("IMAGE 307 run=%s -> %s", run_id, url)
    resp = RedirectResponse(url=url, status_code=307)
    for k, v in hdrs.items():
        resp.headers.setdefault(k, v)
    return resp

@app.get("/api/runs/{run_id}/waves", dependencies=[Depends(require_run_view)])
async def list_wave_windows(run_id: str, track: str | None = None, request: Request = None):
    t0 = time.time()
    if not track:
        raise HTTPException(status_code=400, detail="Query param 'track' is required",
                            headers=_dbg_headers(request, run_id=run_id, extra={"X-Elapsed-ms":_ms(t0)}))

    base = _run_dir(run_id)
    win_dir = base / "plots" / str(track) / "peak_windows"
    names, src = [], "none"

    if win_dir.exists():
        names = sorted([p.name for p in win_dir.glob("*.png")])
        src = "local"
    elif STORAGE_KIND != "local" and hasattr(runs_storage, "list"):
        try:
            prefix = run_key(run_id, "plots", str(track), "peak_windows")
            keys = await runs_storage.list(prefix=prefix)
            names = sorted([k.rsplit("/", 1)[-1] for k in keys if k.endswith(".png")])
            src = "remote"
        except Exception:
            src = "remote-error"

    urls = [f"/files/{run_id}/plots/{track}/peak_windows/{n}" for n in names]
    hdrs = _dbg_headers(request, run_id=run_id, served_from=src, extra={"X-Image-Count": len(names), "X-Elapsed-ms": _ms(t0)})
    log_dbg.info("WAVES 200 run=%s track=%s src=%s count=%d ms=%d",
                 run_id, track, src, len(names), _ms(t0))
    return JSONResponse({"images": urls}, headers=hdrs)

@app.get("/api/runs/{run_id}/tracks", dependencies=[Depends(require_run_view)])
async def list_tracks(run_id: str, request: Request):
    t0 = time.time()
    base = _run_dir(run_id)
    preferred = base / "output" / "tracks"
    names, src = [], "none"

    if preferred.exists():
        names = sorted([p.name for p in preferred.glob("*.npy")])
        src = "local"
    elif STORAGE_KIND != "local" and hasattr(runs_storage, "list"):
        try:
            keys = await runs_storage.list(run_key(run_id, "output", "tracks"))
            names = sorted([k.rsplit("/", 1)[-1] for k in keys if k.endswith(".npy")])
            src = "remote"
        except Exception:
            src = "remote-error"

    urls = [f"/files/{run_id}/output/tracks/{n}" for n in names]
    hdrs = _dbg_headers(request, run_id=run_id, served_from=src, extra={"X-Track-Count": len(names), "X-Elapsed-ms": _ms(t0)})
    log_dbg.info("TRACKS 200 run=%s src=%s count=%d ms=%d", run_id, src, len(names), _ms(t0))
    return JSONResponse({"tracks": urls}, headers=hdrs)


async def _do_cancel(st: _RunState) -> dict:
    # Thread-safe cancel (worker thread sees it)
    if hasattr(st, "cancel"):
        st.cancel()           # new helper on _RunState using threading.Event
    else:
        # fallback haven't added .cancel()
        st.cancel_event.set()
    # Cross-instance hint (optional)
    await _write_control(st.run_id, "cancel.json", {"ts": _iso_now()})
    return {"run_id": st.run_id, "status": st.status, "cancel_requested": True, "ok": True}


async def _do_resume(st: _RunState, *, verbose: bool = False) -> dict:
    # Clear prior cancel
    if hasattr(st, "reset_cancel"):
        st.reset_cancel()
    else:
        st.cancel_event.clear()
    await _write_control(st.run_id, "resume.json", {"ts": _iso_now(), "verbose": bool(verbose)})

    # If already running, don't double-start
    if st.worker_task and not st.worker_task.done():
        return {"run_id": st.run_id, "status": "RUNNING", "note": "already running", "ok": True}

    # (Re)launch the pipeline
    st.worker_task = asyncio.create_task(_run_pipeline(st, config_overrides=None, verbose=bool(verbose)))
    # reflect immediately for UI; the worker will eventually publish terminal state
    st.set_status("RUNNING")
    return {"run_id": st.run_id, "status": "RUNNING", "ok": True}

@app.post("/api/runs/{run_id}/cancel", dependencies=[Depends(require_run_view)])
async def cancel_run(run_id: str):
    st = _RUNS.get(run_id)
    if not st:
        raise HTTPException(status_code=404, detail="Unknown run_id")

    before = getattr(st.cancel_event, "is_set", lambda: False)()
    st.cancel()  # threading.Event under the hood
    after  = st.is_cancel_requested()

    # Emit an SSE line so it is seen in the browser log immediately
    try:
        await st.publish(JobEvent(phase="CANCEL", message="cancel requested by user"))
    except Exception:
        pass

    print(f"[CANCEL] run={run_id} st.ref={getattr(st,'ref','?')} before={before} after={after}")
    await _write_control(run_id, "cancel.json", {"ts": _iso_now()})

    return {"ok": True, "run_id": run_id, "ref": getattr(st, "ref", "?"), "cancel_set": after}


@app.post("/api/runs/{run_id}/resume", dependencies=[Depends(require_run_view)])
async def resume_run(run_id: str, verbose: bool = Form(False)):
    st = _RUNS.get(run_id)
    if not st:
        raise HTTPException(status_code=404, detail="Unknown run_id")
    return await _do_resume(st, verbose=verbose)

@app.get("/api/runs/{run_id}/progress")
async def get_progress(run_id: str):
    st = await _load_run_state(run_id)
    if not st:
        raise HTTPException(status_code=404, detail="Run not found")

    p = _progress_json_path(run_id)

    body = _try_read_json_bytes(p)
    if body is None and STORAGE_KIND != "local":
        key = run_key(run_id, "output", "progress.json")
        data = await runs_storage.get(key)
        if data:
            body = data
            try:
                p.write_bytes(data)  # opportunistic cache
            except Exception:
                pass

    if body is None:
        return Response(status_code=204)

    return Response(content=body, media_type="application/json")

@app.get("/api/health")
def health():
    return {"ok": True, "version": app.version, "runs_dir": str(RUNS_DIR), "default_config": str(DEFAULT_CONFIG)}

@app.get("/api/debug/whoami")
def whoami(request: Request):
    return {"sid": get_sid(request)}

@app.delete("/api/runs/{run_id}", dependencies=[Depends(require_run_view)])
async def delete_run(run_id: str, force: int = Query(0, description="1 = cancel if running, then delete")):
    st = _RUNS.get(run_id)
    base = _run_dir(run_id)
    if not base.exists() and not st:
        raise HTTPException(status_code=404, detail="Unknown run_id")

    # If active
    if st and st.status in ("QUEUED", "RUNNING"):
        if not force:
            raise HTTPException(status_code=409, detail="Cannot delete an active run (pass ?force=1 to cancel+delete)")
        # request cancel
        st.cancel_event.set()
        # give the worker a brief chance to exit cleanly
        try:
            if st.worker_task:
                await asyncio.wait_for(st.worker_task, timeout=2.0)
        except Exception:
            # best-effort: try to cancel the task cooperatively
            try:
                st.worker_task.cancel()  # type: ignore
            except Exception:
                pass

    # Remove directory
    try:
        shutil.rmtree(base, ignore_errors=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")

    _RUNS.pop(run_id, None)

    await _publish_runs_dirty()

    return {"deleted": run_id}

def _sample_name_from_arr_path(arr_path: Path) -> str:
    base = arr_path.parent.parent.name
    return base[:-8] if base.endswith("_heatmap") else base

def _parse_index_range(spec: Optional[str], n: int) -> Optional[Tuple[int, int]]:
    """
    Parse 'lo:hi' (inclusive) into clamped indices. Either side may be blank.
    Examples: '100:300', '200:', ':500'.
    Returns None if spec is falsy.
    """
    if not spec:
        return None
    try:
        lo_s, hi_s = (spec.split(":", 1) + [""])[:2]
        lo = int(lo_s) if lo_s.strip() else 0
        hi = int(hi_s) if hi_s.strip() else (n - 1)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid range format; expected 'lo:hi'")
    lo = max(0, min(n - 1, lo))
    hi = max(0, min(n - 1, hi))
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi

def _lookup_metrics_freq(run_dir: Path, track_id: str | int) -> Optional[float]:
    """Try to pull dominant_frequency from overlay/tracks.json."""
    try:
        overlay = run_dir / "overlay" / "tracks.json"
        if not overlay.exists():
            return None
        with overlay.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        tid = str(track_id)
        for t in payload.get("tracks", []):
            if str(t.get("id")) == tid:
                fval = t.get("metrics", {}).get("dominant_frequency")
                if fval is None:
                    return None
                fval = float(fval)
                return fval if math.isfinite(fval) and fval > 0 else None
    except Exception:
        return None
    return None

@app.get("/api/runs/{run_id}/snapshot", dependencies=[Depends(require_run_view)])
async def get_snapshot(run_id: str, request: Request):
    t0 = time.time()
    st = await _load_run_state(run_id)
    if not st:
        hdrs = _dbg_headers(request, run_id=run_id, extra={"X-Dbg-Why":"no-run","X-Elapsed-ms":_ms(t0)})
        log_dbg.info("SNAPSHOT 404 run=%s", run_id)
        raise HTTPException(status_code=404, detail="Run not found", headers=hdrs)

    # --- existing local Path.exists() flags ---
    base = _run_dir(run_id)
    flags = {
        "overlay_tracks_json": (base / "output" / "overlay" / "tracks.json").exists(),
        "overlay_tracks_ndjson": (base / "output" / "overlay" / "tracks.ndjson").exists(),
        "base_image": (base / "output" / "base.png").exists(),
        "progress_json": (base / "output" / "progress.json").exists(),
    }

    # Remote fallback checks if needed
    remote_checked = 0
    if STORAGE_KIND != "local" and not all(flags.values()):
        remote_keys = {
            "overlay_tracks_json": run_key(run_id, "output", "overlay", "tracks.json"),
            "overlay_tracks_partial": run_key(run_id, "output", "overlay", "tracks.partial.json"),
            "overlay_tracks_ndjson": run_key(run_id, "output", "overlay", "tracks.ndjson"),
            "base_image": run_key(run_id, "output", "base.png"),
        }
        for name, key in remote_keys.items():
            if not flags.get(name, False):
                try:
                    if await runs_storage.exists(key):
                        flags[name] = True
                except Exception:
                    pass

    payload = {
        "run_id": run_id,
        "status": st.status,
        "error": st.error,
        "overlay_version": getattr(st, "overlay_version", 0),
        "artifacts": flags,
        "progress": None,
    }
    hdrs = _dbg_headers(request, run_id=run_id, extra={"X-Remote-Checked": remote_checked, "X-Elapsed-ms": _ms(t0)})
    log_dbg.info("SNAPSHOT 200 run=%s remote_checked=%d ms=%d", run_id, remote_checked, _ms(t0))
    return JSONResponse(payload, headers=hdrs)

def _try_ndjson_bytes_as_overlay_bytes(
    ndjson_bytes: bytes,
    version: int | None = None,
) -> Optional[bytes]:
    """
    Extract the most recent overlay payload from an .ndjson stream and return it
    as JSON-encoded bytes. Supports lines that are either:
      • a bare overlay object (has keys like 'tracks'/'windows'/'image'), or
      • a wrapper record (e.g., {'event':'OVERLAY','data': {...}}),
      • or {'overlay': {...}}.

    Returns None if nothing overlay-like is found.
    """
    last_overlay: Optional[dict] = None

    # decode safely and strip any BOM
    text = ndjson_bytes.decode("utf-8", errors="replace").lstrip("\ufeff")

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # tolerate SSE-style "data: { ... }" lines if ever present
        if line.startswith("data:"):
            line = line[5:].strip()

        try:
            obj = json.loads(line)
        except Exception:
            continue

        cand = None
        if isinstance(obj, dict):
            if "overlay" in obj and isinstance(obj["overlay"], dict):
                cand = obj["overlay"]
            elif obj.get("event") in ("OVERLAY", "overlay") and isinstance(obj.get("data"), dict):
                cand = obj["data"]
            elif obj.get("type") in ("OVERLAY", "overlay") and isinstance(obj.get("data"), dict):
                cand = obj["data"]
            elif any(k in obj for k in ("tracks", "windows", "image")):
                # treat as a bare overlay snapshot
                cand = obj

        if cand:
            last_overlay = cand  # keep the most recent snapshot

    if last_overlay is None:
        return None

    # align/attach version if provided
    if version is not None:
        try:
            if int(last_overlay.get("version", -1)) < int(version):
                last_overlay["version"] = int(version)
        except Exception:
            last_overlay["version"] = version

    try:
        return json.dumps(last_overlay, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except Exception:
        return None

@app.get("/api/runs/{run_id}/overlay", dependencies=[Depends(require_run_view)])
async def get_overlay(run_id: str, request: Request):
    t0 = time.time()
    st = await _load_run_state(run_id)
    if not st:
        hdrs = _dbg_headers(request, run_id=run_id, extra={"X-Dbg-Why":"no-run","X-Elapsed-ms":_ms(t0)})
        log_dbg.info("OVERLAY 404 run=%s", run_id)
        raise HTTPException(status_code=404, detail="Run not found", headers=hdrs)

    j = _overlay_json_path(run_id)
    nd = _overlay_ndjson_path(run_id)
    body = None
    src = None

    if j.exists():
        body = _try_read_json_bytes(j)
        src = "local-json"
    if body is None and nd.exists():
        body = _try_read_ndjson_as_overlay_bytes(nd, version=getattr(st, "overlay_version", 0))
        src = "local-ndjson"

    if body is None and STORAGE_KIND != "local":
        try:
            data = await runs_storage.get(run_key(run_id, "output", "overlay", "tracks.json"))
            if data:
                body = data
                src = "remote-json"
        except Exception:
            pass
        if body is None:
            try:
                ndbytes = await runs_storage.get(run_key(run_id, "output", "overlay", "tracks.ndjson"))
                if ndbytes:
                    body = _try_ndjson_bytes_as_overlay_bytes(ndbytes, version=getattr(st, "overlay_version", 0))
                    src = "remote-ndjson"
                    # best-effort cache
                    try:
                        nd.parent.mkdir(parents=True, exist_ok=True)
                        nd.write_bytes(ndbytes)
                    except Exception:
                        pass
            except Exception:
                pass

    if body is None:
        hdrs = _dbg_headers(request, run_id=run_id, served_from="miss", extra={"X-Dbg-Why":"no-overlay","X-Elapsed-ms":_ms(t0)})
        log_dbg.info("OVERLAY 204 run=%s", run_id)
        return Response(status_code=204, headers=hdrs)

    hdrs = _dbg_headers(request, run_id=run_id, served_from=src, extra={"X-Elapsed-ms": _ms(t0)})
    log_dbg.info("OVERLAY 200 run=%s src=%s bytes=%d ms=%d", run_id, src, len(body), _ms(t0))
    return Response(content=body, media_type="application/json", headers=hdrs)



logger = logging.getLogger("waves.api")

@app.get("/api/runs/{run_id}/tracks/{track_id}", dependencies=[Depends(require_run_view)])
async def get_track_detail(
    run_id: str,
    track_id: str,
    include_sine: bool = Query(False, description="Include phase-anchored sine overlay (baseline + fitted residual)."),
    include_residual: bool = Query(False, description="Include residual array (position - baseline)."),
    index_range: Optional[str] = Query(None, alias="range", description="Optional index window 'lo:hi' (inclusive)."),
    freq_source: Literal["auto", "metrics"] = Query("auto", description="Use 'auto' (recompute) or 'metrics' (overlay) frequency."),
):
    # --- runs root (support either constant name) ---
    
    runs_root = RUNS_DIR

    run_dir = runs_root / run_id
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise HTTPException(status_code=404, detail="config.yaml not found for run")

    cfg = load_config(cfg_path)

    # --- locate track npy (local first) ---
    npy: Optional[Path] = None
    for p in run_dir.rglob(f"{track_id}.npy"):
        npy = p
        break

    if npy is None:
        candidates = list(run_dir.rglob("*.npy"))
        if candidates:
            pat_suffix = re.compile(rf".*_(?:0+)?{re.escape(track_id)}$", re.IGNORECASE)
            pat_prefix = re.compile(rf"^(?:0+)?{re.escape(track_id)}_.*$", re.IGNORECASE)
            def score(path: Path) -> int:
                s = path.stem
                base = 0
                if "output/tracks" in str(path).replace("\\", "/"): base += 10
                if s.lower() == track_id.lower(): base += 5
                elif pat_suffix.match(s) or pat_prefix.match(s): base += 3
                if track_id.isdigit():
                    m = re.search(r"(\d+)$", s)
                    if m and int(m.group(1)) == int(track_id): base += 2
                return base
            scored = sorted(candidates, key=score, reverse=True)
            if scored and score(scored[0]) > 0:
                npy = scored[0]

    # --- remote (GCS) fallback when configured ---
    blob_bytes: Optional[bytes] = None
    remote_key: Optional[str] = None
    try:
        use_remote = (STORAGE_KIND != "local")
    except Exception:
        use_remote = False

    if npy is None and use_remote:
        try:
            keys = await runs_storage.list(run_key(run_id, "output", "tracks"))
        except Exception:
            keys = []
        if keys:
            def stem(k: str) -> str:
                n = k.rsplit("/", 1)[-1]
                return n[:-4] if n.lower().endswith(".npy") else n
            pat_suffix = re.compile(rf".*_(?:0+)?{re.escape(track_id)}$", re.IGNORECASE)
            pat_prefix = re.compile(rf"^(?:0+)?{re.escape(track_id)}_.*$", re.IGNORECASE)
            def kscore(k: str) -> int:
                s = stem(k)
                base = 0
                if s.lower() == track_id.lower(): base += 5
                elif pat_suffix.match(s) or pat_prefix.match(s): base += 3
                if track_id.isdigit():
                    m = re.search(r"(\d+)$", s)
                    if m and int(m.group(1)) == int(track_id): base += 2
                return base
            keys_sorted = sorted([k for k in keys if k.endswith(".npy")], key=kscore, reverse=True)
            if keys_sorted and kscore(keys_sorted[0]) > 0:
                remote_key = keys_sorted[0]
                blob_bytes = await runs_storage.get(remote_key)

    if npy is None and blob_bytes is None:
        raise HTTPException(status_code=404, detail="track not found")

    # --- load data ---
    if npy is not None:
        xy = np.load(npy, allow_pickle=False)  # expected (N,2): [[y,row],[x,col]]
        sample_name = _sample_name_from_arr_path(npy)
    else:
        xy = np.load(io.BytesIO(blob_bytes), allow_pickle=False)
        sample_name = remote_key.rsplit("/", 1)[-1] if remote_key else None

    if xy.ndim != 2 or xy.shape[1] != 2:
        raise HTTPException(status_code=500, detail="Invalid track array shape")

    time_index = xy[:, 0].astype(float)   # rows (time)
    position   = xy[:, 1].astype(float)   # cols (position)
    N = len(time_index)

    # --- config pieces ---
    detrend_cfg = (cfg.get("detrend") or {}).copy()
    degree = int(detrend_cfg.pop("degree", 1))
    peaks_cfg = cfg.get("peaks", {}) or {}
    period_cfg = (cfg.get("period", {}) or {}).copy()
    io_cfg = cfg.get("io", {}) or {}
    sampling_rate = float(io_cfg.get("sampling_rate", period_cfg.get("sampling_rate", 1.0)))
    period_cfg.setdefault("sampling_rate", sampling_rate)

    # --- baseline via RANSAC (parity with CLI) ---
    model = fit_baseline_ransac(time_index, position, degree=degree, **detrend_cfg)
    baseline_pos = model.predict(time_index.reshape(-1, 1)).astype(float)
    residual_pos = (position - baseline_pos).astype(float)

    # --- peaks ---
    peaks_idx, _ = detect_peaks(residual_pos, **peaks_cfg)
    peaks_idx = peaks_idx.astype(int) if hasattr(peaks_idx, "astype") else np.asarray(peaks_idx, dtype=int)

    # --- async metrics freq fallback (FIXED: helper is async) ---
    async def _metrics_freq_fallback() -> float:
        # local overlay first
        meta_path = run_dir / "output" / "overlay" / "tracks.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                tid_num = int(track_id) if track_id.isdigit() else track_id
                for tr in meta.get("tracks", []):
                    if tr.get("id") == tid_num or str(tr.get("id")) == str(tid_num):
                        m = (tr.get("metrics") or {})
                        f = m.get("dominant_frequency") or m.get("freq") or None
                        if f is not None:
                            return float(f)
            except Exception:
                pass
        # remote overlay if storage is remote
        if use_remote:
            try:
                k = run_key(run_id, "output", "overlay", "tracks.json")
                blob = await runs_storage.get(k)
                meta = json.loads(blob.decode("utf-8"))
                tid_num = int(track_id) if track_id.isdigit() else track_id
                for tr in meta.get("tracks", []):
                    if tr.get("id") == tid_num or str(tr.get("id")) == str(tid_num):
                        m = (tr.get("metrics") or {})
                        f = m.get("dominant_frequency") or m.get("freq") or None
                        if f is not None:
                            return float(f)
            except Exception:
                pass
        return float("nan")

    # --- frequency / period ---
    if freq_source == "metrics":
        freq = await _metrics_freq_fallback()
        if not (isinstance(freq, float) and math.isfinite(freq) and freq > 0):
            try:
                freq = float(estimate_dominant_frequency(residual_pos, **period_cfg))
            except Exception:
                freq = float("nan")
    else:
        try:
            freq = float(estimate_dominant_frequency(residual_pos, **period_cfg))
        except Exception:
            freq = float("nan")

    period = float(frequency_to_period(freq)) if (isinstance(freq, float) and math.isfinite(freq) and freq > 0) \
             else float("nan")

    # strongest residual peak for phase anchoring
    strongest_peak_idx: Optional[int] = None
    if peaks_idx.size > 0:
        try:
            strongest_peak_idx = int(peaks_idx[int(np.argmax(residual_pos[peaks_idx]))])
        except Exception:
            strongest_peak_idx = int(peaks_idx[0])

    # --- optional phase-anchored sine overlay (keep helper signature) ---
    sine_fit_pos: Optional[np.ndarray] = None
    if include_sine and isinstance(freq, float) and math.isfinite(freq) and freq > 0:
        yfit_res, A, phi, c = _fit_global_sine(
            residual_pos, time_index, sampling_rate, freq, center_peak_idx=strongest_peak_idx
        )
        if yfit_res is not None:
            sine_fit_pos = (baseline_pos + yfit_res).astype(float)

    # --- optional slicing ---
    lo, hi = 0, N - 1
    if index_range:
        lo, hi = _parse_index_range(index_range, N)  # may raise 400
        time_index_view = time_index[lo : hi + 1]
        baseline_view = baseline_pos[lo : hi + 1]
        residual_view = residual_pos[lo : hi + 1] if include_residual else None
        sine_view = sine_fit_pos[lo : hi + 1] if sine_fit_pos is not None else None
        peaks_in_slice: List[int] = [int(i) for i in peaks_idx.tolist() if lo <= int(i) <= hi]
    else:
        time_index_view = time_index
        baseline_view = baseline_pos
        residual_view = residual_pos if include_residual else None
        sine_view = sine_fit_pos
        peaks_in_slice = peaks_idx.tolist()

    # --- metrics (same definition as pipeline) ---
    if peaks_idx.size > 0:
        try:
            mean_amp = float(residual_pos[peaks_idx].mean())
        except Exception:
            mean_amp = float("nan")
    else:
        mean_amp = float("nan")

    out = {
        "id": str(track_id),
        "sample": sample_name,
        "coords": {"poly_format": "[y, x]", "x_name": "position_px", "y_name": "time_row"},
        "time_index": time_index_view.tolist(),
        "baseline": baseline_view.tolist(),
        "residual": (residual_view.tolist() if residual_view is not None else None),
        "sine_fit": (sine_view.tolist() if sine_view is not None else None),
        "regression": {"method": "ransac_poly", "degree": degree, "params": detrend_cfg},
        "peaks": [int(i) for i in peaks_idx.tolist()],
        "peaks_in_slice": [int(i) for i in peaks_in_slice],
        "strongest_peak_idx": strongest_peak_idx,
        "metrics": {
            "dominant_frequency": freq if math.isfinite(freq) else None,
            "period": period if math.isfinite(period) else None,
            "num_peaks": int(len(peaks_idx)),
            "mean_amplitude": mean_amp if math.isfinite(mean_amp) else None,
        },
    }
    if index_range:
        out["slice"] = {"lo": lo, "hi": hi}

    return _sanitize_for_json(out)

@app.post("/api/uploads/sign", dependencies=[Depends(require_run_view)])
async def sign_upload(payload: dict = Body(...)):
    filename = payload.get("filename")
    content_type = payload.get("content_type", "application/octet-stream")
    if not filename:
        raise HTTPException(400, "filename required")
    key = f"staged/{filename}"
    client = storage.Client()
    bucket = client.bucket(os.environ["GCS_BUCKET"])  # set this env at deploy
    blob = bucket.blob(key)
    url = blob.generate_signed_url(
        version="v4", method="PUT",
        content_type=content_type, expiration=timedelta(minutes=15)
    )
    return {"key": key, "url": url}

# run with `python -m src.service.api`
if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.getenv("PORT", "8080"))  # Cloud Run sets PORT=8080
    reload_dev = os.getenv("DEV_RELOAD", "0") == "1"

    uvicorn.run("src.service.api:app", host="0.0.0.0", port=port, reload=reload_dev, log_level="info")
