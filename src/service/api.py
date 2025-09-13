# src/service/api.py
from __future__ import annotations
from collections import deque
from contextlib import asynccontextmanager
import glob
import logging
import math
import os

import aiofiles
import numpy as np

from ..signal.detrend import fit_baseline_ransac
from ..signal.peaks import detect_peaks
from ..signal.period import estimate_dominant_frequency, frequency_to_period
from ..visualize import _fit_global_sine

from ..utils import load_config
# Ensure headless plotting in worker threads (must be set BEFORE any Matplotlib import anywhere)
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

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

# Pipeline generator (emits JobEvent)
from .pipeline import _sample_name_from_arr_path, iter_run_project, JobEvent


# --- SSE tuning (env-configurable) ---
# Set SSE_PING_SECONDS=0 in dev to disable heartbeats (reduces socket.send noise).
SSE_PING_SECONDS = int(os.getenv("SSE_PING_SECONDS", "25"))  # 0 → disables pings
SSE_LOG_SUBS = os.getenv("SSE_LOG_SUBS", "0") == "1"         # 1 → log +sub/-sub

logger = logging.getLogger("wavecalling.sse")

# ---- Config ----
APP_ROOT = Path(__file__).resolve().parents[2]  # repo root
DEFAULT_CONFIG = APP_ROOT / "configs" / "default.yaml"
WEB_DIR = APP_ROOT / "web"
RUNS_ROOT = APP_ROOT / "runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)


# -----------------------
# Persistence helpers
# -----------------------
def _run_dir(run_id: str) -> Path:
    return RUNS_ROOT / run_id

def _events_path(run_id: str) -> Path:
    return _run_dir(run_id) / "events.ndjson"

def _run_json_path(run_id: str) -> Path:
    return _run_dir(run_id) / "run.json"

def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)

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
    for d in RUNS_ROOT.iterdir():
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


# ---- FastAPI app ----
app = FastAPI(title="WaveCalling API", version="0.2.0", lifespan=lifespan)

ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*").split(",")
app.add_middleware(CORSMiddleware, allow_origins=ALLOW_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Expose the runs directory for downloading artifacts (CSV/plots/overlay)
app.mount("/runs", StaticFiles(directory=str(RUNS_ROOT)), name="runs")

# Static smoke-test UI (keep during dev; React/Vite build will be mounted at /app later)
app.mount("/ui", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


# -----------------------
# In-memory run registry
# -----------------------
class RunInfo(BaseModel):
    run_id: str
    name: str
    created_at: str
    status: str                 # QUEUED | RUNNING | DONE | ERROR | CANCELLED
    error: Optional[str] = None
    input_dir: str
    output_dir: str
    plots_dir: Optional[str] = None
    config_path: str


class _RunState:
    """
    Internal state for each run: dirs, subscribers, status, and persistence.
    Persists run.json and events.ndjson so UI can list/replay after refresh.
    """
    def __init__(self, run_id: str, name: str, base_dir: Path, config_path: Path,
                 created_at_iso: Optional[str] = None, status: str = "QUEUED", error: Optional[str] = None):
        self.run_id = run_id
        self.name = name
        self.base_dir = base_dir
        self.input_dir = base_dir / "input"
        self.output_dir = base_dir / "output"
        self.plots_dir = base_dir / "plots"
        self.config_path = config_path

        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.created_at = (
            datetime.fromisoformat(created_at_iso.replace("Z", "+00:00"))
            if created_at_iso else datetime.now(timezone.utc)
        )
        self.status = status
        self.error: Optional[str] = error

        self.cancel_event: asyncio.Event = asyncio.Event()
        self.worker_task: Optional[asyncio.Task] = None

        self._subscribers: List[asyncio.Queue] = []
        self._sub_lock = asyncio.Lock()

        # Persist initial snapshot
        self._persist_run_json()

        # versioning for overlay changes
        self.overlay_version: int = 0
        self._overlay_mtime_ns: int = 0  # last seen mtime for whichever overlay file is active

    def subscribers_count(self) -> int:
        return len(self._subscribers)

    # ---------- persistence ----------
    def _persist_run_json(self) -> None:
        payload = {
            "run_id": self.run_id,
            "name": self.name,
            "created_at": self.created_at.isoformat().replace("+00:00", "Z"),
            "status": self.status,
            "error": self.error,
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "plots_dir": str(self.plots_dir),
            "config_path": str(self.config_path),
        }
        _write_json(_run_json_path(self.run_id), payload)

    def _persist_event(self, evt: JobEvent) -> None:
        try:
            _append_event(self.run_id, evt)
        except Exception:
            # best-effort; don't crash on fs hiccups
            pass

    # ---------- public API ----------
    def info(self) -> RunInfo:
        return RunInfo(
            run_id=self.run_id,
            name=self.name,
            created_at=self.created_at.isoformat().replace("+00:00", "Z"),
            status=self.status,
            error=self.error,
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir),
            plots_dir=str(self.plots_dir),
            config_path=str(self.config_path),
        )

    async def add_subscriber(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        async with self._sub_lock:
            self._subscribers.append(q)
        return q

    async def remove_subscriber(self, q: asyncio.Queue) -> None:
        async with self._sub_lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    async def publish(self, evt: JobEvent) -> None:
        # persist then fan-out
        self._persist_event(evt)
        async with self._sub_lock:
            for q in list(self._subscribers):
                try:
                    q.put_nowait(evt)
                except asyncio.QueueFull:
                    pass

    def set_status(self, status: str, error: Optional[str] = None) -> None:
        self.status = status
        if error is not None:
            self.error = error
        self._persist_run_json()

    # ---------- overlay versioning ----------
    def maybe_bump_overlay_version(self) -> bool:
        """Detect if overlay output changed (by mtime) and bump version."""
        overlay_dir = self.output_dir / "overlay"
        if not overlay_dir.exists():
            return False
        # prefer final > partial > ndjson
        cand = [
            overlay_dir / "tracks.json",
            overlay_dir / "tracks.partial.json",
            overlay_dir / "tracks.ndjson",
        ]
        p = next((x for x in cand if x.exists()), None)
        if not p:
            return False
        try:
            mtime_ns = p.stat().st_mtime_ns
        except FileNotFoundError:
            return False
        if mtime_ns > self._overlay_mtime_ns:
            self._overlay_mtime_ns = mtime_ns
            self.overlay_version += 1
            return True
        return False


# Global registry (in-memory)
_RUNS: Dict[str, _RunState] = {}
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

# -----------------------
# Models / payloads
# -----------------------
class CreateRunResponse(BaseModel):
    run_id: str
    status: str
    info: RunInfo

class RunStatusResponse(BaseModel):
    info: RunInfo
    artifacts: Dict[str, str] = Field(
        default_factory=dict,
        description="Paths (URLs) to main artifacts once available",
    )


# -----------------------
# Helpers
# -----------------------

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

def _artifact_urls_by_state(state: _RunState) -> Dict[str, str]:
    base = f"/runs/{state.run_id}"
    return {
        "tracks_csv": f"{base}/output/metrics.csv",
        "waves_csv": f"{base}/output/metrics_waves.csv",
        "overlay_json": f"{base}/output/overlay/tracks.json",
        "overlay_json_partial": f"{base}/output/overlay/tracks.partial.json",
        "overlay_ndjson": f"{base}/output/overlay/tracks.ndjson",
        "run_json": f"{base}/run.json",
        "events_ndjson": f"{base}/events.ndjson",
        "progress_json": f"{base}/output/progress.json",
        "plots_dir": f"{base}/plots",
        "output_dir": f"{base}/output",
        "base_image": f"{base}/output/base.png",
    }

def _artifact_urls_by_id(run_id: str) -> Dict[str, str]:
    base = f"/runs/{run_id}"
    return {
        "tracks_csv": f"{base}/output/metrics.csv",
        "waves_csv": f"{base}/output/metrics_waves.csv",
        "overlay_json": f"{base}/output/overlay/tracks.json",
        "overlay_json_partial": f"{base}/output/overlay/tracks.partial.json",
        "overlay_ndjson": f"{base}/output/overlay/tracks.ndjson",
        "run_json": f"{base}/run.json",
        "events_ndjson": f"{base}/events.ndjson",
        "progress_json": f"{base}/output/progress.json",
        "plots_dir": f"{base}/plots",
        "output_dir": f"{base}/output",
        "base_image": f"{base}/output/base.png",
    }

async def _run_pipeline(state: _RunState, config_overrides: Optional[dict], verbose: bool) -> None:
    state.set_status("RUNNING")
    loop = asyncio.get_running_loop()
    bridge_q: asyncio.Queue[JobEvent] = asyncio.Queue()
    terminal_seen = False

    def _runner_sync():
        got_terminal = False
        try:
            for evt in iter_run_project(
                input_dir=state.input_dir,
                config_path=state.config_path,
                output_dir=state.output_dir,
                plots_out=state.plots_dir,
                progress_cb=None,
                config_overrides=config_overrides,
                verbose=verbose,
                cancel_cb=lambda: state.cancel_event.is_set(),
            ):
                if evt.phase in ("DONE", "ERROR", "CANCELLED"):
                    got_terminal = True
                asyncio.run_coroutine_threadsafe(bridge_q.put(evt), loop)
        except Exception as e:
            asyncio.run_coroutine_threadsafe(
                bridge_q.put(JobEvent(phase="ERROR", message=str(e), progress=1.0)),
                loop,
            )
        finally:
            if not got_terminal:
                asyncio.run_coroutine_threadsafe(
                    bridge_q.put(JobEvent(phase="DONE", message="Run finished", progress=1.0)),
                    loop,
                )

    state.worker_task = asyncio.create_task(asyncio.to_thread(_runner_sync))

    try:
        while True:
            evt: JobEvent = await bridge_q.get()
            if evt.phase in ("ERROR", "DONE", "CANCELLED"):
                terminal_seen = True
                state.set_status(evt.phase, error=(evt.message if evt.phase == "ERROR" else None))
            await state.publish(evt)
            if evt.phase in ("ERROR", "DONE", "CANCELLED"):
                break

        if state.worker_task:
            await state.worker_task

    except Exception as e:
        state.set_status("ERROR", error=str(e))
        await state.publish(JobEvent(phase="ERROR", message=str(e), progress=1.0))
    finally:
        if not terminal_seen:
            terminal = state.status if state.status in ("DONE", "ERROR", "CANCELLED") else "ERROR"
            await state.publish(JobEvent(phase=terminal, message="Run finished", progress=1.0))

def _synthesize_info_from_dir(d: Path) -> Optional[RunInfo]:
    rj = _run_json_path(d.name)
    meta = _load_run_json(rj) or {}
    if not meta and not d.is_dir():
        return None
    created = meta.get("created_at") or datetime.fromtimestamp(
        d.stat().st_mtime, tz=timezone.utc
    ).isoformat().replace("+00:00", "Z")

    run_id = meta.get("run_id") or d.name
    name = meta.get("name") or f"run-{run_id}"
    status = meta.get("status") or "DONE"
    input_dir = meta.get("input_dir") or str(d / "input")
    output_dir = meta.get("output_dir") or str(d / "output")
    plots_dir = meta.get("plots_dir") or str(d / "plots")
    config_path = meta.get("config_path") or str(d / "config.yaml")

    return RunInfo(
        run_id=run_id, name=name, created_at=created, status=status,
        error=meta.get("error"), input_dir=input_dir, output_dir=output_dir,
        plots_dir=plots_dir, config_path=config_path
    )


# -----------------------
# Endpoints
# -----------------------

@app.post("/api/runs", response_model=CreateRunResponse)
async def create_run(
    files: List[UploadFile] = File(..., description="One or more CSV/XLS/PNG/JPG"),
    run_name: Optional[str] = Form(None),
    config_overrides: Optional[str] = Form(None, description="JSON string with partial overrides to default YAML"),
    verbose: bool = Form(False),
):
    """
    Create a new run:
      - Saves uploads under runs/<run_id>/input
      - Copies default config next to the run (traceability)
      - Starts the pipeline in background
      - Returns run_id immediately
    """
    run_id = uuid.uuid4().hex[:10]
    base_dir = _run_dir(run_id)
    (base_dir / "input").mkdir(parents=True, exist_ok=True)

    # Save inputs
    saved = await _save_uploads(files, base_dir / "input")
    if not saved:
        raise HTTPException(status_code=400, detail="No files uploaded")

    # Copy config
    if not DEFAULT_CONFIG.exists():
        raise HTTPException(status_code=500, detail=f"Default config not found: {DEFAULT_CONFIG}")
    config_path = base_dir / "config.yaml"
    shutil.copyfile(DEFAULT_CONFIG, config_path)

    # Parse overrides (stored only in memory; the generator consumes them)
    overrides: Optional[dict] = None
    if config_overrides:
        try:
            overrides = json.loads(config_overrides)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid config_overrides JSON: {e}")

    # Create state, persist run.json
    state = _RunState(
        run_id=run_id,
        name=run_name or f"run-{run_id}",
        base_dir=base_dir,
        config_path=config_path,
        created_at_iso=_iso_now(),
        status="QUEUED",
    )
    _RUNS[run_id] = state

    # Kick off worker task (async)
    asyncio.create_task(_run_pipeline(state, overrides, verbose))

    # notify dashboards to refresh run list
    await _publish_runs_dirty()

    return CreateRunResponse(run_id=run_id, status=state.status, info=state.info())


@app.get("/api/runs", response_model=List[RunInfo])
async def list_runs():
    # disk scan
    disk: dict[str, RunInfo] = {}
    for d in RUNS_ROOT.iterdir():
        if d.is_dir():
            info = _synthesize_info_from_dir(d)
            if info:
                disk[info.run_id] = info

    # live overrides (authoritative for RUNNING/QUEUED)
    for run_id, st in _RUNS.items():
        disk[run_id] = st.info()

    infos = list(disk.values())
    infos.sort(key=lambda r: r.created_at, reverse=True)
    return infos


@app.get("/api/runs/{run_id}", response_model=RunStatusResponse)
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

@app.get("/api/runs/{run_id}/events")
async def stream_events(run_id: str, replay: int = Query(0, description=">1 = last N historical events first")):
    """
    Server-Sent Events stream of JobEvent for the given run.
    Browser usage:
      const es = new EventSource(`/api/runs/${runId}/events?replay=100`);
      es.onmessage = (e) => { const evt = JSON.parse(e.data); ... }
    """
    st = _RUNS.get(run_id)
    if not st:
        raise HTTPException(status_code=404, detail="Unknown run_id")

    q = await st.add_subscriber()
    if SSE_LOG_SUBS:
        try:
            logger.info("[SSE] +sub %s subs=%d", run_id, st.subscribers_count())
        except Exception:
            logger.info("[SSE] +sub %s", run_id)

    def _to_url(p: str) -> str:
        """Map a local file path to its served URL under /runs/<id>/..."""
        try:
            p_path = Path(p)
            # output dir
            try:
                rel = p_path.relative_to(st.output_dir)
                return f"/runs/{st.run_id}/output/{rel.as_posix()}"
            except ValueError:
                pass
            # plots dir
            try:
                rel = p_path.relative_to(st.plots_dir)
                return f"/runs/{st.run_id}/plots/{rel.as_posix()}"
            except ValueError:
                pass
            # base dir as a fallback
            try:
                rel = p_path.relative_to(st.base_dir)
                return f"/runs/{st.run_id}/{rel.as_posix()}"
            except ValueError:
                pass
        except Exception:
            pass
        return p  # unknown root or already a URL

    def _rewrite_payload(payload: dict) -> dict:
        extra = payload.get("extra")
        if isinstance(extra, dict):
            new_extra = {}
            for k, v in extra.items():
                if isinstance(v, str):
                    new_extra[k] = _to_url(v)
                elif isinstance(v, list):
                    new_extra[k] = [_to_url(x) if isinstance(x, str) else x for x in v]
                else:
                    new_extra[k] = v
            payload["extra"] = new_extra
        return payload

    async def gen():
        try:
            # Initial snapshot
            yield {"event": "message", "data": json.dumps({"phase": st.status, "message": "subscribed", "progress": 0.0})}

            # OPTIONAL REPLAY: stream historical events from NDJSON first
            if replay and replay > 0:
                nd = _events_path(st.run_id)
                if nd.exists():
                    try:
                        with nd.open() as f:
                            lines_iter = f if replay == 1 else deque(f, maxlen=replay)
                            for line in lines_iter:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    payload = json.loads(line)
                                    payload = _rewrite_payload(payload)
                                    yield {"event": "message", "data": json.dumps(payload)}
                                except Exception:
                                    continue
                    except Exception:
                        # ignore replay errors, continue with live
                        pass

            # Now live events
            while True:
                evt: JobEvent = await q.get()
                payload = asdict(evt)
                payload = _rewrite_payload(payload)
                yield {"event": "message", "data": json.dumps(payload)}
                if evt.phase in ("DONE", "ERROR", "CANCELLED"):
                    break
        except asyncio.CancelledError:
            # client disconnected mid-stream (normal)
            return
        finally:
            await st.remove_subscriber(q)
            if SSE_LOG_SUBS:
                try:
                    logger.info("[SSE] -sub %s subs=%d", run_id, st.subscribers_count())
                except Exception:
                    logger.info("[SSE] -sub %s", run_id)

    # Response headers: avoid buffering and caching
    headers = {
        "Cache-Control": "no-store",
        "X-Accel-Buffering": "no",  # helps with nginx (no buffering of event stream)
    }

    # Set ping to env value; 0 disables heartbeats (quiet dev logs)
    ping_value = None if SSE_PING_SECONDS <= 0 else SSE_PING_SECONDS

    return EventSourceResponse(gen(), ping=ping_value, headers=headers)

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

@app.get("/api/runs/{run_id}/overlay")
async def get_overlay(run_id: str, request: Request):
    st = _RUNS.get(run_id)
    base = _run_dir(run_id)
    if not (st or base.exists()):
        raise HTTPException(status_code=404, detail="Unknown run_id")
    output_dir = (st.output_dir if st else (base / "output"))
    final = Path(output_dir) / "overlay" / "tracks.json"
    partial = Path(output_dir) / "overlay" / "tracks.partial.json"
    ndjson = Path(output_dir) / "overlay" / "tracks.ndjson"

    target = next((p for p in (final, partial, ndjson) if p.exists()), None)
    if not target:
        raise HTTPException(status_code=404, detail="Overlay not available yet")

    etag = _weak_file_etag(target)
    inm = request.headers.get("if-none-match")
    if inm and inm == etag:
        return Response(status_code=304, headers={"ETag": etag})

    if target == ndjson:
        tracks = []
        with target.open() as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                tracks.append(_sanitize_for_json(obj))
        body = json.dumps({"version": 1, "tracks": tracks}).encode("utf-8")
    else:
        with target.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Invalid overlay JSON: {e}")
        body = json.dumps(_sanitize_for_json(data)).encode("utf-8")

    return Response(content=body, media_type="application/json", headers={"ETag": etag})


@app.get("/api/runs/{run_id}/image")
async def get_base_image(run_id: str):
    """Serve the base image used for overlay (copied by the pipeline as base.png)."""
    st = _RUNS.get(run_id)
    base = _run_dir(run_id)
    if not (st or base.exists()):
        raise HTTPException(status_code=404, detail="Unknown run_id")
    output_dir = Path(st.output_dir) if st else (base / "output")
    img = output_dir / "base.png"
    if not img.exists():
        raise HTTPException(status_code=404, detail="Base image not available yet")
    return FileResponse(str(img), media_type="image/png")


@app.get("/api/runs/{run_id}/waves")
async def list_wave_windows(run_id: str, track: Optional[str] = None):
    """List PNGs under plots/<track>/peak_windows (as /runs URLs)."""
    st = _RUNS.get(run_id)
    base = _run_dir(run_id)
    if not (st or base.exists()):
        raise HTTPException(status_code=404, detail="Unknown run_id")
    if not track:
        raise HTTPException(status_code=400, detail="Query param 'track' is required")
    plots_dir = Path(st.plots_dir) if st else (base / "plots")
    win_dir = plots_dir / str(track) / "peak_windows"
    if not win_dir.exists():
        return JSONResponse({"images": []})
    files = sorted([p for p in win_dir.glob("*.png")])
    urls = [f"/runs/{run_id}/plots/{track}/peak_windows/{p.name}" for p in files]
    return JSONResponse({"images": urls})


@app.get("/api/runs/{run_id}/tracks")
async def list_tracks(run_id: str):
    """
    List per-track .npy files with URLs.
    Preference: /runs/<id>/output/tracks/*.npy
    Fallback: search for **/kymobutler_output/*.npy within the run directory.
    """
    base = _run_dir(run_id)
    if not base.exists():
        raise HTTPException(status_code=404, detail="Unknown run_id")

    preferred = base / "output" / "tracks"
    paths: List[Path] = []
    if preferred.exists():
        paths = sorted(preferred.glob("*.npy"))

    # fallback search
    if not paths:
        paths = sorted(base.glob("**/kymobutler_output/*.npy"))

    items = [
        {"id": p.stem, "url": f"/runs/{run_id}/{p.relative_to(base).as_posix()}"}
        for p in paths
    ]
    return {"count": len(items), "tracks": items}


@app.post("/api/runs/{run_id}/cancel")
async def cancel_run(run_id: str):
    """
    Signal a running job to stop early. The pipeline should:
      - Check cancel_cb() at safe points,
      - Write partial artifacts,
      - Emit a CANCELLED terminal event.
    """
    st = _RUNS.get(run_id)
    if not st:
        raise HTTPException(status_code=404, detail="Unknown run_id")
    if st.status not in ("QUEUED", "RUNNING"):
        return {"run_id": run_id, "status": st.status, "message": "Run is not active"}

    st.cancel_event.set()
    return {"run_id": run_id, "status": "CANCEL_REQUESTED"}


@app.post("/api/runs/{run_id}/resume")
async def resume_run(run_id: str, verbose: bool = Form(False)):
    """
    Basic resume: re-run the pipeline for this run_id using existing inputs/config.
    (Your pipeline already prefers existing .npy → will skip Kymo phase.)
    """
    st = _RUNS.get(run_id)
    if not st:
        raise HTTPException(status_code=404, detail="Unknown run_id")

    if st.status == "RUNNING":
        return {"run_id": run_id, "status": st.status, "message": "Run is already running"}

    if st.cancel_event.is_set():
        st.cancel_event.clear()

    asyncio.create_task(_run_pipeline(st, config_overrides=None, verbose=verbose))
    return {"run_id": run_id, "status": "RUNNING"}


@app.get("/api/runs/{run_id}/progress")
async def get_progress(run_id: str, request: Request) -> Response:
    """
    Return pipeline progress:
      {
        "totalTracks": int|null,
        "processedCount": int,
        "skippedCount": int|null,
        "lastUpdatedAt": iso8601|null,
        "source": "file" | "synthesized"
      }
    """
    st = _RUNS.get(run_id)

    # Preferred: with a live state (accurate config resolution)
    if st:
        progress_path, marker_dirs = _resolve_progress_paths_for_state(st)
    else:
        # Disk fallback (after restart)
        base = _run_dir(run_id)
        if not base.exists():
            raise HTTPException(status_code=404, detail="Unknown run_id")
        progress_path, marker_dirs = _resolve_progress_paths_for_disk(run_id)

    payload: Optional[dict] = None

    # 1) Preferred: read progress.json
    if progress_path.exists():
        try:
            with progress_path.open() as f:
                data = json.load(f)
            data.setdefault("source", "file")
            payload = data
        except Exception:
            # fall through to synthesized
            payload = None

    # 2) Synthesized fallback using marker dirs + events.ndjson
    if payload is None:
        processed_count = 0
        for md in marker_dirs:
            if md.exists():
                try:
                    processed_count = max(processed_count, sum(1 for _ in md.glob("*.done")))
                except Exception:
                    continue

        total_tracks: Optional[int] = None
        skipped_count: Optional[int] = None
        last_updated: Optional[str] = None

        nd = _events_path(run_id)
        if nd.exists():
            try:
                max_total_tracks = 0
                with nd.open() as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            evt = json.loads(line)
                            if isinstance(evt.get("extra"), dict):
                                extra = evt["extra"]
                                if "total" in extra and isinstance(extra["total"], int):
                                    max_total_tracks = max(max_total_tracks, int(extra["total"]))
                                if "total_tracks" in extra and isinstance(extra["total_tracks"], int):
                                    max_total_tracks = max(max_total_tracks, int(extra["total_tracks"]))
                            last_updated = evt.get("ts") or last_updated
                        except Exception:
                            continue
                total_tracks = max_total_tracks or None
            except Exception:
                pass

        payload = {
            "totalTracks": total_tracks if total_tracks is not None else (processed_count if processed_count else None),
            "processedCount": int(processed_count),
            "skippedCount": skipped_count,
            "lastUpdatedAt": last_updated,
            "source": "synthesized",
        }

    # ---- ETag / If-None-Match handling ----
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    etag = _sha1_etag_bytes(body)

    inm = request.headers.get("if-none-match")
    if inm:
        # Handle multiple ETags and weak validators
        def _norm(tag: str) -> str:
            t = tag.strip()
            if t.startswith("W/"):
                t = t[2:].strip()
            return t
        client_tags = [_norm(t) for t in inm.split(",")]
        if etag in client_tags or "*" in client_tags:
            # Not modified; return headers but no body
            return Response(status_code=304, headers={"ETag": etag})

    return Response(
        content=body,
        media_type="application/json",
        headers={
            "ETag": etag,
            "Cache-Control": "private, must-revalidate",
        },
    )


@app.get("/api/health")
def health():
    return {"ok": True, "version": app.version}


@app.delete("/api/runs/{run_id}")
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

@app.get("/api/runs/{run_id}/snapshot")
async def get_snapshot(run_id: str, request: Request):
    """
    Authoritative, compact state for a run:
      - status, error
      - overlay_version (monotone)
      - artifact existence flags
      - (optional) tiny progress summary when available
    """
    st = _RUNS.get(run_id)
    # allow disk fallback so snapshots work after restart
    if not st:
        d = _run_dir(run_id)
        if not d.exists():
            raise HTTPException(status_code=404, detail="Unknown run_id")
        info = _synthesize_info_from_dir(d)
        if not info:
            raise HTTPException(status_code=404, detail="Unknown run_id")
        # cheap overlay version from mtime
        overlay_dir = d / "output" / "overlay"
        ov_paths = [overlay_dir / "tracks.json", overlay_dir / "tracks.partial.json", overlay_dir / "tracks.ndjson"]
        try:
            mtime_ns = max((p.stat().st_mtime_ns for p in ov_paths if p.exists()), default=0)
        except Exception:
            mtime_ns = 0
        overlay_version = 1 if mtime_ns else 0
        paths = _artifact_paths_by_state(_RunState(run_id=info.run_id, name=info.name, base_dir=d,
                                                   config_path=Path(info.config_path),
                                                   created_at_iso=info.created_at, status=info.status))
        flags = {k: p.exists() for k, p in paths.items()}
        payload = {
            "run_id": info.run_id,
            "status": info.status,
            "error": info.error,
            "overlay_version": overlay_version,
            "artifacts": flags,
        }
    else:
        # live run
        st.maybe_bump_overlay_version()
        paths = _artifact_paths_by_state(st)
        flags = {k: p.exists() for k, p in paths.items()}
        # try to include tiny progress if file exists (no heavy synth here)
        prog_path, _ = _resolve_progress_paths_for_state(st)
        prog = None
        if prog_path.exists():
            try:
                with prog_path.open() as f:
                    prog = json.load(f)
            except Exception:
                prog = None
        payload = {
            "run_id": st.run_id,
            "status": st.status,
            "error": st.error,
            "overlay_version": st.overlay_version,
            "artifacts": flags,
            "progress": prog if isinstance(prog, dict) else None,
        }

    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    etag = _sha1_etag_bytes(body)
    inm = request.headers.get("if-none-match")
    if inm and inm == etag:
        return Response(status_code=304, headers={"ETag": etag})
    return Response(content=body, media_type="application/json", headers={"ETag": etag})


@app.get("/api/runs/{run_id}/tracks/{track_id}")
def get_track_detail(
    run_id: str,
    track_id: str,
    include_sine: bool = Query(False, description="Include phase-anchored sine overlay (baseline + fitted residual)."),
    include_residual: bool = Query(False, description="Include residual array (position - baseline)."),
    index_range: Optional[str] = Query(None, alias="range", description="Optional index window 'lo:hi' (inclusive)."),
    freq_source: Literal["auto", "metrics"] = Query("auto", description="Use 'auto' (recompute) or 'metrics' (overlay) frequency."),
):
    """
    Return exact analysis (parity with CLI):
      • Baseline from RANSAC polynomial (same degree/kwargs as config)
      • Residual = position - baseline (optional)
      • Optional phase-anchored sine overlay (anchored at strongest residual peak)
      • Explicit time_index and coordinate metadata
    Arrays align 1:1 with the original .npy order (poly[i] = [y_row, x_pos]).

    If `range=lo:hi` is provided, arrays are sliced and we return:
      - `slice: {lo, hi}`
      - `peaks_in_slice` = peak indices that fall inside the window
    """
    run_dir = RUNS_ROOT / run_id
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise HTTPException(status_code=404, detail="config.yaml not found for run")

    cfg = load_config(cfg_path)

    # Locate the .npy for this track id
    npy: Optional[Path] = None
    for p in run_dir.rglob(f"{track_id}.npy"):
        npy = p
        break
    if npy is None:
        raise HTTPException(status_code=404, detail="track not found")

    # ----- Load data -----
    xy = np.load(npy)  # shape (N,2), [[y,row],[x,col]]
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise HTTPException(status_code=500, detail="Invalid track array shape")
    time_index = xy[:, 0].astype(float)   # rows (time)
    position = xy[:, 1].astype(float)     # cols (position)
    N = len(time_index)

    # ----- Config -----
    detrend_cfg = (cfg.get("detrend") or {}).copy()
    degree = int(detrend_cfg.pop("degree", 1))
    peaks_cfg = cfg.get("peaks", {}) or {}
    period_cfg = (cfg.get("period", {}) or {}).copy()
    io_cfg = cfg.get("io", {}) or {}
    sampling_rate = float(io_cfg.get("sampling_rate", period_cfg.get("sampling_rate", 1.0)))
    period_cfg.setdefault("sampling_rate", sampling_rate)

    # ----- Baseline via RANSAC (parity with CLI) -----
    # Regress position ~ poly(time_index) with RANSAC
    model = fit_baseline_ransac(time_index, position, degree=degree, **detrend_cfg)
    baseline_pos = model.predict(time_index.reshape(-1, 1)).astype(float)
    residual_pos = (position - baseline_pos).astype(float)

    # ----- Peaks -----
    peaks_idx, _ = detect_peaks(residual_pos, **peaks_cfg)
    peaks_idx = peaks_idx.astype(int) if hasattr(peaks_idx, "astype") else np.asarray(peaks_idx, dtype=int)

    # ----- Frequency / period -----
    if freq_source == "metrics":
        freq = _lookup_metrics_freq(run_dir, track_id) or float("nan")
        if not (isinstance(freq, float) and math.isfinite(freq) and freq > 0):
            # fallback to computed if metrics missing/invalid
            try:
                freq = float(estimate_dominant_frequency(residual_pos, **period_cfg))
            except Exception:
                freq = float("nan")
    else:
        try:
            freq = float(estimate_dominant_frequency(residual_pos, **period_cfg))
        except Exception:
            freq = float("nan")

    period = float(frequency_to_period(freq)) if (isinstance(freq, float) and math.isfinite(freq) and freq > 0) else float("nan")

    # Strongest residual peak for phase anchoring
    strongest_peak_idx: Optional[int] = None
    if peaks_idx.size > 0:
        try:
            strongest_peak_idx = int(peaks_idx[int(np.argmax(residual_pos[peaks_idx]))])
        except Exception:
            strongest_peak_idx = int(peaks_idx[0])

    # ----- Optional phase-anchored sine overlay -----
    sine_fit_pos: Optional[np.ndarray] = None
    if include_sine and math.isfinite(freq) and freq > 0:
        yfit_res, A, phi, c = _fit_global_sine(
            residual_pos, time_index, sampling_rate, freq, center_peak_idx=strongest_peak_idx
        )
        if yfit_res is not None:
            sine_fit_pos = (baseline_pos + yfit_res).astype(float)

    # ----- Optional slicing -----
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

    # ----- Metrics (same definition as pipeline) -----
    if peaks_idx.size > 0:
        try:
            mean_amp = float(residual_pos[peaks_idx].mean())
        except Exception:
            mean_amp = float("nan")
    else:
        mean_amp = float("nan")

    out = {
        "id": str(track_id),
        "sample": _sample_name_from_arr_path(npy),
        "coords": {"poly_format": "[y, x]", "x_name": "position_px", "y_name": "time_row"},
        "time_index": time_index_view.tolist(),           # time rows aligned to array indices
        "baseline": baseline_view.tolist(),               # predicted position per index
        "residual": (residual_view.tolist() if residual_view is not None else None),
        "sine_fit": (sine_view.tolist() if sine_view is not None else None),
        "regression": {
            "method": "ransac_poly",
            "degree": degree,
            "params": detrend_cfg,                        # kwargs used (sans degree)
        },
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

@app.get("/api/runs/{run_id}/debug/{layer}")
async def get_debug_image(run_id: str, layer: str):
    """Serve debug layer PNGs; search a few likely locations."""
    base = _run_dir(run_id)
    if not base.exists():
        raise HTTPException(status_code=404, detail="Unknown run_id")

    # 1) Preferred (published) location
    p1 = base / "output" / f"{layer}.png"
    if p1.exists():
        return FileResponse(str(p1), media_type="image/png")

    # 2) Fallbacks (older/native layouts)
    for pat in [
        base / "output" / "overlay" / "debug" / "*" / f"{layer}.png",
        base / "input" / "generated_heatmaps" / "*" / "debug" / f"{layer}.png",
        base / "input" / "debug" / f"{layer}.png",
    ]:
        matches = glob.glob(str(pat))
        if matches:
            return FileResponse(matches[0], media_type="image/png")

    raise HTTPException(status_code=404, detail=f"Debug layer not found: {layer}")

# Dev convenience: run with `python -m src.service.api`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.service.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # keep for dev
        # Only watch source & web; explicitly ignore output-heavy dirs
        reload_includes=["src/*", "web/*", "configs/*"],
        reload_excludes=["runs/*", "out/*", "data/generated_heatmaps/*"],
    )
