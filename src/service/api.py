# src/service/api.py
from __future__ import annotations
from collections import deque
from contextlib import asynccontextmanager
import os

import aiofiles

from ..utils import load_config
# Ensure headless plotting in worker threads (must be set BEFORE any Matplotlib import anywhere)
os.environ.setdefault("MPLBACKEND", "Agg")

import re
import asyncio
import json
import shutil
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

# Pipeline generator (emits JobEvent)
from .pipeline import iter_run_project, JobEvent

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


# Global registry (in-memory)
_RUNS: Dict[str, _RunState] = {}


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
            # client disconnected
            return
        finally:
            await st.remove_subscriber(q)

    # keep-alive ping (seconds) — slightly longer to reduce noisy disconnect warnings
    return EventSourceResponse(gen(), ping=25)


@app.get("/api/runs/{run_id}/overlay")
async def get_overlay(run_id: str):
    """Return overlay JSON (tracks.json or partial if still running)."""
    st = _RUNS.get(run_id)
    base = _run_dir(run_id)
    if not (st or base.exists()):
        raise HTTPException(status_code=404, detail="Unknown run_id")

    output_dir = Path(st.output_dir) if st else (base / "output")
    final = output_dir / "overlay" / "tracks.json"
    partial = output_dir / "overlay" / "tracks.partial.json"
    path = final if final.exists() else partial
    if not path.exists():
        raise HTTPException(status_code=404, detail="Overlay not available yet")
    with open(path) as f:
        data = json.load(f)
    return JSONResponse(data)


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
async def get_progress(run_id: str):
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

    # 1) Preferred: read progress.json
    if progress_path.exists():
        try:
            with progress_path.open() as f:
                data = json.load(f)
            data.setdefault("source", "file")
            return JSONResponse(data)
        except Exception:
            # fall through to synthesized
            pass

    # 2) Synthesized fallback using marker dirs + events.ndjson
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
    return JSONResponse(payload)


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
    return {"deleted": run_id}
    

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
