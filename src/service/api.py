# src/service/api.py
from __future__ import annotations
import os
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
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
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


# ---- FastAPI app ----
app = FastAPI(title="WaveCalling API", version="0.1.0")

# (optional) CORS so the React app can talk to this API during dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    """Internal state for each run: dirs, subscribers, and status."""
    def __init__(self, run_id: str, name: str, base_dir: Path, config_path: Path):
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

        self.created_at = datetime.now(timezone.utc)
        self.status = "QUEUED"
        self.error: Optional[str] = None

        self.cancel_event: asyncio.Event = asyncio.Event()
        self.worker_task: Optional[asyncio.Task] = None

        self._subscribers: List[asyncio.Queue] = []
        self._sub_lock = asyncio.Lock()

    def info(self) -> RunInfo:
        return RunInfo(
            run_id=self.run_id,
            name=self.name,
            created_at=self.created_at.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
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
        # fan-out without blocking
        async with self._sub_lock:
            for q in list(self._subscribers):
                try:
                    q.put_nowait(evt)
                except asyncio.QueueFull:
                    # drop if back-pressured
                    pass


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
    """
    Save FastAPI UploadFile objects into out_dir.
    - Ensures the directory exists.
    - Sanitizes filenames.
    - Streams in chunks to avoid large memory spikes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    def _safe_name(name: str) -> str:
        # Keep basename only, strip directories, and allow [A-Za-z0-9._-]
        base = Path(name or "upload.bin").name
        return re.sub(r"[^A-Za-z0-9._-]+", "_", base)

    saved_paths: list[Path] = []
    for uf in files or []:
        fname = _safe_name(getattr(uf, "filename", "") or "upload.bin")
        out_path = out_dir / fname

        # Stream to disk
        with out_path.open("wb") as w:
            while True:
                chunk = await uf.read(64 * 1024)
                if not chunk:
                    break
                w.write(chunk)
        await uf.close()
        saved_paths.append(out_path)

    return saved_paths

def _artifact_urls(state: _RunState) -> Dict[str, str]:
    """
    Produce best-effort artifact URLs. Files may not exist yet while the run
    is still in progress.
    """
    base = f"/runs/{state.run_id}"
    # known outputs
    tracks_csv = Path(base) / "output" / "metrics.csv"
    waves_csv = Path(base) / "output" / "metrics_waves.csv"
    overlay_json = Path(base) / "output" / "overlay" / "tracks.json"
    manifest_json = Path(base) / "output" / "manifest.json"
    plots_dir = Path(base) / "plots"
    base_image = Path(base) / "output" / "base.png"

    return {
        "tracks_csv": str(tracks_csv),
        "waves_csv": str(waves_csv),
        "overlay_json": str(overlay_json),
        "manifest_json": str(manifest_json),
        "plots_dir": str(plots_dir),
        "output_dir": str(Path(base) / "output"),
        "base_image": str(base_image),
    }


async def _run_pipeline(state: _RunState, config_overrides: Optional[dict], verbose: bool) -> None:
    state.status = "RUNNING"
    loop = asyncio.get_running_loop()
    bridge_q: asyncio.Queue[JobEvent] = asyncio.Queue()

    def _runner_sync():
        got_terminal = False
        try:
            for evt in iter_run_project(
                input_dir=state.input_dir,
                config_path=state.config_path,
                output_dir=state.output_dir,
                plots_out=state.plots_dir,
                progress_cb=None,  # events go via bridge_q
                config_overrides=config_overrides,
                verbose=verbose,
                # PASS CANCELLATION HERE
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

    # Track worker so we can await it on shutdown or after cancel
    state.worker_task = asyncio.create_task(asyncio.to_thread(_runner_sync))

    try:
        while True:
            evt: JobEvent = await bridge_q.get()
            if evt.phase in ("ERROR", "DONE", "CANCELLED"):
                state.status = evt.phase
                if evt.phase == "ERROR":
                    state.error = evt.message or "Unknown error"
            await state.publish(evt)
            if evt.phase in ("ERROR", "DONE", "CANCELLED"):
                break

        # wait for the worker to finish cleanly
        await state.worker_task

    except Exception as e:
        state.status = "ERROR"
        state.error = str(e)
        await state.publish(JobEvent(phase="ERROR", message=str(e), progress=1.0))
    finally:
        terminal = state.status if state.status in ("DONE", "ERROR", "CANCELLED") else "ERROR"
        await state.publish(JobEvent(phase=terminal, message="Run finished", progress=1.0))


# -----------------------
# Endpoints
# -----------------------

@app.post("/api/runs", response_model=CreateRunResponse)
async def create_run(
    files: List[UploadFile] = File(..., description="One or more CSV/XLS/PNG/JPG"),
    run_name: Optional[str] = Form(None),
    config_overrides: Optional[str] = Form(
        None, description="JSON string with partial overrides to default YAML"
    ),
    verbose: bool = Form(False),
):
    """
    Create a new run:
      - Saves uploads under runs/<run_id>/input
      - Copies/uses default config YAML (plus optional overrides)
      - Starts the pipeline in background
      - Returns run_id immediately
    """
    run_id = uuid.uuid4().hex[:10]
    base_dir = RUNS_ROOT / run_id
    (base_dir / "input").mkdir(parents=True, exist_ok=True)

    # Save inputs
    saved = await _save_uploads(files, base_dir / "input")
    if not saved:
        raise HTTPException(status_code=400, detail="No files uploaded")

    # Resolve config (copy default next to run for traceability)
    if not DEFAULT_CONFIG.exists():
        raise HTTPException(status_code=500, detail=f"Default config not found: {DEFAULT_CONFIG}")
    config_path = base_dir / "config.yaml"
    shutil.copyfile(DEFAULT_CONFIG, config_path)

    # Parse overrides if provided
    overrides: Optional[dict] = None
    if config_overrides:
        try:
            overrides = json.loads(config_overrides)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid config_overrides JSON: {e}")

    # Create state & registry entry
    state = _RunState(run_id=run_id, name=run_name or f"run-{run_id}", base_dir=base_dir, config_path=config_path)
    _RUNS[run_id] = state

    # Kick off worker task (async)
    asyncio.create_task(_run_pipeline(state, overrides, verbose))

    return CreateRunResponse(run_id=run_id, status=state.status, info=state.info())


@app.get("/api/runs", response_model=List[RunInfo])
async def list_runs():
    """List recent runs (in-memory registry)."""
    return [st.info() for st in _RUNS.values()]


@app.get("/api/runs/{run_id}", response_model=RunStatusResponse)
async def get_run(run_id: str):
    """Run status + best-effort artifact URLs."""
    st = _RUNS.get(run_id)
    if not st:
        raise HTTPException(status_code=404, detail="Unknown run_id")
    return RunStatusResponse(info=st.info(), artifacts=_artifact_urls(st))


@app.get("/api/runs/{run_id}/artifacts")
async def get_run_artifacts(run_id: str):
    """Raw artifact URL map (even if files not yet available)."""
    st = _RUNS.get(run_id)
    if not st:
        raise HTTPException(status_code=404, detail="Unknown run_id")
    return JSONResponse(_artifact_urls(st))


@app.get("/api/runs/{run_id}/events")
async def stream_events(run_id: str):
    """
    Server-Sent Events stream of JobEvent for the given run.
    Browser usage:
      const es = new EventSource(`/api/runs/${runId}/events`);
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
        return p  # if already URL-like or unknown root

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
            # push an initial status snapshot to new subscribers
            yield {"event": "message", "data": json.dumps({"phase": st.status, "message": "subscribed", "progress": 0.0})}
            while True:
                evt: JobEvent = await q.get()
                payload = asdict(evt)
                payload = _rewrite_payload(payload)
                yield {"event": "message", "data": json.dumps(payload)}
                if evt.phase in ("DONE", "ERROR"):
                    break
        finally:
            await st.remove_subscriber(q)

    # ping keeps the connection alive through proxies (seconds)
    return EventSourceResponse(gen(), ping=15)


@app.get("/api/runs/{run_id}/overlay")
async def get_overlay(run_id: str):
    """Return overlay JSON (tracks.json or partial if still running)."""
    st = _RUNS.get(run_id)
    if not st:
        raise HTTPException(status_code=404, detail="Unknown run_id")
    # Prefer final file, fall back to partial if present
    final = st.output_dir / "overlay" / "tracks.json"
    partial = st.output_dir / "overlay" / "tracks.partial.json"
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
    if not st:
        raise HTTPException(status_code=404, detail="Unknown run_id")
    img = st.output_dir / "base.png"
    if not img.exists():
        raise HTTPException(status_code=404, detail="Base image not available yet")
    return FileResponse(str(img), media_type="image/png")


@app.get("/api/runs/{run_id}/waves")
async def list_wave_windows(run_id: str, track: Optional[str] = None):
    """List PNGs under plots/<track>/peak_windows (as /runs URLs)."""
    st = _RUNS.get(run_id)
    if not st:
        raise HTTPException(status_code=404, detail="Unknown run_id")
    if not track:
        raise HTTPException(status_code=400, detail="Query param 'track' is required")
    win_dir = st.plots_dir / str(track) / "peak_windows"
    if not win_dir.exists():
        return JSONResponse({"images": []})
    files = sorted([p for p in win_dir.glob("*.png")])
    urls = [f"/runs/{st.run_id}/plots/{track}/peak_windows/{p.name}" for p in files]
    return JSONResponse({"images": urls})

@app.post("/api/runs/{run_id}/cancel")
async def cancel_run(run_id: str):
    """
    Signal a running job to stop early. The pipeline will:
      - Finish the current micro-step,
      - Write partial artifacts (.partial.csv / tracks.partial.json),
      - Emit a CANCELLED terminal event.
    """
    st = _RUNS.get(run_id)
    if not st:
        raise HTTPException(status_code=404, detail="Unknown run_id")
    if st.status not in ("QUEUED", "RUNNING"):
        return {"run_id": run_id, "status": st.status, "message": "Run is not active"}

    # flip the flag; the generator will exit soon
    st.cancel_event.set()
    return {"run_id": run_id, "status": "CANCEL_REQUESTED"}


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
