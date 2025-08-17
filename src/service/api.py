# src/service/api.py
from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")



# -----------------------
# In-memory run registry
# -----------------------
class RunInfo(BaseModel):
    run_id: str
    name: str
    created_at: str
    status: str                 # QUEUED | RUNNING | DONE | ERROR
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

        self.created_at = datetime.utcnow()
        self.status = "QUEUED"
        self.error: Optional[str] = None

        # Each subscriber gets its own queue; publisher fan-outs events to all
        self._subscribers: List[asyncio.Queue] = []
        self._sub_lock = asyncio.Lock()

    def info(self) -> RunInfo:
        return RunInfo(
            run_id=self.run_id,
            name=self.name,
            created_at=self.created_at.isoformat() + "Z",
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
def _save_uploads(files: List[UploadFile], dest_dir: Path) -> List[Path]:
    saved: List[Path] = []
    for f in files:
        # preserve original filename
        out_path = dest_dir / Path(f.filename).name
        with out_path.open("wb") as w:
            shutil.copyfileobj(f.file, w)
        saved.append(out_path)
    return saved


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

    return {
        "tracks_csv": str(tracks_csv),
        "waves_csv": str(waves_csv),
        "overlay_json": str(overlay_json),
        "manifest_json": str(manifest_json),
        "plots_dir": str(plots_dir),
        "output_dir": str(Path(base) / "output"),
    }


async def _run_pipeline(state: _RunState, config_overrides: Optional[dict], verbose: bool) -> None:
    """
    Execute the pipeline in a worker thread and forward JobEvents to subscribers.
    """
    state.status = "RUNNING"

    def _progress(evt: JobEvent):
        # Bridge sync callback into async fan-out
        # Use asyncio.run_coroutine_threadsafe to schedule publish
        loop = asyncio.get_event_loop()
        asyncio.run_coroutine_threadsafe(state.publish(evt), loop)

    try:
        # The generator itself yields JobEvents we can forward as well
        for evt in iter_run_project(
            input_dir=state.input_dir,
            config_path=state.config_path,
            output_dir=state.output_dir,
            plots_out=state.plots_dir,
            progress_cb=_progress,
            config_overrides=config_overrides,
            verbose=verbose,
        ):
            await state.publish(evt)

        state.status = "DONE"
    except Exception as e:
        state.status = "ERROR"
        state.error = str(e)
        # Emit an error event so clients are notified
        await state.publish(JobEvent(phase="ERROR", message=str(e), progress=1.0))
    finally:
        # Emit a terminal DONE/ERROR event for clients that missed it
        terminal = "DONE" if state.status == "DONE" else "ERROR"
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
    base_dir.mkdir(parents=True, exist_ok=True)

    # Save inputs
    saved = _save_uploads(files, base_dir / "input")
    if not saved:
        raise HTTPException(status_code=400, detail="No files uploaded")

    # Resolve config (copy default next to run for traceability)
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

    async def gen():
        try:
            # push an initial status snapshot to new subscribers
            yield {"event": "message", "data": json.dumps({"phase": st.status, "message": "subscribed", "progress": 0.0})}
            while True:
                evt: JobEvent = await q.get()
                payload = asdict(evt)
                yield {"event": "message", "data": json.dumps(payload)}
                if evt.phase in ("DONE", "ERROR"):
                    break
        finally:
            await st.remove_subscriber(q)

    return EventSourceResponse(gen())


# Dev convenience: run with `python -m src.service.api`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.service.api:app", host="0.0.0.0", port=8000, reload=True)
