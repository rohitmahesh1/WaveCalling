# src/service/runs.py
from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import StreamingResponse, RedirectResponse, Response
from pydantic import BaseModel
from datetime import datetime, timezone
import json, io

from .storage import get_storage

router = APIRouter(prefix="/runs", tags=["runs"])
storage = get_storage()

class RunMeta(BaseModel):
    run_id: str
    owner_sid: str
    created_at: str  # ISO8601

def _meta_key(sid: str, run_id: str) -> str:
    return f"runs/{sid}/{run_id}/meta.json"

def _out_key(sid: str, run_id: str, name: str) -> str:
    return f"runs/{sid}/{run_id}/output/{name}"

async def _load_meta(run_id: str, sid: str) -> RunMeta | None:
    raw = await storage.get(_meta_key(sid, run_id))
    if not raw: return None
    return RunMeta(**json.loads(raw))

@router.post("", response_model=RunMeta)
async def create_run(request: Request):
    sid = request.state.sid
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")[-12:]
    meta = RunMeta(run_id=run_id, owner_sid=sid, created_at=datetime.now(timezone.utc).isoformat())
    await storage.put(_meta_key(sid, run_id), json.dumps(meta.dict()).encode("utf-8"), "application/json")
    return meta

@router.get("/{run_id}/info", response_model=RunMeta)
async def get_info(run_id: str, request: Request):
    sid = request.state.sid
    meta = await _load_meta(run_id, sid)
    if not meta: raise HTTPException(404)
    return meta

@router.put("/{run_id}/output/{name:path}")
async def put_output(run_id: str, name: str, request: Request):
    sid = request.state.sid
    meta = await _load_meta(run_id, sid)
    if not meta: raise HTTPException(404)  # no access or doesn’t exist

    data = await request.body()
    ct = request.headers.get("content-type", "application/octet-stream")
    await storage.put(_out_key(sid, run_id, name), data, ct)
    return {"ok": True}

@router.head("/{run_id}/output/{name:path}")
async def head_output(run_id: str, name: str, request: Request):
    sid = request.state.sid
    meta = await _load_meta(run_id, sid)
    if not meta: return Response(status_code=404)
    exists = await storage.exists(_out_key(sid, run_id, name))
    return Response(status_code=200 if exists else 404)

@router.get("/{run_id}/output/{name:path}")
async def get_output(run_id: str, name: str, request: Request):
    sid = request.state.sid
    meta = await _load_meta(run_id, sid)
    if not meta: raise HTTPException(404)

    key = _out_key(sid, run_id, name)
    signed = storage.signed_url(key, ttl_secs=3600)
    if signed:
        return RedirectResponse(signed, status_code=302)

    data = await storage.get(key)
    if data is None: raise HTTPException(404)
    # detect basic content type by extension
    ext = name.rsplit(".", 1)[-1].lower()
    ctype = {
        "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "json": "application/json", "csv": "text/csv",
    }.get(ext, "application/octet-stream")
    return StreamingResponse(io.BytesIO(data), media_type=ctype)
