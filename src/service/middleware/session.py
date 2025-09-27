# src/service/middleware/session.py
import uuid, hmac, hashlib, base64, os
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

SECRET = os.environ.get("SESSION_SECRET", "dev-secret-change-me")
COOKIE = "sid"
AGE = 60*60*24*7  # 7 days

def _sign(v: str) -> str:
    sig = hmac.new(SECRET.encode(), v.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(sig).decode().rstrip("=")

def _pack(sid_raw: str) -> str: return f"{sid_raw}.{_sign(sid_raw)}"

def _valid(packed: str) -> bool:
    try:
        sid_raw, sig = packed.split(".", 1)
        return hmac.compare_digest(_sign(sid_raw), sig)
    except: return False

class SessionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        packed = request.cookies.get(COOKIE)
        if not packed or not _valid(packed):
            raw = uuid.uuid4().hex
            packed = _pack(raw)
            request.state.new_cookie = packed
            sid_raw = raw
        else:
            sid_raw = packed.split(".", 1)[0]
        request.state.sid = sid_raw

        resp: Response = await call_next(request)
        if getattr(request.state, "new_cookie", None):
            resp.set_cookie(
                COOKIE, packed, max_age=AGE, httponly=True,
                samesite="Lax", secure=True  # set true in prod
            )
        return resp
