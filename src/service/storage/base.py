# src/service/storage/base.py
from __future__ import annotations

import mimetypes
import posixpath
import re
from typing import Iterable, List, Optional, Protocol

__all__ = [
    "Storage",
    "safe_join",
    "content_type_from_name",
]

# Allow only URL-safe / object-key-safe characters
_ALLOWED = re.compile(r"^[a-zA-Z0-9._~/+\-]+$")


def safe_join(*parts: str) -> str:
    """
    Join path segments into a POSIX key suitable for GCS/local object storage.
    - Prevents leading //, .., and backslashes.
    - Collapses redundant slashes.
    """
    cleaned: List[str] = []
    for p in parts:
        if p is None:
            continue
        s = str(p).strip().replace("\\", "/")
        if not s:
            continue
        if s.startswith("/"):
            s = s.lstrip("/")
        if s in (".", "./"):
            continue
        if ".." in s.split("/"):
            raise ValueError("unsafe path segment")
        cleaned.append(s)
    key = posixpath.join(*cleaned) if cleaned else ""
    if not _ALLOWED.match(key):
        raise ValueError("unsafe characters in path")
    return key


# Extend mimetypes with a few we care about
mimetypes.add_type("application/x-ndjson", ".ndjson")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("text/csv", ".csv")
mimetypes.add_type("text/tab-separated-values", ".tsv")


def content_type_from_name(name: str) -> str:
    ctype, _ = mimetypes.guess_type(name, strict=False)
    return ctype or "application/octet-stream"


class Storage(Protocol):
    """
    Minimal async storage interface used by the API:
      - put(path, data, content_type)
      - get(path) -> bytes|None
      - exists(path) -> bool
      - list(prefix) -> list[str]  (NEW)
      - signed_url(path, ttl_secs) -> str|None (legacy)
      - get_signed_url(path, method, ttl) -> str|None (preferred)
    """

    async def put(self, path: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        ...

    async def get(self, path: str) -> Optional[bytes]:
        ...

    async def exists(self, path: str) -> bool:
        ...

    async def list(self, prefix: str) -> List[str]:
        ...

    def signed_url(self, path: str, ttl_secs: int = 3600) -> Optional[str]:
        ...

    async def get_signed_url(self, path: str, method: str = "GET", ttl: int = 300) -> Optional[str]:
        """
        Default shim: if the backend only provides signed_url(), call it and ignore 'method'.
        Backends can override to honor HEAD/PUT/DELETE, etc.
        """
        url = self.signed_url(path, ttl_secs=ttl)
        return url
