# src/service/storage/local.py
import os
from typing import List, Optional
import aiofiles
from .base import Storage

class LocalStorage(Storage):
    def __init__(self, root: str):
        self.root = root

    def _fs(self, path: str) -> str:
        full = os.path.join(self.root, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        return full

    async def put(self, path: str, data: bytes, content_type="application/octet-stream") -> None:
        async with aiofiles.open(self._fs(path), "wb") as f:
            await f.write(data)

    async def get(self, path: str) -> Optional[bytes]:
        fp = self._fs(path)
        if not os.path.exists(fp):
            return None
        async with aiofiles.open(fp, "rb") as f:
            return await f.read()

    async def exists(self, path: str) -> bool:
        return os.path.exists(self._fs(path))

    async def list(self, prefix: str) -> List[str]:
        """
        Return object keys under 'prefix'. If prefix resolves to a file, return [prefix].
        """
        root_prefix = self._fs(prefix)
        results: List[str] = []
        if os.path.isfile(root_prefix):
            return [prefix]
        if os.path.isdir(root_prefix):
            for dirpath, _, files in os.walk(root_prefix):
                for fn in files:
                    full = os.path.join(dirpath, fn)
                    rel = os.path.relpath(full, self.root).replace(os.sep, "/")
                    results.append(rel)
        return sorted(results)

    def signed_url(self, path: str, ttl_secs=3600) -> Optional[str]:
        # No external server; return None so the API proxies bytes or does HEAD locally.
        return None

    async def get_signed_url(self, path: str, method: str = "GET", ttl: int = 300) -> Optional[str]:
        # Mirror the base shim behavior
        return self.signed_url(path, ttl_secs=ttl)
