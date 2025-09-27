# src/service/storage/gcs.py
import asyncio
import datetime
from typing import List, Optional
from google.cloud import storage as gcs_storage
from google.api_core.exceptions import NotFound
from .base import Storage

class GCSStorage(Storage):
    def __init__(self, bucket_name: str):
        self.client = gcs_storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    async def put(self, path: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        def _upload():
            blob = self.bucket.blob(path)
            blob.upload_from_string(data, content_type=content_type)
        await asyncio.to_thread(_upload)

    async def get(self, path: str) -> Optional[bytes]:
        def _download():
            blob = self.bucket.blob(path)
            try:
                return blob.download_as_bytes()
            except NotFound:
                return None
        return await asyncio.to_thread(_download)

    async def exists(self, path: str) -> bool:
        def _exists():
            return self.bucket.blob(path).exists()
        return await asyncio.to_thread(_exists)

    async def list(self, prefix: str) -> List[str]:
        def _list():
            # list_blobs handles both "dir" prefixes and file prefixes
            return [b.name for b in self.client.list_blobs(self.bucket, prefix=prefix)]
        return await asyncio.to_thread(_list)

    def signed_url(self, path: str, ttl_secs: int = 3600) -> Optional[str]:
        """
        Legacy signer (GET only). Prefer get_signed_url().
        """
        blob = self.bucket.blob(path)
        return blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(seconds=ttl_secs),
            method="GET",
        )

    async def get_signed_url(self, path: str, method: str = "GET", ttl: int = 300) -> Optional[str]:
        """
        Sign a URL for the given method. We map HEAD→GET, since GCS
        doesn't require a separate HEAD signature and many clients probe with HEAD.
        """
        verb = (method or "GET").upper()
        if verb == "HEAD":
            verb = "GET"
        def _sign():
            blob = self.bucket.blob(path)
            return blob.generate_signed_url(
                version="v4",
                expiration=datetime.timedelta(seconds=ttl),
                method=verb,
            )
        return await asyncio.to_thread(_sign)
