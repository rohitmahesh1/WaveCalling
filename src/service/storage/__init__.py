# src/service/storage/__init__.py
import os
from typing import TYPE_CHECKING
from .base import Storage

if TYPE_CHECKING:
    from .local import LocalStorage  # noqa: F401
    from .gcs import GCSStorage      # noqa: F401


def get_storage() -> Storage:
    """
    Decide storage backend at runtime.

    STORAGE=local  (default) -> LocalStorage
    STORAGE=gcs                -> GCSStorage (requires GCS_BUCKET)
    """
    kind = (os.getenv("STORAGE") or "local").strip().lower()

    if kind == "gcs":
        try:
            from .gcs import GCSStorage  # lazy import; avoid importing google libs if not needed
        except Exception as e:
            raise RuntimeError(
                "STORAGE=gcs selected but google-cloud-storage is not available "
                "or a conflicting 'google' package is installed."
            ) from e

        bucket = os.environ.get("GCS_BUCKET")
        if not bucket:
            raise RuntimeError("STORAGE=gcs requires GCS_BUCKET to be set")
        return GCSStorage(bucket_name=bucket)

    # default: local
    from .local import LocalStorage  # <- lazy import
    root = os.getenv("RUNS_DIR", "/app/runs")
    return LocalStorage(root=root)
