"""Minimal requests-backed Zarr store.

Zarr v2 needs a mapping from keys to bytes. This module keeps that mapping
private and deliberately small so the package does not require fsspec.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import requests
from zarr.storage import Store

from .exceptions import StoreUnavailableError

_RETRY_STATUS = {429, 500, 502, 503, 504}


class RequestsStore(Store):
    """Read-only Zarr store backed by plain HTTP requests."""

    def __init__(
        self,
        base_url: str,
        session: requests.Session,
        *,
        timeout: float = 30.0,
        retries: int = 3,
    ) -> None:
        """Initialize a read-only store for ``base_url``.

        Parameters
        ----------
        base_url:
            Root URL of the Zarr store. A trailing slash is added.
        session:
            HTTP session used for all requests.
        timeout:
            Per-request timeout in seconds.
        retries:
            Number of attempts for each key before giving up.
        """
        self.base_url = base_url.rstrip("/") + "/"
        self.session = session
        self.timeout = float(timeout)
        self.retries = max(1, int(retries))

    def __getitem__(self, key: str) -> bytes:
        """Return the raw bytes stored under ``key``.

        Transient HTTP failures are retried. A missing key raises
        :class:`KeyError` and other permanent failures raise
        :class:`StoreUnavailableError`.
        """
        url = self.base_url + key
        for attempt in range(self.retries):
            try:
                response = self.session.get(url, timeout=self.timeout)
            except requests.RequestException as exc:
                if attempt + 1 == self.retries:
                    raise StoreUnavailableError(
                        f"cannot read {url}: {exc}"
                    ) from exc
            else:
                if response.status_code == 200:
                    return response.content
                if response.status_code == 404:
                    raise KeyError(key)
                if response.status_code not in _RETRY_STATUS:
                    raise StoreUnavailableError(
                        f"HTTP {response.status_code} while reading {url}"
                    )
            time.sleep(0.5 * (2**attempt))
        raise StoreUnavailableError(f"cannot read {url}")

    def getitems(self, keys, *, contexts=None):
        """Read several chunk keys, skipping missing chunks."""

        result: dict[str, bytes] = {}
        for key in keys:
            try:
                result[key] = self[key]
            except KeyError:
                continue
        return result

    def __contains__(self, key: object) -> bool:
        """Return whether ``key`` exists in the remote store."""
        if not isinstance(key, str):
            return False
        try:
            response = self.session.head(
                self.base_url + key, timeout=self.timeout
            )
        except requests.RequestException:
            response = None
        if response is not None:
            if response.status_code == 200:
                return True
            if response.status_code == 404:
                return False
        try:
            self[key]
        except KeyError:
            return False
        return True

    def __setitem__(self, key: str, value: bytes) -> None:
        """Reject writes because the remote store is read-only."""
        raise TypeError("RequestsStore is read-only")

    def __delitem__(self, key: str) -> None:
        """Reject deletions because the remote store is read-only."""
        raise TypeError("RequestsStore is read-only")

    def __iter__(self) -> Iterator[str]:
        """Return no keys; consolidated Zarr metadata is used instead."""
        return iter(())

    def __len__(self) -> int:
        """Return zero because this store does not list keys."""
        return 0
