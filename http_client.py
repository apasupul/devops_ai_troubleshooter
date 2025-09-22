#!/usr/bin/env python3
"""
http_client.py
----------------
A small, production-friendly HTTP utility built on `requests` with:

- Session pooling & connection reuse
- Retries with exponential backoff (including POST)
- Sensible timeouts
- Per-call overrides for headers, auth, proxies, verify
- JSON helpers and streaming download
- Simple auth helpers (Bearer, Basic)
- Minimal, structured exceptions
- Optional base_url for concise relative paths
- Debug logging hooks

Usage:
    from http_client import HttpClient, HttpError

    with HttpClient(base_url="https://api.example.com", timeout=10) as http:
        # JSON GET
        data = http.get_json("/v1/items", params={"limit": 10})

        # POST JSON
        created = http.post_json("/v1/items", json={"name": "foo"})

        # Raw request
        resp = http.get("/status/200")
        resp.raise_for_status()

        # Download a file
        http.download("/v1/bigfile", "bigfile.bin")

    # Quick one-off:
    from http_client import quick_get_json
    info = quick_get_json("https://httpbin.org/get")

Set LOG level to DEBUG to see request/response details:
    import logging
    logging.basicConfig(level=logging.DEBUG)

Requirements:
    pip install requests
"""

from __future__ import annotations

import io
import json as _json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

import requests
from requests import Response, Session
from requests.auth import AuthBase, HTTPBasicAuth
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


__all__ = [
    "HttpClient",
    "HttpError",
    "quick_get",
    "quick_get_json",
    "quick_post_json",
]


# --- Logging -----------------------------------------------------------------

log = logging.getLogger("http_client")


# --- Exceptions ---------------------------------------------------------------

@dataclass
class HttpError(Exception):
    """Raised when an HTTP request fails (4xx/5xx) with useful context."""
    status_code: int
    url: str
    message: str
    response_text: str = ""
    headers: Optional[Mapping[str, str]] = None

    def __str__(self) -> str:  # pragma: no cover - simple representation
        base = f"HTTP {self.status_code} for {self.url}: {self.message}"
        if self.response_text:
            trimmed = self.response_text.strip()
            if len(trimmed) > 300:
                trimmed = trimmed[:297] + "..."
            base += f" | body: {trimmed}"
        return base


# --- Helpers -----------------------------------------------------------------

_DEFAULT_ALLOWED_METHODS = frozenset({"HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"})

def _build_retry(
    total: int = 3,
    backoff_factor: float = 0.3,
    status_forcelist: Iterable[int] = (500, 502, 503, 504, 522, 524),
    allowed_methods: Iterable[str] = _DEFAULT_ALLOWED_METHODS,
) -> Retry:
    # Respect Retry-After for 429/5xx when present.
    return Retry(
        total=total,
        connect=total,
        read=total,
        status=total,
        backoff_factor=backoff_factor,
        status_forcelist=set(status_forcelist),
        allowed_methods=set(m.upper() for m in allowed_methods),
        raise_on_status=False,
        respect_retry_after_header=True,
    )


def _default_headers() -> Dict[str, str]:
    return {
        "User-Agent": os.getenv("HTTP_CLIENT_UA", "http-client/1.0 (+https://example.local)"),
        "Accept": "application/json, */*;q=0.8",
    }


# --- Core Client --------------------------------------------------------------

class HttpClient:
    """
    High-level HTTP client.

    Parameters
    ----------
    base_url : str | None
        If provided, relative paths passed to request methods are joined to this base.
    timeout : float | Tuple[float, float]
        Default timeout (seconds). Either a single float (total) or (connect, read).
    retries : int
        Total retries for transient errors.
    backoff_factor : float
        Exponential backoff factor between retries.
    status_forcelist : Iterable[int]
        HTTP statuses that should trigger a retry.
    verify : bool | str
        TLS verification (bool) or path to CA bundle.
    proxies : Mapping[str, str] | None
        Optional requests-compatible proxies mapping.
    headers : Mapping[str, str] | None
        Default headers merged into every request.
    auth : requests.auth.AuthBase | tuple | None
        Default auth. Use set_bearer() / set_basic() helpers for convenience.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        timeout: Union[float, Tuple[float, float]] = 10.0,
        retries: int = 3,
        backoff_factor: float = 0.3,
        status_forcelist: Iterable[int] = (500, 502, 503, 504, 522, 524),
        verify: Union[bool, str] = True,
        proxies: Optional[Mapping[str, str]] = None,
        headers: Optional[Mapping[str, str]] = None,
        auth: Optional[Union[AuthBase, Tuple[str, str]]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        self.timeout = timeout
        self.verify = verify
        self.proxies = dict(proxies) if proxies else None
        self.session: Session = Session()
        self.session.headers.update(_default_headers())
        if headers:
            self.session.headers.update(dict(headers))
        if auth:
            self.session.auth = auth  # type: ignore[assignment]

        retry = _build_retry(total=retries, backoff_factor=backoff_factor, status_forcelist=status_forcelist)
        adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Optional: respect environment proxies unless explicitly given.
        self.session.trust_env = self.proxies is None

        log.debug("HttpClient initialized (base_url=%r, timeout=%r, retries=%r)", self.base_url, self.timeout, retries)

    # --- Context manager ------------------------------------------------------

    def __enter__(self) -> "HttpClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self.session.close()

    # --- Auth helpers ---------------------------------------------------------

    def set_bearer(self, token: str) -> None:
        """Set Bearer token auth (overrides any previous auth)."""
        self.session.headers["Authorization"] = f"Bearer {token}"

    def set_basic(self, username: str, password: str) -> None:
        """Set HTTP Basic auth."""
        self.session.auth = HTTPBasicAuth(username, password)

    # --- URL handling ---------------------------------------------------------

    def _make_url(self, path_or_url: str) -> str:
        if self.base_url and not path_or_url.lower().startswith(("http://", "https://")):
            return f"{self.base_url}/{path_or_url.lstrip('/')}"
        return path_or_url

    # --- Core request ---------------------------------------------------------

    def _request(
        self,
        method: str,
        path_or_url: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        data: Optional[Union[Mapping[str, Any], bytes, io.BufferedReader]] = None,
        json: Optional[Any] = None,
        headers: Optional[Mapping[str, str]] = None,
        files: Optional[Mapping[str, Any]] = None,
        stream: bool = False,
        timeout: Optional[Union[float, Tuple[float, float]]] = None,
        allow_redirects: bool = True,
        **kwargs: Any,
    ) -> Response:
        url = self._make_url(path_or_url)
        merged_headers = dict(self.session.headers)
        if headers:
            merged_headers.update(headers)

        # Log request at DEBUG level (safe basics only).
        log.debug(
            "HTTP %s %s | params=%s json=%s data=%s headers=%s",
            method, url, params, _safe_json_preview(json), _safe_data_preview(data), _redact_headers(merged_headers)
        )

        resp = self.session.request(
            method=method.upper(),
            url=url,
            params=params,
            data=data,
            json=json,
            headers=merged_headers,
            files=files,
            stream=stream,
            timeout=self._resolve_timeout(timeout),
            verify=self.verify,
            proxies=self.proxies,
            allow_redirects=allow_redirects,
            **kwargs,
        )

        # Log response at DEBUG level.
        log.debug("HTTP %s %s -> %s | length=%s", method, url, resp.status_code, resp.headers.get("Content-Length"))

        if 400 <= resp.status_code:
            # Try to extract a concise error message.
            message = _extract_error_message(resp)
            raise HttpError(
                status_code=resp.status_code,
                url=url,
                message=message,
                response_text=_safe_text(resp),
                headers=dict(resp.headers),
            )
        return resp

    # --- Public verbs ---------------------------------------------------------

    def get(self, path_or_url: str, **kwargs: Any) -> Response:
        return self._request("GET", path_or_url, **kwargs)

    def post(self, path_or_url: str, **kwargs: Any) -> Response:
        return self._request("POST", path_or_url, **kwargs)

    def put(self, path_or_url: str, **kwargs: Any) -> Response:
        return self._request("PUT", path_or_url, **kwargs)

    def delete(self, path_or_url: str, **kwargs: Any) -> Response:
        return self._request("DELETE", path_or_url, **kwargs)

    # --- JSON helpers ---------------------------------------------------------

    def get_json(self, path_or_url: str, **kwargs: Any) -> Any:
        """GET and parse JSON; raises HttpError for 4xx/5xx and ValueError for invalid JSON."""
        resp = self.get(path_or_url, **kwargs)
        return self._parse_json(resp)

    def post_json(self, path_or_url: str, *, json: Any = None, **kwargs: Any) -> Any:
        """POST JSON and parse response JSON."""
        resp = self.post(path_or_url, json=json, **kwargs)
        return self._parse_json(resp)

    def put_json(self, path_or_url: str, *, json: Any = None, **kwargs: Any) -> Any:
        """PUT JSON and parse response JSON."""
        resp = self.put(path_or_url, json=json, **kwargs)
        return self._parse_json(resp)

    def delete_json(self, path_or_url: str, **kwargs: Any) -> Any:
        """DELETE and parse response JSON."""
        resp = self.delete(path_or_url, **kwargs)
        return self._parse_json(resp)

    # --- Download -------------------------------------------------------------

    def download(self, path_or_url: str, dest_path: str, *, chunk_size: int = 8192, **kwargs: Any) -> str:
        """Stream a download to disk safely. Returns the destination path."""
        resp = self.get(path_or_url, stream=True, **kwargs)
        resp.raise_for_status()  # Shouldn't happen due to _request handling, but keep for robustness.

        os.makedirs(os.path.dirname(os.path.abspath(dest_path)) or ".", exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)
        return dest_path

    # --- Internals ------------------------------------------------------------

    def _resolve_timeout(self, per_call_timeout: Optional[Union[float, Tuple[float, float]]]) -> Union[float, Tuple[float, float]]:
        return self.timeout if per_call_timeout is None else per_call_timeout

    @staticmethod
    def _parse_json(resp: Response) -> Any:
        try:
            return resp.json()
        except ValueError:
            # Fallback: try to parse manually to give better errors.
            text = resp.text
            try:
                return _json.loads(text)
            except Exception as e:  # pragma: no cover
                raise ValueError(f"Response body is not valid JSON (status {resp.status_code})") from e


# --- Convenience one-offs -----------------------------------------------------

def quick_get(url: str, **kwargs: Any) -> Response:
    with HttpClient() as http:
        return http.get(url, **kwargs)

def quick_get_json(url: str, **kwargs: Any) -> Any:
    with HttpClient() as http:
        return http.get_json(url, **kwargs)

def quick_post_json(url: str, json: Any = None, **kwargs: Any) -> Any:
    with HttpClient() as http:
        return http.post_json(url, json=json, **kwargs)


# --- Safe previews & redaction -----------------------------------------------

def _safe_json_preview(payload: Any, limit: int = 200) -> str:
    if payload is None:
        return "None"
    try:
        s = _json.dumps(payload, ensure_ascii=False)
    except Exception:
        s = str(payload)
    return s if len(s) <= limit else s[:limit] + "..."

def _safe_data_preview(data: Any, limit: int = 200) -> str:
    if data is None:
        return "None"
    if isinstance(data, (bytes, bytearray)):
        return f"<{len(data)} bytes>"
    if hasattr(data, "read"):
        return "<stream>"
    s = str(data)
    return s if len(s) <= limit else s[:limit] + "..."

def _redact_headers(headers: Mapping[str, str]) -> Mapping[str, str]:
    redacted = dict(headers)
    for k in list(redacted.keys()):
        if k.lower() in {"authorization", "proxy-authorization"}:
            redacted[k] = "<redacted>"
    return redacted

def _extract_error_message(resp: Response) -> str:
    # Try common JSON error shapes.
    try:
        body = resp.json()
        for key in ("error", "message", "detail", "title"):
            if isinstance(body, dict) and key in body and isinstance(body[key], (str, int, float)):
                return str(body[key])
        # Some APIs nest errors
        if isinstance(body, dict):
            if "errors" in body and isinstance(body["errors"], (list, tuple)) and body["errors"]:
                first = body["errors"][0]
                if isinstance(first, dict):
                    for key in ("message", "detail"):
                        if key in first:
                            return str(first[key])
                return str(first)
    except Exception:
        pass
    # Fallback to status phrase or nothing.
    return resp.reason or "HTTP error"

def _safe_text(resp: Response, limit: int = 1000) -> str:
    try:
        text = resp.text
        return text if len(text) <= limit else text[:limit] + "..."
    except Exception:
        return "<unavailable>"
