"""Getting dataset images onto local disk, where LLaMA-Factory can open them."""

import hashlib
import http.client
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


class ImageFetchError(Exception):
    """An image could not be fetched; the message says why."""


NETWORK_ERROR = 0  # status used for connection failures, timeouts and truncated reads
MAX_RETRY_AFTER = 300.0  # seconds; a longer Retry-After is capped rather than stalling the run
RETRYABLE = {NETWORK_ERROR, 429, 500, 502, 503, 504}
# Raster formats PIL (used by LLaMA-Factory) opens. Wikimedia also hosts e.g. SVG, which it can't.
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
IMAGE_SUBDIR = "images"


def _urllib_get(url: str, headers: dict) -> tuple[int, dict, bytes]:
    """GET returning (status, headers, body). Network-level failures come back as
    NETWORK_ERROR instead of raising, so one bad image can't stop a long run."""
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), b""
    except (OSError, http.client.HTTPException) as e:  # URLError, timeouts, IncompleteRead, ...
        return NETWORK_ERROR, {"X-Error": repr(e)}, b""


def _header(headers: dict, name: str) -> str | None:
    """Case-insensitive header lookup."""
    return next((v for k, v in headers.items() if k.lower() == name.lower()), None)


class WikimediaImageStore:
    """Downloads Wikimedia images into <media_dir>/images/, one file per URL.

    Calling the store with a URL returns the image path relative to media_dir (what
    LLaMA-Factory's `images` column expects), or raises ImageFetchError.
    """

    def __init__(
        self,
        media_dir: Path,
        *,
        user_agent: str,
        min_interval: float = 1.0,
        max_attempts: int = 5,
        http_get=_urllib_get,
        clock=time.monotonic,
        sleep=time.sleep,
    ):
        self.media_dir = Path(media_dir)
        self.user_agent = user_agent
        self.http_get = http_get
        self.clock = clock
        self.sleep = sleep
        self.min_interval = min_interval  # Wikimedia allows roughly 1 image/s per IP
        self._last_request = None
        self.max_attempts = max_attempts

    def __call__(self, url: str) -> str:
        relative = self._relative_path(url)
        if Path(relative).suffix not in SUPPORTED_EXTENSIONS:
            raise ImageFetchError(f"unsupported image format: {Path(relative).suffix or '(none)'}")
        target = self.media_dir / relative
        if target.is_file() and target.stat().st_size > 0:
            return relative  # downloaded by an earlier run
        body = self._download(url)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file first so an interrupted run never leaves a truncated image.
        partial = target.with_name(target.name + ".part")
        partial.write_bytes(body)
        os.replace(partial, target)
        return relative

    def _download(self, url: str) -> bytes:
        for attempt in range(1, self.max_attempts + 1):
            status, headers, body = self._request(url)
            if status == 200:
                content_type = _header(headers, "Content-Type") or ""
                if not content_type.startswith("image/"):
                    raise ImageFetchError(f"not an image: Content-Type {content_type or '(missing)'}")
                return body
            if status not in RETRYABLE or attempt == self.max_attempts:
                break
            self.sleep(_retry_delay(headers, attempt))
        raise ImageFetchError(f"HTTP {status} after {attempt} attempt(s)")

    def _request(self, url: str):
        if self._last_request is not None:
            wait = self._last_request + self.min_interval - self.clock()
            if wait > 0:
                self.sleep(wait)
        self._last_request = self.clock()
        return self.http_get(url, {"User-Agent": self.user_agent})

    def _relative_path(self, url: str) -> str:
        digest = hashlib.sha1(url.encode()).hexdigest()
        ext = os.path.splitext(urllib.parse.urlparse(url).path)[1].lower()
        return f"{IMAGE_SUBDIR}/{digest[:2]}/{digest}{ext}"


def _retry_delay(headers: dict, attempt: int) -> float:
    """Seconds to wait before retrying: the server's Retry-After, else exponential backoff."""
    try:
        return min(MAX_RETRY_AFTER, max(0.0, float(_header(headers, "Retry-After"))))
    except (TypeError, ValueError):
        return min(60.0, 2.0**attempt)
