import hashlib

import pytest

from llmtrain.images import ImageFetchError, WikimediaImageStore

URL = "https://upload.wikimedia.org/wikipedia/commons/0/05/Zubov.jpg"
JPEG = b"\xff\xd8\xff fake jpeg bytes"


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeHttp:
    """Serves queued (status, headers, body) responses per URL and records each request."""

    def __init__(self, clock, responses):
        self.clock = clock
        self.responses = {url: list(queue) for url, queue in responses.items()}
        self.requests = []  # (url, headers, time)

    def __call__(self, url, headers):
        self.requests.append((url, headers, self.clock()))
        return self.responses[url].pop(0)


def ok(body=JPEG, content_type="image/jpeg"):
    return 200, {"Content-Type": content_type}, body


def store(tmp_path, http, clock, **kw):
    return WikimediaImageStore(
        tmp_path, user_agent="LLMtrain/0.1 (test)", http_get=http, clock=clock, sleep=clock.sleep, **kw
    )


def test_downloads_image_to_a_stable_path_relative_to_media_dir(tmp_path):
    clock = FakeClock()
    http = FakeHttp(clock, {URL: [ok()]})

    path = store(tmp_path, http, clock)(URL)

    digest = hashlib.sha1(URL.encode()).hexdigest()
    assert path == f"images/{digest[:2]}/{digest}.jpg"
    assert (tmp_path / path).read_bytes() == JPEG
    assert http.requests[0][1]["User-Agent"] == "LLMtrain/0.1 (test)"


def test_already_downloaded_image_is_not_requested_again_by_a_new_run(tmp_path):
    clock = FakeClock()
    first = store(tmp_path, FakeHttp(clock, {URL: [ok()]}), clock)(URL)
    http = FakeHttp(clock, {URL: []})

    assert store(tmp_path, http, clock)(URL) == first
    assert http.requests == []


def test_requests_are_spaced_by_the_minimum_interval(tmp_path):
    clock = FakeClock()
    urls = [f"https://upload.wikimedia.org/{i}.jpg" for i in range(3)]
    http = FakeHttp(clock, {u: [ok()] for u in urls})
    images = store(tmp_path, http, clock, min_interval=1.0)

    for u in urls:
        images(u)

    times = [t for _, _, t in http.requests]
    assert times == [0.0, 1.0, 2.0]


def test_429_waits_for_retry_after_then_succeeds(tmp_path):
    clock = FakeClock()
    http = FakeHttp(clock, {URL: [(429, {"Retry-After": "7"}, b""), ok()]})

    path = store(tmp_path, http, clock)(URL)

    assert (tmp_path / path).read_bytes() == JPEG
    first, second = (t for _, _, t in http.requests)
    assert second - first >= 7


def test_gives_up_after_max_attempts_of_rate_limiting(tmp_path):
    clock = FakeClock()
    http = FakeHttp(clock, {URL: [(429, {"Retry-After": "1"}, b"")] * 3})

    with pytest.raises(ImageFetchError, match="429"):
        store(tmp_path, http, clock, max_attempts=3)(URL)
    assert len(http.requests) == 3
    assert not list(tmp_path.rglob("*.jpg"))


def test_missing_image_fails_without_retrying(tmp_path):
    clock = FakeClock()
    http = FakeHttp(clock, {URL: [(404, {}, b"")]})

    with pytest.raises(ImageFetchError, match="404"):
        store(tmp_path, http, clock)(URL)
    assert len(http.requests) == 1


def test_formats_pil_cannot_open_fail_without_a_request(tmp_path):
    clock = FakeClock()
    svg = "https://upload.wikimedia.org/wikipedia/commons/a/ab/Logo.svg"
    http = FakeHttp(clock, {svg: []})

    with pytest.raises(ImageFetchError, match="unsupported"):
        store(tmp_path, http, clock)(svg)
    assert http.requests == []


def test_non_image_response_is_a_failure(tmp_path):
    clock = FakeClock()
    http = FakeHttp(clock, {URL: [ok(body=b"<html>error</html>", content_type="text/html")]})

    with pytest.raises(ImageFetchError, match="text/html"):
        store(tmp_path, http, clock)(URL)
    assert not list(tmp_path.rglob("*.jpg"))
