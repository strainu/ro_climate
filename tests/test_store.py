from collections import deque

import pytest
import requests

from ro_climate._store import RequestsStore
from ro_climate.exceptions import StoreUnavailableError


class FakeResponse:
    def __init__(self, status_code=200, content=b""):
        self.status_code = status_code
        self.content = content


class FakeSession:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(("GET", url))
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response

    def head(self, url, timeout=None):
        self.calls.append(("HEAD", url))
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response


def make_store(responses, **kwargs):
    session = FakeSession(responses)
    return (
        RequestsStore("https://example.invalid/store", session, **kwargs),
        session,
    )


def test_getitem():
    store, session = make_store([FakeResponse(content=b"payload")])
    assert store["max_temp/0.0.0"] == b"payload"
    assert session.calls == [
        ("GET", "https://example.invalid/store/max_temp/0.0.0")
    ]


def test_getitem_missing():
    store, _ = make_store([FakeResponse(status_code=404)])
    with pytest.raises(KeyError):
        store["missing"]


def test_getitem_retries_server_errors(monkeypatch):
    monkeypatch.setattr("ro_climate._store.time.sleep", lambda _: None)
    store, session = make_store(
        [FakeResponse(status_code=500), FakeResponse(content=b"ok")],
        retries=2,
    )
    assert store["a"] == b"ok"
    assert len(session.calls) == 2


def test_getitem_gives_up(monkeypatch):
    monkeypatch.setattr("ro_climate._store.time.sleep", lambda _: None)
    store, _ = make_store(
        [FakeResponse(status_code=503), FakeResponse(status_code=503)],
        retries=2,
    )
    with pytest.raises(StoreUnavailableError):
        store["a"]


def test_getitem_network_error(monkeypatch):
    monkeypatch.setattr("ro_climate._store.time.sleep", lambda _: None)
    store, _ = make_store([requests.ConnectionError("boom")], retries=1)
    with pytest.raises(StoreUnavailableError):
        store["a"]


def test_getitems_skips_missing():
    store, _ = make_store(
        [
            FakeResponse(content=b"a"),
            FakeResponse(status_code=404),
            FakeResponse(content=b"c"),
        ]
    )
    assert store.getitems(["a", "missing", "c"], contexts={}) == {
        "a": b"a",
        "c": b"c",
    }


def test_contains_uses_head():
    store, session = make_store(
        [FakeResponse(status_code=200), FakeResponse(status_code=404)]
    )
    assert "present" in store
    assert "missing" not in store
    assert session.calls == [
        ("HEAD", "https://example.invalid/store/present"),
        ("HEAD", "https://example.invalid/store/missing"),
    ]


def test_contains_falls_back_to_get():
    store, session = make_store(
        [FakeResponse(status_code=405), FakeResponse(content=b"data")]
    )
    assert "x/0" in store
    assert session.calls == [
        ("HEAD", "https://example.invalid/store/x/0"),
        ("GET", "https://example.invalid/store/x/0"),
    ]


def test_store_is_read_only():
    store, _ = make_store([])
    with pytest.raises(TypeError):
        store["x"] = b"value"
    with pytest.raises(TypeError):
        del store["x"]


def test_base_url_normalisation():
    session = FakeSession([])
    store = RequestsStore("https://example.invalid/store", session)
    assert store.base_url == "https://example.invalid/store/"
