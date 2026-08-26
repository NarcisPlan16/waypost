"""Fixture: a plausible Python module, exercising every form parse.py claims.

Not a toy. The point of the fixture is that the shapes here are the ones a
real repository contains -- decorators, properties, nested classes, async
defs, dunder methods, module constants -- so a recall number measured against
it means something.
"""

from __future__ import annotations

import json
import os.path
from dataclasses import dataclass
from typing import Any as AnyValue

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
_INTERNAL_SENTINEL = object()


@dataclass
class Config:
    """Runtime configuration."""

    timeout: int = DEFAULT_TIMEOUT

    def to_dict(self) -> dict[str, AnyValue]:
        """Serialise to a plain dict."""
        return {"timeout": self.timeout}

    @classmethod
    def from_env(cls) -> Config:
        """Build a Config from environment variables."""
        return cls(timeout=int(os.path.basename("30")))

    def validate(self) -> bool:
        """Whether this configuration is usable."""
        return self._check(self.timeout)

    def _check(self, value: int) -> bool:
        return value > 0


class BaseClient:
    """Shared transport behaviour."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def request(self, method: str, url: str):
        """Issue a single request."""
        raise NotImplementedError

    def close(self) -> None:
        """Release the underlying connection."""


class Client(BaseClient):
    """HTTP client with retry and pooling."""

    def request(self, method: str, url: str, *, timeout: int | None = None):
        """Issue a request, retrying on transient failures."""
        return self._retry(method, url)

    def get(self, url: str):
        """Convenience wrapper for GET."""
        return self.request("GET", url)

    def post(self, url: str, body: AnyValue):
        """Convenience wrapper for POST."""
        return self.request("POST", url)

    def _retry(self, method: str, url: str):
        return None

    @property
    def user_agent(self) -> str:
        """The User-Agent header this client sends."""
        return "waypost"

    def __repr__(self) -> str:
        return "Client()"

    class Pool:
        """Connection pool bound to one client."""

        def acquire(self):
            """Take a connection out of the pool."""

        def release(self, conn) -> None:
            """Return a connection to the pool."""


def build_client(config: Config | None = None) -> Client:
    """Construct a Client with sensible defaults."""
    return Client(config or Config())


def _normalise(url: str) -> str:
    return url.strip()


def parse_payload(raw: str) -> AnyValue:
    """Decode a JSON payload, returning None when it is not JSON."""

    def _attempt(text: str):
        return json.loads(text)

    return _attempt(raw)


async def fetch_all(urls: list[str]) -> list[AnyValue]:
    """Fetch every URL concurrently."""
    return [parse_payload(_normalise(u)) for u in urls]


def main() -> int:
    """Entry point."""
    build_client()
    return 0


SCHEMA_VERSION = 1


class Registry:
    """Name-to-factory lookup."""

    def register(self, name: str, factory) -> None:
        """Bind a name to a factory."""

    def resolve(self, name: str):
        """Look a name up, or raise KeyError."""
        raise KeyError(name)


def _trace(fn):
    """Decorator: mark a callable as traced."""
    return fn


@_trace
@dataclass
class Envelope:
    """A request envelope. Two stacked decorators, on purpose."""

    body: str


@_trace
async def stream_pages(urls: list[str]):
    """Async generator: yield each normalised URL as it is produced."""
    for url in urls:
        yield _normalise(url)


if os.path.sep == "/":
    # Block-scoped even at module level. Python's own rule already excludes
    # this (the parent is an `if`, not the module), and the fixture pins it.
    POSIX_ONLY = True
