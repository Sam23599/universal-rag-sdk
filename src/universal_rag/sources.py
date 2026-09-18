"""Composable document sources for files, APIs, databases, and in-memory data."""

from __future__ import annotations

import asyncio
import inspect
import mimetypes
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Iterable
from pathlib import Path

from .models import RawDocument


class IterableSource:
    def __init__(self, documents: Iterable[RawDocument]) -> None:
        self._documents = documents

    async def __aiter__(self) -> AsyncIterator[RawDocument]:
        for document in self._documents:
            yield document


class AsyncIterableSource:
    def __init__(self, documents: AsyncIterable[RawDocument]) -> None:
        self._documents = documents

    async def __aiter__(self) -> AsyncIterator[RawDocument]:
        async for document in self._documents:
            yield document


class CallableSource:
    """Adapter for API or database loaders returning sync/async iterables."""

    def __init__(
        self,
        loader: Callable[[], Iterable[RawDocument] | AsyncIterable[RawDocument] | Awaitable[Iterable[RawDocument]]],
    ) -> None:
        self._loader = loader

    async def __aiter__(self) -> AsyncIterator[RawDocument]:
        loaded = self._loader()
        if inspect.isawaitable(loaded):
            loaded = await loaded
        if isinstance(loaded, AsyncIterable):
            async for document in loaded:
                yield document
        else:
            for document in loaded:
                yield document


class DirectorySource:
    """Read a bounded directory tree without blocking the event loop."""

    def __init__(
        self,
        path: str | Path,
        *,
        recursive: bool = True,
        extensions: Iterable[str] | None = None,
        metadata: dict | None = None,
        acl: Iterable[str] = (),
    ) -> None:
        self.path = Path(path)
        self.recursive = recursive
        self.extensions = {value.casefold() for value in extensions} if extensions else None
        self.metadata = dict(metadata or {})
        self.acl = tuple(acl)

    async def __aiter__(self) -> AsyncIterator[RawDocument]:
        if not self.path.is_dir():
            raise ValueError(f"Document source directory does not exist: {self.path}")
        paths = await asyncio.to_thread(
            lambda: sorted(
                path for path in (self.path.rglob("*") if self.recursive else self.path.glob("*")) if path.is_file()
            )
        )
        for path in paths:
            if self.extensions is not None and path.suffix.casefold() not in self.extensions:
                continue
            content = await asyncio.to_thread(path.read_bytes)
            relative = path.relative_to(self.path).as_posix()
            yield RawDocument(
                id=relative,
                content=content,
                mime_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                source_uri=path.resolve().as_uri(),
                metadata={**self.metadata, "file_name": path.name, "relative_path": relative},
                acl=self.acl,
            )
