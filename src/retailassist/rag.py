"""Semantic search over the store's policy documents.

Each "## " section of each policy document becomes one chunk in a Chroma vector database, embedded
with a small local model (all-MiniLM-L6-v2, downloaded once on first use). Search results are turned
into Claude `search_result` blocks, so Claude's answers come back with citations pointing at the
exact policy section and passage they used.

The index is rebuilt automatically whenever the policy files change.
"""

import hashlib
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.api.types import EmbeddingFunction

from .store import DATA_DIR

POLICY_DIR = DATA_DIR / "policies"
COLLECTION = "store_policies"
# Cosine distance runs from 0 (same meaning) to 2 (opposite). Results further away than this are dropped.
MAX_DISTANCE = 0.75


@dataclass
class PolicyChunk:
    id: str             # e.g. "refund-policy#refunds"
    source: str         # where it came from, e.g. "data/policies/refund-policy.md#refunds"
    title: str          # e.g. "Returns & Refund Policy > Refunds"
    passages: list[str]  # paragraphs and list items; Claude cites these individually

    @property
    def text(self) -> str:
        return "\n".join(self.passages)


@dataclass
class PolicyHit:
    chunk: PolicyChunk
    distance: float

    def to_search_result(self) -> dict:
        """Format as a Claude search_result content block, with citations turned on."""
        return {
            "type": "search_result",
            "source": self.chunk.source,
            "title": self.chunk.title,
            "content": [{"type": "text", "text": passage} for passage in self.chunk.passages],
            "citations": {"enabled": True},
        }


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def load_chunks(policy_dir: Path = POLICY_DIR) -> list[PolicyChunk]:
    """Split every policy document into one chunk per "## " section."""
    chunks = []
    for path in sorted(policy_dir.glob("*.md")):
        text = path.read_text()
        doc_title = text.splitlines()[0].lstrip("# ").strip()
        for section in text.split("\n## ")[1:]:
            heading, _, body = section.partition("\n")
            passages = _split_passages(body)
            if not passages:
                continue
            anchor = _slug(heading)
            chunks.append(PolicyChunk(
                id=f"{path.stem}#{anchor}",
                source=f"{_display_path(path)}#{anchor}",
                title=f"{doc_title} > {heading.strip()}",
                passages=passages,
            ))
    return chunks


def _display_path(path: Path) -> str:
    """The path relative to the project root, e.g. data/policies/refund-policy.md."""
    try:
        return str(path.resolve().relative_to(DATA_DIR.parent))
    except ValueError:
        return path.name


def _split_passages(body: str) -> list[str]:
    """Split a section into paragraphs, list items, and tables, so citations can point at each one."""
    passages = []
    for block in re.split(r"\n\s*\n", body.strip()):
        lines = [line for line in block.splitlines() if line.strip()]
        if lines and all(re.match(r"\s*([-*]|\d+\.)\s", line) for line in lines):
            passages.extend(line.strip() for line in lines)  # one passage per list item
        elif lines:
            passages.append("\n".join(lines))
    return passages


class PolicyIndex:
    def __init__(self, path: str | os.PathLike, embedding_function: EmbeddingFunction | None = None,
                 policy_dir: Path = POLICY_DIR):
        self._client = chromadb.PersistentClient(path=str(path))
        self._embedding_function = embedding_function
        self._policy_dir = policy_dir
        self._chunks: dict[str, PolicyChunk] = {}
        self._collection = None
        self.refresh()

    def refresh(self) -> None:
        """Load the index, rebuilding it if the policy documents have changed since it was built."""
        chunks = load_chunks(self._policy_dir)
        fingerprint = hashlib.sha256("".join(f"{c.id}{c.text}" for c in chunks).encode()).hexdigest()
        self._chunks = {c.id: c for c in chunks}

        kwargs = {"embedding_function": self._embedding_function} if self._embedding_function else {}
        collection = self._client.get_or_create_collection(
            COLLECTION, configuration={"hnsw": {"space": "cosine"}}, **kwargs)
        if (collection.metadata or {}).get("fingerprint") == fingerprint:
            self._collection = collection
            return

        self._client.delete_collection(COLLECTION)
        collection = self._client.create_collection(
            COLLECTION, configuration={"hnsw": {"space": "cosine"}}, metadata={"fingerprint": fingerprint}, **kwargs)
        collection.add(
            ids=[c.id for c in chunks],
            documents=[f"{c.title}\n{c.text}" for c in chunks],
            metadatas=[{"source": c.source, "title": c.title} for c in chunks],
        )
        self._collection = collection

    def search(self, query: str, limit: int = 3, max_distance: float = MAX_DISTANCE) -> list[PolicyHit]:
        result = self._collection.query(query_texts=[query], n_results=min(limit, len(self._chunks)))
        hits = []
        for chunk_id, distance in zip(result["ids"][0], result["distances"][0]):
            if distance <= max_distance:
                hits.append(PolicyHit(self._chunks[chunk_id], distance))
        return hits

    def __len__(self) -> int:
        return self._collection.count()


@lru_cache
def get_policy_index() -> PolicyIndex:
    return PolicyIndex(os.environ.get("RETAILASSIST_VECTOR_DB", DATA_DIR / "vectordb"))
