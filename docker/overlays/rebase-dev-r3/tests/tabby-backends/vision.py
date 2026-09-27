"""Vision utilities for ExLlamaV3."""

from collections import OrderedDict
from hashlib import blake2b
from typing import TYPE_CHECKING, Iterable, Optional

from common.optional_dependencies import dependencies
from common.image_util import get_image
from common.logger import xlogger

# Since this is used outside the Exl3 backend, the dependency
# may be optional
if dependencies.exllamav3:
    from exllamav3.tokenizer import MMEmbedding

if TYPE_CHECKING:
    from backends.exllamav3.model import ExllamaV3Container

DEFAULT_CACHE_MB = 1024


def image_key(url: str) -> bytes:
    return blake2b(url.encode("utf-8"), digest_size=16).digest()


def embedding_nbytes(embedding) -> int:
    """Storage size of an MMEmbedding: the embedding tensor plus any deepstack tensors."""

    tensors = [embedding.embeddings]
    tensors += list(getattr(embedding, "deepstack_embeddings", None) or [])
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


class ImageEmbeddingCache:
    """
    LRU cache of image embeddings keyed by image URL, bounded by the storage size
    of the embeddings rather than a fixed count.

    Eviction never touches entries the current request has already resolved. A
    context with more images than the budget holds is cached only as far as the
    budget allows, instead of every image evicting the previous one and defeating
    the cache for the whole context on the next turn.
    """

    def __init__(self, capacity_mb: int = DEFAULT_CACHE_MB):
        self._entries: OrderedDict[bytes, tuple[str, "MMEmbedding", int]] = OrderedDict()
        self.capacity_bytes = 0
        self.size_bytes = 0
        self.configure(capacity_mb)

    def configure(self, capacity_mb: Optional[int]):
        """Set the budget in MB. Entries beyond the new budget are dropped, oldest first."""

        capacity_mb = DEFAULT_CACHE_MB if capacity_mb is None else max(capacity_mb, 0)
        self.capacity_bytes = capacity_mb * 1024 * 1024
        self._evict(protected=(), target=self.capacity_bytes)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: bytes) -> bool:
        return key in self._entries

    def get(self, key: bytes, url: str) -> Optional["MMEmbedding"]:
        entry = self._entries.get(key)
        if entry is None or entry[0] != url:
            return None

        self._entries.move_to_end(key)
        return entry[1]

    def put(self, key: bytes, url: str, embedding, protected: Iterable[bytes] = ()) -> bool:
        """
        Store an embedding, evicting least recently used entries to make room.
        Entries whose keys are in `protected` are never evicted. Returns False, and
        stores nothing, when the embedding doesn't fit without evicting one of them.
        """

        size = embedding_nbytes(embedding)
        protected = set(protected)

        if key in self._entries:
            self._remove(key)

        if size > self.capacity_bytes:
            return False

        self._evict(protected, target=self.capacity_bytes - size)
        if self.size_bytes + size > self.capacity_bytes:
            return False

        self._entries[key] = (url, embedding, size)
        self.size_bytes += size
        return True

    def clear(self):
        self._entries.clear()
        self.size_bytes = 0

    def _remove(self, key: bytes):
        _, _, size = self._entries.pop(key)
        self.size_bytes -= size

    def _evict(self, protected, target: int):
        """Drop unprotected entries, oldest first, until the cache holds at most `target` bytes."""

        for key in list(self._entries):
            if self.size_bytes <= target:
                break
            if key not in protected:
                self._remove(key)


image_embedding_cache = ImageEmbeddingCache()


async def get_image_embedding_exl3(
    container: "ExllamaV3Container",
    url: str,
    protected: Iterable[bytes] = (),
) -> tuple["MMEmbedding", bool]:
    """
    Fetch or compute the embedding for an image URL. Returns the embedding and
    whether it is held in the cache afterwards. `protected` lists the cache keys
    the calling request already depends on; those are never evicted to make room.
    """

    key = image_key(url)

    embedding = image_embedding_cache.get(key, url)
    if embedding is not None:
        return embedding, True

    image = await get_image(url)
    embedding = container.vision_model.get_image_embeddings(
        tokenizer=container.tokenizer,
        image=image,
        text_alias=None,
    )

    cached = image_embedding_cache.put(key, url, embedding, protected)
    size = embedding_nbytes(embedding)

    xlogger.debug(
        f"Created MMEmbedding: {embedding.mm_length} tokens, {size / (1024 * 1024):.1f} MB, "
        f"{'cached' if cached else 'not cached'} "
        f"({len(image_embedding_cache)} entries, "
        f"{image_embedding_cache.size_bytes / (1024 * 1024):.0f} MB in cache)",
        {
            "text_alias": embedding.text_alias,
            "metadata": embedding.metadata,
            "token_length": embedding.mm_length,
            "size_bytes": size,
            "cached": cached,
            "cache_entries": len(image_embedding_cache),
            "cache_size_bytes": image_embedding_cache.size_bytes,
        },
    )

    return embedding, cached


def clear_image_embedding_cache():
    image_embedding_cache.clear()
