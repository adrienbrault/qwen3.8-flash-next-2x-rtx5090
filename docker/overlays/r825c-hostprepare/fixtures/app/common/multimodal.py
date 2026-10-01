from typing import List, TYPE_CHECKING

from pydantic import BaseModel, Field

from backends.exllamav3.vision import get_image_embedding_exl3, image_embedding_cache, image_key
from common.logger import xlogger

if TYPE_CHECKING:
    from backends.exllamav3.model import ExllamaV3Container


class MultimodalEmbeddingWrapper(BaseModel):
    """Common multimodal embedding wrapper"""

    content: list = Field(default_factory=list)
    text_alias: List[str] = Field(default_factory=list)

    # Cache keys of the images this request has resolved so far. They are
    # protected from eviction while the rest of the request is processed
    cache_keys: List[bytes] = Field(default_factory=list)
    uncached: int = 0

    async def add(self, container: "ExllamaV3Container", url: str):
        embedding, cached = await get_image_embedding_exl3(container, url, self.cache_keys)
        self.content.append(embedding)
        self.text_alias.append(embedding.text_alias)

        if cached:
            self.cache_keys.append(image_key(url))
        else:
            self.uncached += 1
            if self.uncached == 1:
                capacity_mb = image_embedding_cache.capacity_bytes // (1024 * 1024)
                xlogger.warning(
                    f"Image {len(self.content)} of this request does not fit in the "
                    f"multimodal embedding cache ({capacity_mb} MB) alongside the "
                    "request's other images, so it will be re-encoded on every turn. "
                    "Increase memory.sysmem_multimodal_cache to cache the whole context."
                )
