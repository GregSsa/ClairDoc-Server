import asyncio
from typing import Any

LOCAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LOCAL_DIMENSIONS = 384


class EmbeddingService:
    def __init__(self, storage: Any, settings: Any, client_factory: Any) -> None:
        self.storage = storage
        self.settings = settings
        self.client_factory = client_factory
        self._local_model: Any = None
        self._local_lock = asyncio.Lock()

    def configuration(self) -> tuple[str, str, int]:
        provider = self.storage.get_embedding_provider(self.settings.embedding_provider)
        if provider == "local":
            return provider, LOCAL_MODEL, LOCAL_DIMENSIONS
        return "openai", self.settings.embedding_model, self.settings.embedding_dimensions

    async def embed(
        self,
        texts: list[str],
        configuration: tuple[str, str, int] | None = None,
        query: bool = False,
    ) -> list[list[float]]:
        provider, model, dimensions = configuration or self.configuration()
        if provider == "local":
            async with self._local_lock:
                return await asyncio.to_thread(self._embed_local, texts, model, query)
        vectors: list[list[float]] = []
        client = self.client_factory()
        for start in range(0, len(texts), 64):
            response = await client.embeddings.create(
                model=model, input=texts[start : start + 64], dimensions=dimensions
            )
            vectors.extend(item.embedding for item in response.data)
        return vectors

    def _embed_local(self, texts: list[str], model: str, query: bool) -> list[list[float]]:
        from fastembed import TextEmbedding

        if self._local_model is None:
            self._local_model = TextEmbedding(
                model_name=model,
                cache_dir=str(self.storage.root / "models"),
                threads=2,
            )
        values = (
            self._local_model.query_embed(texts)
            if query
            else self._local_model.passage_embed(texts)
        )
        return [vector.tolist() for vector in values]

    async def query(self, text: str, index: dict[str, Any]) -> list[float]:
        configuration = (
            str(index.get("embedding_provider") or "openai"),
            str(index["embedding_model"]),
            int(index["embedding_dimensions"]),
        )
        return (await self.embed([text], configuration, query=True))[0]
