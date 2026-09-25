import asyncio
import hashlib
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from openai import AsyncOpenAI
from pypdf import PdfReader

from .config import Settings
from .models import AskResponse, Citation, IndexResponse, JobStatus
from .storage import LocalStorage, RecordNotFoundError

DEFAULT_SYSTEM_PROMPT = """Tu es l'assistant documentaire ClairDoc.
Réponds uniquement à partir des extraits de documents fournis.
Si les extraits ne permettent pas de répondre, dis-le clairement sans inventer.
Indique tes sources avec les repères [1], [2], etc. présents dans le contexte.
Réponds en français, de façon simple et précise."""


class OpenAIConfigurationError(RuntimeError):
    pass


class NoDocumentsError(RuntimeError):
    pass


class ProjectIndexNotFoundError(RuntimeError):
    pass


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    normalized = "\n".join(line.strip() for line in text.splitlines())
    normalized = "\n".join(part for part in normalized.splitlines() if part)
    if not normalized:
        return []
    if overlap >= size:
        overlap = size // 4

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + size, len(normalized))
        if end < len(normalized):
            boundary = max(normalized.rfind("\n", start, end), normalized.rfind(". ", start, end))
            if boundary > start + size // 2:
                end = boundary + 1
        value = normalized[start:end].strip()
        if value:
            chunks.append(value)
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _extract_pdf_text(path: Path) -> str:
    reader = PdfReader(path)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()


class RagService:
    def __init__(self, storage: LocalStorage, settings: Settings) -> None:
        self.storage = storage
        self.settings = settings
        self._index_lock = asyncio.Lock()

    def _client(self) -> AsyncOpenAI:
        if not self.settings.openai_api_key:
            raise OpenAIConfigurationError("La clé OPENAI_API_KEY n'est pas configurée.")
        return AsyncOpenAI(api_key=self.settings.openai_api_key)

    def _system_prompt(self) -> str:
        path = self.storage.rag_prompt_path()
        if not path.exists():
            path.write_text(DEFAULT_SYSTEM_PROMPT, encoding="utf-8")
        return path.read_text(encoding="utf-8").strip()

    async def _embeddings(self, texts: list[str]) -> list[list[float]]:
        client = self._client()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 64):
            response = await client.embeddings.create(
                model=self.settings.embedding_model,
                input=texts[start : start + 64],
                dimensions=self.settings.embedding_dimensions,
            )
            vectors.extend(item.embedding for item in response.data)
        return vectors

    async def index_project(self, project_id: UUID) -> IndexResponse:
        self._client()
        self.storage.get_project(project_id)
        async with self._index_lock:
            completed_jobs = [
                job
                for job in self.storage.iter_jobs()
                if job.project_id == project_id
                and job.status == JobStatus.COMPLETED
                and self.storage.output_path(job.id).is_file()
            ]
            if not completed_jobs:
                raise NoDocumentsError("Aucun document OCRisé n'est disponible pour ce projet.")

            existing_documents: dict[str, Any] = {}
            try:
                existing = self.storage.read_index(project_id)
                same_configuration = (
                    existing.get("embedding_model") == self.settings.embedding_model
                    and existing.get("embedding_dimensions") == self.settings.embedding_dimensions
                    and existing.get("chunk_chars") == self.settings.rag_chunk_chars
                    and existing.get("chunk_overlap") == self.settings.rag_chunk_overlap
                )
                if same_configuration:
                    existing_documents = {
                        str(document["job_id"]): document
                        for document in existing.get("documents", [])
                        if isinstance(document, dict) and "job_id" in document
                    }
            except RecordNotFoundError:
                pass

            documents: list[dict[str, Any]] = []
            reused = 0
            indexed = 0
            for job in sorted(completed_jobs, key=lambda item: item.created_at):
                pdf_path = self.storage.output_path(job.id)
                document_hash = await asyncio.to_thread(_file_hash, pdf_path)
                previous = existing_documents.get(str(job.id))
                if previous and previous.get("document_hash") == document_hash:
                    documents.append(previous)
                    reused += 1
                    continue

                text = await asyncio.to_thread(_extract_pdf_text, pdf_path)
                if not text and self.storage.text_path(job.id).is_file():
                    text = self.storage.text_path(job.id).read_text(
                        encoding="utf-8", errors="replace"
                    )
                chunks = chunk_text(
                    text,
                    self.settings.rag_chunk_chars,
                    self.settings.rag_chunk_overlap,
                )
                if not chunks:
                    continue
                embeddings = await self._embeddings(chunks)
                documents.append(
                    {
                        "job_id": str(job.id),
                        "document_name": job.original_filename,
                        "document_hash": document_hash,
                        "chunks": [
                            {"index": position, "text": chunk, "embedding": embedding}
                            for position, (chunk, embedding) in enumerate(
                                zip(chunks, embeddings, strict=True)
                            )
                        ],
                    }
                )
                indexed += 1

            if not documents:
                raise NoDocumentsError("Aucun texte exploitable n'a été trouvé dans les documents.")

            payload: dict[str, object] = {
                "version": 1,
                "project_id": str(project_id),
                "created_at": datetime.now(UTC).isoformat(),
                "embedding_model": self.settings.embedding_model,
                "embedding_dimensions": self.settings.embedding_dimensions,
                "chunk_chars": self.settings.rag_chunk_chars,
                "chunk_overlap": self.settings.rag_chunk_overlap,
                "documents": documents,
            }
            self.storage.write_index(project_id, payload)
            return IndexResponse(
                project_id=project_id,
                documents_indexed=indexed,
                documents_reused=reused,
                chunks_indexed=sum(len(document["chunks"]) for document in documents),
                embedding_model=self.settings.embedding_model,
            )

    async def ask(self, project_id: UUID, question: str, top_k: int | None) -> AskResponse:
        try:
            index = self.storage.read_index(project_id)
        except RecordNotFoundError as exc:
            raise ProjectIndexNotFoundError(
                "Le projet doit être indexé avant de poser une question."
            ) from exc

        model = str(index["embedding_model"])
        dimensions = int(index["embedding_dimensions"])
        client = self._client()
        query = await client.embeddings.create(
            model=model,
            input=question,
            dimensions=dimensions,
        )
        query_vector = query.data[0].embedding
        ranked: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for document in index.get("documents", []):
            for chunk in document.get("chunks", []):
                ranked.append(
                    (cosine_similarity(query_vector, chunk["embedding"]), document, chunk)
                )
        limit = top_k or self.settings.rag_top_k
        selected = sorted(ranked, key=lambda item: item[0], reverse=True)[:limit]
        if not selected:
            raise NoDocumentsError("L'index ne contient aucun extrait exploitable.")

        context_parts = [
            f"[{number}] Document: {document['document_name']}\n{chunk['text']}"
            for number, (_, document, chunk) in enumerate(selected, start=1)
        ]
        response = await client.responses.create(
            model=self.settings.llm_model,
            instructions=self._system_prompt(),
            input=f"Question de l'utilisateur :\n{question}\n\nExtraits :\n\n"
            + "\n\n".join(context_parts),
        )
        citations = [
            Citation(
                document_name=str(document["document_name"]),
                job_id=UUID(str(document["job_id"])),
                chunk_index=int(chunk["index"]),
                score=round(score, 4),
                excerpt=str(chunk["text"])[:300],
            )
            for score, document, chunk in selected
        ]
        return AskResponse(
            answer=response.output_text,
            citations=citations,
            model=self.settings.llm_model,
        )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as document:
        while block := document.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()
