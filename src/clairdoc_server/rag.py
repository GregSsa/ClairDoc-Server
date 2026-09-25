import asyncio
import hashlib
import math
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from openai import AsyncOpenAI
from pypdf import PdfReader

from .config import Settings
from .models import (
    AskResponse,
    Citation,
    DocumentLibrary,
    DocumentRelationship,
    DocumentSummary,
    IndexEstimate,
    IndexResponse,
    JobStatus,
)
from .storage import LocalStorage, RecordNotFoundError

DEFAULT_SYSTEM_PROMPT = """Tu es l'assistant documentaire ClairDoc.
Réponds uniquement à partir des extraits de documents fournis.
Si les extraits ne permettent pas de répondre, dis-le clairement sans inventer.
Indique tes sources avec les repères [1], [2], etc. présents dans le contexte.
Réponds en français, de façon simple et précise."""

CATEGORY_RULES = {
    "Factures": ("facture", "invoice", "total ttc", "montant dû"),
    "Banque": ("banque", "relevé de compte", "iban", "prélèvement"),
    "Impôts": ("impôt", "fiscal", "déclaration de revenus", "taxe foncière"),
    "Assurances": ("assurance", "sinistre", "contrat d'assurance", "cotisation"),
    "Santé": ("santé", "médecin", "ordonnance", "mutuelle", "remboursement"),
    "Logement": ("loyer", "bail", "propriétaire", "locataire", "électricité", "gaz"),
    "Emploi": ("bulletin de paie", "salaire", "employeur", "contrat de travail"),
    "Identité": ("passeport", "carte nationale", "état civil", "acte de naissance"),
    "Courriers": ("objet :", "madame", "monsieur", "courrier"),
}
TOKEN_PATTERN = re.compile(r"[\wÀ-ÿ]{2,}", re.UNICODE)
DATE_PATTERN = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\b")
ISO_DATE_PATTERN = re.compile(r"\b(20\d{2}|19\d{2})-(\d{2})-(\d{2})\b")
AMOUNT_PATTERN = re.compile(
    r"\b\d{1,3}(?:[ .]\d{3})*(?:[,.]\d{2})?\s?(?:€|EUR)(?=\s|$)", re.I
)
PERSON_PATTERN = re.compile(
    r"\b(?:M(?:me|lle)?\.?|Monsieur|Madame)\s+([A-ZÀ-ÖØ-Ý][\wÀ-ÿ'-]+(?:\s+[A-ZÀ-ÖØ-Ý][\wÀ-ÿ'-]+){0,3})"
)


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


def _tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", text.lower())
    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return set(TOKEN_PATTERN.findall(normalized))


def keyword_similarity(question: str, text: str) -> float:
    query_tokens = _tokens(question)
    text_tokens = _tokens(text)
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def _extract_pdf_pages(path: Path) -> list[tuple[int | None, str]]:
    reader = PdfReader(path)
    return [(number, page.extract_text() or "") for number, page in enumerate(reader.pages, 1)]


def extract_metadata(text: str, filename: str) -> dict[str, Any]:
    sample = text[:12000]
    lowered = f"{filename}\n{sample}".lower()
    category = "Autres"
    for candidate, keywords in CATEGORY_RULES.items():
        if any(keyword in lowered for keyword in keywords):
            category = candidate
            break

    document_date: str | None = None
    if match := ISO_DATE_PATTERN.search(sample):
        document_date = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    elif match := DATE_PATTERN.search(sample):
        day, month, year = match.groups()
        if len(year) == 2:
            year = f"20{year}"
        document_date = f"{year}-{int(month):02d}-{int(day):02d}"

    organization = None
    for line in (line.strip() for line in sample.splitlines()[:30]):
        if 3 <= len(line) <= 80 and any(
            marker in line.lower()
            for marker in ("sarl", "sas", "assurance", "banque", "mutuelle", "caisse", "service")
        ):
            organization = line
            break

    return {
        "category": category,
        "document_type": category.removesuffix("s"),
        "date": document_date,
        "organization": organization,
        "people": sorted(set(PERSON_PATTERN.findall(sample)))[:10],
        "amounts": list(dict.fromkeys(AMOUNT_PATTERN.findall(sample)))[:10],
    }


class RagService:
    def __init__(self, storage: LocalStorage, settings: Settings) -> None:
        self.storage = storage
        self.settings = settings
        self._index_lock = asyncio.Lock()

    def _client(self) -> AsyncOpenAI:
        if not self.settings.openai_api_key:
            raise OpenAIConfigurationError("La clé OPENAI_API_KEY n'est pas configurée.")
        return AsyncOpenAI(
            api_key=self.settings.openai_api_key,
            max_retries=self.settings.openai_max_retries,
            timeout=self.settings.openai_timeout_seconds,
        )

    def _system_prompt(self) -> str:
        path = self.storage.rag_prompt_path()
        if not path.exists():
            path.write_text(DEFAULT_SYSTEM_PROMPT, encoding="utf-8")
        return path.read_text(encoding="utf-8").strip()

    def document_library(self, project_id: UUID) -> DocumentLibrary:
        self.storage.get_project(project_id)
        try:
            index = self.storage.read_index(project_id)
        except RecordNotFoundError:
            index = {"documents": []}
        indexed = {
            str(document.get("job_id")): document
            for document in index.get("documents", [])
            if isinstance(document, dict)
        }
        documents: list[DocumentSummary] = []
        for job in self.storage.jobs_for_project(project_id):
            if job.status != JobStatus.COMPLETED:
                continue
            record = indexed.get(str(job.id), {})
            metadata = record.get("metadata", {}) if isinstance(record, dict) else {}
            documents.append(
                DocumentSummary(
                    job_id=job.id,
                    name=job.original_filename,
                    source_relative_path=job.source_relative_path or job.original_filename,
                    status="indexed" if record else "ready",
                    category=str(metadata.get("category") or "À indexer"),
                    document_date=metadata.get("date"),
                    organization=metadata.get("organization"),
                    people=list(metadata.get("people") or []),
                    amounts=list(metadata.get("amounts") or []),
                    chunks=len(record.get("chunks", [])) if isinstance(record, dict) else 0,
                )
            )

        relationships: list[DocumentRelationship] = []
        for position, source in enumerate(documents):
            for target in documents[position + 1 :]:
                shared_people = sorted(set(source.people) & set(target.people))
                candidates = [
                    (
                        "organization",
                        source.organization,
                        bool(source.organization and source.organization == target.organization),
                    ),
                    ("person", shared_people[0] if shared_people else None, bool(shared_people)),
                    (
                        "category",
                        source.category,
                        source.category not in {"Autres", "À indexer"}
                        and source.category == target.category,
                    ),
                    (
                        "year",
                        source.document_date[:4] if source.document_date else None,
                        bool(
                            source.document_date
                            and target.document_date
                            and source.document_date[:4] == target.document_date[:4]
                        ),
                    ),
                ]
                for kind, label, matches in candidates:
                    if matches and label:
                        relationships.append(
                            DocumentRelationship(
                                source_job_id=source.job_id,
                                target_job_id=target.job_id,
                                kind=kind,
                                label=str(label),
                            )
                        )
        return DocumentLibrary(
            project_id=project_id,
            documents=documents,
            categories=sorted({document.category for document in documents}),
            relationships=relationships,
        )

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

    def _reusable_documents(self, project_id: UUID) -> dict[str, Any]:
        try:
            existing = self.storage.read_index(project_id)
        except RecordNotFoundError:
            return {}
        same_configuration = (
            existing.get("version") == 2
            and existing.get("embedding_model") == self.settings.embedding_model
            and existing.get("embedding_dimensions") == self.settings.embedding_dimensions
            and existing.get("chunk_chars") == self.settings.rag_chunk_chars
            and existing.get("chunk_overlap") == self.settings.rag_chunk_overlap
        )
        if not same_configuration:
            return {}
        return {
            str(document["job_id"]): document
            for document in existing.get("documents", [])
            if isinstance(document, dict) and "job_id" in document
        }

    async def estimate_project(self, project_id: UUID) -> IndexEstimate:
        self.storage.get_project(project_id)
        jobs = [
            job
            for job in self.storage.jobs_for_project(project_id)
            if job.status == JobStatus.COMPLETED and self.storage.text_path(job.id).is_file()
        ]
        reusable = self._reusable_documents(project_id)
        characters = 0
        reused = 0
        to_embed = 0
        for job in jobs:
            source_path = self.storage.source_path(job.id)
            document_hash = await asyncio.to_thread(_file_hash, source_path)
            previous = reusable.get(str(job.id))
            if previous and previous.get("document_hash") == document_hash:
                reused += 1
                continue
            to_embed += 1
            if self.storage.output_path(job.id).is_file():
                pages = await asyncio.to_thread(
                    _extract_pdf_pages, self.storage.output_path(job.id)
                )
                characters += sum(len(text) for _, text in pages)
            else:
                characters += len(
                    self.storage.text_path(job.id).read_text(
                        encoding="utf-8", errors="replace"
                    )
                )
        estimated_tokens = math.ceil(characters / 4)
        estimated_cost = (
            estimated_tokens / 1_000_000 * self.settings.embedding_price_per_million_usd
        )
        return IndexEstimate(
            project_id=project_id,
            documents_total=len(jobs),
            documents_to_embed=to_embed,
            documents_reused=reused,
            estimated_tokens=estimated_tokens,
            estimated_cost_usd=round(estimated_cost, 6),
            price_per_million_tokens_usd=self.settings.embedding_price_per_million_usd,
            embedding_model=self.settings.embedding_model,
        )

    async def index_project(self, project_id: UUID) -> IndexResponse:
        self._client()
        self.storage.get_project(project_id)
        async with self._index_lock:
            completed_jobs = [
                job
                for job in self.storage.jobs_for_project(project_id)
                if job.status == JobStatus.COMPLETED and self.storage.text_path(job.id).is_file()
            ]
            if not completed_jobs:
                raise NoDocumentsError("Aucun document analysé n'est disponible pour ce projet.")

            existing_documents = self._reusable_documents(project_id)

            documents: list[dict[str, Any]] = []
            reused = 0
            indexed = 0
            for job in completed_jobs:
                source_path = self.storage.source_path(job.id)
                document_hash = await asyncio.to_thread(_file_hash, source_path)
                previous = existing_documents.get(str(job.id))
                if previous and previous.get("document_hash") == document_hash:
                    documents.append(previous)
                    reused += 1
                    continue

                if self.storage.output_path(job.id).is_file():
                    pages = await asyncio.to_thread(
                        _extract_pdf_pages, self.storage.output_path(job.id)
                    )
                else:
                    text = self.storage.text_path(job.id).read_text(
                        encoding="utf-8", errors="replace"
                    )
                    pages = [(None, text)]
                full_text = "\n".join(text for _, text in pages)
                metadata = extract_metadata(full_text, job.original_filename)
                chunk_records: list[dict[str, Any]] = []
                chunk_texts: list[str] = []
                for page_number, page_text in pages:
                    values = chunk_text(
                        page_text,
                        self.settings.rag_chunk_chars,
                        self.settings.rag_chunk_overlap,
                    )
                    for value in values:
                        chunk_records.append(
                            {"index": len(chunk_records), "page_number": page_number, "text": value}
                        )
                        chunk_texts.append(value)
                if not chunk_records:
                    continue
                embeddings = await self._embeddings(chunk_texts)
                for record, embedding in zip(chunk_records, embeddings, strict=True):
                    record["embedding"] = embedding
                documents.append(
                    {
                        "job_id": str(job.id),
                        "document_name": job.original_filename,
                        "source_relative_path": job.source_relative_path or job.original_filename,
                        "document_hash": document_hash,
                        "metadata": metadata,
                        "chunks": chunk_records,
                    }
                )
                indexed += 1

            if not documents:
                raise NoDocumentsError("Aucun texte exploitable n'a été trouvé dans les documents.")
            payload: dict[str, object] = {
                "version": 2,
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
        client = self._client()
        query = await client.embeddings.create(
            model=str(index["embedding_model"]),
            input=question,
            dimensions=int(index["embedding_dimensions"]),
        )
        query_vector = query.data[0].embedding
        ranked: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for document in index.get("documents", []):
            metadata_text = " ".join(str(value) for value in document.get("metadata", {}).values())
            for chunk in document.get("chunks", []):
                vector_score = (cosine_similarity(query_vector, chunk["embedding"]) + 1) / 2
                keyword_score = keyword_similarity(question, f"{chunk['text']} {metadata_text}")
                ranked.append((0.75 * vector_score + 0.25 * keyword_score, document, chunk))
        selected = sorted(ranked, key=lambda item: item[0], reverse=True)[
            : top_k or self.settings.rag_top_k
        ]
        if not selected:
            raise NoDocumentsError("L'index ne contient aucun extrait exploitable.")

        context_parts = []
        for number, (_, document, chunk) in enumerate(selected, start=1):
            page = f", page {chunk['page_number']}" if chunk.get("page_number") else ""
            context_parts.append(
                f"[{number}] Document: {document['document_name']}{page}\n{chunk['text']}"
            )
        response = await client.responses.create(
            model=self.storage.get_llm_model(self.settings.llm_model),
            instructions=self._system_prompt(),
            input=f"Question de l'utilisateur :\n{question}\n\nExtraits :\n\n"
            + "\n\n".join(context_parts),
        )
        citations = [
            Citation(
                document_name=str(document["document_name"]),
                job_id=UUID(str(document["job_id"])),
                chunk_index=int(chunk["index"]),
                page_number=chunk.get("page_number"),
                score=round(score, 4),
                excerpt=str(chunk["text"])[:300],
            )
            for score, document, chunk in selected
        ]
        return AskResponse(
            answer=response.output_text,
            citations=citations,
            model=self.storage.get_llm_model(self.settings.llm_model),
        )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as document:
        while block := document.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()
