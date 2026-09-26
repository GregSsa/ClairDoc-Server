import asyncio
from pathlib import Path
from types import SimpleNamespace

from clairdoc_server.config import Settings
from clairdoc_server.embeddings import LOCAL_MODEL, EmbeddingService
from clairdoc_server.models import JobStatus, ProjectCreate
from clairdoc_server.rag import RagService
from clairdoc_server.storage import LocalStorage


def test_local_index_needs_no_openai_key(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    storage.initialize()
    storage.update_runtime({"embedding_provider": "local", "llm_model": "gpt-6-sol"})
    storage.set_llm_model("gpt-6-luna")
    assert storage.get_embedding_provider() == "local"
    project = storage.create_project(ProjectCreate(name="Test"))
    job = storage.create_job("document.txt", project.id, "document.txt")
    storage.source_path(job.id).write_text("Facture eau", encoding="utf-8")
    storage.text_path(job.id).write_text("Facture eau", encoding="utf-8")
    job.status = JobStatus.COMPLETED
    storage.save_job(job)
    rag = RagService(storage, Settings(data_dir=tmp_path, api_key="secret"))
    rag.embeddings._embed_local = lambda texts, model, query: [[1.0] * 384 for _ in texts]
    result = asyncio.run(rag.index_project(project.id))
    assert result.embedding_model == LOCAL_MODEL
    assert storage.read_index(project.id)["embedding_provider"] == "local"
    assert asyncio.run(rag.estimate_project(project.id)).estimated_cost_usd == 0
    storage.update_runtime({"embedding_provider": "openai"})
    vector = asyncio.run(rag.embeddings.query("eau", storage.read_index(project.id)))
    assert len(vector) == 384  # Old index keeps its own provider after settings change.


def test_legacy_index_keeps_openai_provider(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    storage.initialize()
    storage.update_runtime({"embedding_provider": "local"})
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0])])

    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    service = EmbeddingService(storage, Settings(api_key="secret"), lambda: client)
    vector = asyncio.run(
        service.query("question", {"embedding_model": "legacy-model", "embedding_dimensions": 2})
    )
    assert vector == [1.0, 0.0]
    assert calls[0]["model"] == "legacy-model"


def test_empty_document_indexes_name_and_refreshes_when_text_changes(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    storage.initialize()
    storage.update_runtime({"embedding_provider": "local"})
    project = storage.create_project(ProjectCreate(name="Maison"))
    job = storage.create_job("Plan maison.txt", project.id, "Plan maison.txt")
    storage.source_path(job.id).write_text("", encoding="utf-8")
    storage.text_path(job.id).write_text("", encoding="utf-8")
    job.status = JobStatus.COMPLETED
    storage.save_job(job)
    rag = RagService(storage, Settings(data_dir=tmp_path, api_key="test"))
    calls = []

    def embed(texts, model, query):
        calls.extend(texts)
        return [[1.0] * 384 for _ in texts]

    rag.embeddings._embed_local = embed
    result = asyncio.run(rag.index_project(project.id))
    assert calls == ["Plan maison.txt"]
    assert result.documents_indexed == 1
    assert result.documents_name_only == 1
    doc = storage.read_index(project.id)["documents"][0]
    assert doc["indexing_mode"] == "name_only"
    assert "Plan maison.txt" in doc["chunks"][0]["text"]
    assert "contenu inconnu" in doc["chunks"][0]["text"]
    assert rag.document_library(project.id).documents[0].status == "indexed_name"
    reused = asyncio.run(rag.index_project(project.id))
    assert reused.documents_reused == 1
    storage.text_path(job.id).write_text("Texte récupéré par OCR", encoding="utf-8")
    changed = asyncio.run(rag.index_project(project.id))
    assert changed.documents_indexed == 1
    assert changed.documents_name_only == 0
    assert rag.document_library(project.id).documents[0].status == "indexed"
