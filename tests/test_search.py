from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient

from clairdoc_server.config import Settings
from clairdoc_server.main import create_app


def _client(tmp_path: Path) -> TestClient:
    return TestClient(
        create_app(
            Settings(
                api_key="secret",
                openai_api_key="",
                data_dir=tmp_path / "data",
                ocr_command="missing-ocrmypdf-command",
            )
        )
    )


def _index(client: TestClient, monkeypatch: object) -> tuple[str, str, str]:
    headers = {"X-ClairDoc-Key": "secret"}
    project_id = client.post("/api/v1/projects", headers=headers, json={"name": "Archives"}).json()[
        "id"
    ]
    bill_id, image_id = str(uuid4()), str(uuid4())
    client.app.state.storage.write_index(
        project_id,
        {
            "embedding_provider": "local",
            "embedding_model": "test",
            "embedding_dimensions": 2,
            "documents": [
                {
                    "job_id": bill_id,
                    "document_name": "facture-electricite.pdf",
                    "source_relative_path": "maison/facture-electricite.pdf",
                    "indexing_mode": "content",
                    "metadata": {"category": "Factures"},
                    "chunks": [
                        {
                            "index": 0,
                            "page_number": 2,
                            "text": "Facture électricité mars 2025",
                            "embedding": [1.0, 0.0],
                        }
                    ],
                },
                {
                    "job_id": image_id,
                    "document_name": "plan-maison.pdf",
                    "source_relative_path": "plans/plan-maison.pdf",
                    "indexing_mode": "name_only",
                    "metadata": {"category": "Logement"},
                    "chunks": [
                        {
                            "index": 0,
                            "page_number": None,
                            "text": "Nom du fichier uniquement : plan-maison.pdf.",
                            "embedding": [0.0, 1.0],
                        }
                    ],
                },
            ],
        },
    )

    async def query(_text: str, _index: dict) -> list[float]:
        return [1.0, 0.0]

    monkeypatch.setattr(client.app.state.rag.embeddings, "query", query)
    return project_id, bill_id, image_id


def test_local_search_returns_ranked_verbatim_passages_and_name_only(
    tmp_path: Path, monkeypatch: object
) -> None:
    with _client(tmp_path) as client:
        project_id, bill_id, image_id = _index(client, monkeypatch)
        headers = {"X-ClairDoc-Key": "secret"}
        path = f"/api/v1/projects/{project_id}/search"
        assert client.post(path, json={"query": "facture électricité"}).status_code == 401
        response = client.post(path, headers=headers, json={"query": "facture électricité"})
        assert response.status_code == 200
        results = response.json()["results"]
        assert results[0]["job_id"] == bill_id
        assert results[0]["passages"][0]["text"] == "Facture électricité mars 2025"
        assert results[0]["passages"][0]["page_number"] == 2
        assert next(item for item in results if item["job_id"] == image_id)["passages"] == []


def test_ai_search_selects_only_existing_passages(tmp_path: Path, monkeypatch: object) -> None:
    with _client(tmp_path) as client:
        project_id, bill_id, _ = _index(client, monkeypatch)

        class Responses:
            async def create(self, **_kwargs: object) -> SimpleNamespace:
                return SimpleNamespace(output_text='{"ids":[0,999]}')

        monkeypatch.setattr(
            client.app.state.rag, "_client", lambda: SimpleNamespace(responses=Responses())
        )
        response = client.post(
            f"/api/v1/projects/{project_id}/search",
            headers={"X-ClairDoc-Key": "secret"},
            json={"query": "facture électricité", "mode": "ai"},
        )
        assert response.status_code == 200
        assert [item["job_id"] for item in response.json()["results"]] == [bill_id]
        assert (
            response.json()["results"][0]["passages"][0]["text"] == "Facture électricité mars 2025"
        )


def test_ai_search_requires_openai_key(tmp_path: Path, monkeypatch: object) -> None:
    with _client(tmp_path) as client:
        project_id, _, _ = _index(client, monkeypatch)
        response = client.post(
            f"/api/v1/projects/{project_id}/search",
            headers={"X-ClairDoc-Key": "secret"},
            json={"query": "facture électricité", "mode": "ai"},
        )
        assert response.status_code == 503
