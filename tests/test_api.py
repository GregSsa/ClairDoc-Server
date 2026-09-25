from pathlib import Path
from time import monotonic, sleep

from fastapi.testclient import TestClient

from clairdoc_server.config import Settings
from clairdoc_server.main import create_app


def make_client(tmp_path: Path, api_key: str | None = "test-secret") -> TestClient:
    settings = Settings(
        api_key=api_key,
        openai_api_key="",
        data_dir=tmp_path / "data",
        ocr_command="missing-ocrmypdf-command",
    )
    return TestClient(create_app(settings))


def test_health_reports_configuration(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": "0.1.0",
        "storage_ready": True,
        "ocr_available": False,
        "authentication_configured": True,
        "openai_configured": False,
    }


def test_protected_route_requires_api_key(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        response = client.post("/api/v1/projects", json={"name": "Archives"})

    assert response.status_code == 401


def test_connection_validates_api_key(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        rejected = client.get("/api/v1/connection")
        accepted = client.get(
            "/api/v1/connection",
            headers={"X-ClairDoc-Key": "test-secret"},
        )

    assert rejected.status_code == 401
    assert accepted.status_code == 200
    assert accepted.json() == {"status": "authenticated", "version": "0.1.0"}


def test_missing_server_key_disables_protected_routes(tmp_path: Path) -> None:
    with make_client(tmp_path, api_key=None) as client:
        response = client.post("/api/v1/projects", json={"name": "Archives"})

    assert response.status_code == 503


def test_project_is_persisted_as_json(tmp_path: Path) -> None:
    headers = {"X-ClairDoc-Key": "test-secret"}
    with make_client(tmp_path) as client:
        created = client.post(
            "/api/v1/projects",
            headers=headers,
            json={"name": "Archives familiales", "description": "Documents à classer"},
        )
        project_id = created.json()["id"]
        fetched = client.get(f"/api/v1/projects/{project_id}", headers=headers)

    assert created.status_code == 201
    assert fetched.status_code == 200
    assert fetched.json()["name"] == "Archives familiales"
    assert (tmp_path / "data" / "projects" / f"{project_id}.json").is_file()


def test_projects_can_be_listed_after_creation(tmp_path: Path) -> None:
    headers = {"X-ClairDoc-Key": "test-secret"}
    with make_client(tmp_path) as client:
        client.post("/api/v1/projects", headers=headers, json={"name": "Archives"})
        response = client.get("/api/v1/projects", headers=headers)

    assert response.status_code == 200
    assert [project["name"] for project in response.json()] == ["Archives"]


def test_duplicate_pdf_is_reused_within_project(tmp_path: Path) -> None:
    headers = {"X-ClairDoc-Key": "test-secret"}
    pdf = b"%PDF-1.4\n%%EOF"
    with make_client(tmp_path) as client:
        project = client.post("/api/v1/projects", headers=headers, json={"name": "Archives"})
        project_id = project.json()["id"]
        paused = client.post(f"/api/v1/projects/{project_id}/ocr/pause", headers=headers)
        first = client.post(
            f"/api/v1/ocr/jobs?project_id={project_id}",
            headers=headers,
            files={"file": ("first.pdf", pdf, "application/pdf")},
        )
        duplicate = client.post(
            f"/api/v1/ocr/jobs?project_id={project_id}",
            headers=headers,
            files={"file": ("copy.pdf", pdf, "application/pdf")},
        )

    assert paused.json()["paused"] is True
    assert duplicate.json()["id"] == first.json()["id"]
    assert len(list((tmp_path / "data" / "jobs").iterdir())) == 1


def test_index_requires_openai_key(tmp_path: Path) -> None:
    headers = {"X-ClairDoc-Key": "test-secret"}
    with make_client(tmp_path) as client:
        project = client.post("/api/v1/projects", headers=headers, json={"name": "Archives"})
        response = client.post(
            f"/api/v1/projects/{project.json()['id']}/index",
            headers=headers,
        )

    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_pdf_upload_is_queued_then_fails_without_ocrmypdf(tmp_path: Path) -> None:
    headers = {"X-ClairDoc-Key": "test-secret"}
    with make_client(tmp_path) as client:
        response = client.post(
            "/api/v1/ocr/jobs",
            headers=headers,
            files={"file": ("scan.pdf", b"%PDF-1.4\n%%EOF", "application/pdf")},
        )
        job_id = response.json()["id"]
        deadline = monotonic() + 2
        while monotonic() < deadline:
            job = client.get(f"/api/v1/ocr/jobs/{job_id}", headers=headers)
            if job.json()["status"] == "failed":
                break
            sleep(0.01)

    assert response.status_code == 202
    assert job.status_code == 200
    assert job.json()["status"] == "failed"
    assert "OCRmyPDF" in job.json()["error"]


def test_non_pdf_content_is_rejected(tmp_path: Path) -> None:
    headers = {"X-ClairDoc-Key": "test-secret"}
    with make_client(tmp_path) as client:
        response = client.post(
            "/api/v1/ocr/jobs",
            headers=headers,
            files={"file": ("fake.pdf", b"not a pdf", "application/pdf")},
        )

    assert response.status_code == 415


def test_text_document_is_extracted(tmp_path: Path) -> None:
    headers = {"X-ClairDoc-Key": "test-secret"}
    with make_client(tmp_path) as client:
        project = client.post("/api/v1/projects", headers=headers, json={"name": "Archives"})
        response = client.post(
            f"/api/v1/document/jobs?project_id={project.json()['id']}&source_relative_path=notes/test.txt",
            headers=headers,
            files={"file": ("test.txt", "Bonjour ClairDoc", "text/plain")},
        )
        job_id = response.json()["id"]
        deadline = monotonic() + 2
        while monotonic() < deadline:
            job = client.get(f"/api/v1/ocr/jobs/{job_id}", headers=headers)
            if job.json()["status"] == "completed":
                break
            sleep(0.01)
        extracted = client.get(f"/api/v1/ocr/jobs/{job_id}/text", headers=headers)

    assert response.status_code == 202
    assert job.json()["source_relative_path"] == "notes/test.txt"
    assert extracted.text == "Bonjour ClairDoc"


def test_index_estimate_is_available_without_openai_key(tmp_path: Path) -> None:
    headers = {"X-ClairDoc-Key": "test-secret"}
    text = "Bonjour ClairDoc"
    with make_client(tmp_path) as client:
        project = client.post("/api/v1/projects", headers=headers, json={"name": "Archives"})
        project_id = project.json()["id"]
        uploaded = client.post(
            f"/api/v1/document/jobs?project_id={project_id}",
            headers=headers,
            files={"file": ("test.txt", text, "text/plain")},
        )
        job_id = uploaded.json()["id"]
        deadline = monotonic() + 2
        while monotonic() < deadline:
            job = client.get(f"/api/v1/ocr/jobs/{job_id}", headers=headers)
            if job.json()["status"] == "completed":
                break
            sleep(0.01)
        estimate = client.get(
            f"/api/v1/projects/{project_id}/index/estimate",
            headers=headers,
        )

    assert estimate.status_code == 200
    assert estimate.json()["documents_total"] == 1
    assert estimate.json()["documents_to_embed"] == 1
    assert estimate.json()["estimated_tokens"] == 4
    assert estimate.json()["estimated_cost_usd"] == 0.0


def test_background_index_task_persists_failure(tmp_path: Path) -> None:
    headers = {"X-ClairDoc-Key": "test-secret"}
    with make_client(tmp_path) as client:
        project = client.post("/api/v1/projects", headers=headers, json={"name": "Archives"})
        project_id = project.json()["id"]
        started = client.post(
            f"/api/v1/projects/{project_id}/index/jobs",
            headers=headers,
        )
        task_id = started.json()["id"]
        deadline = monotonic() + 2
        while monotonic() < deadline:
            task = client.get(f"/api/v1/index/jobs/{task_id}", headers=headers)
            if task.json()["status"] == "failed":
                break
            sleep(0.01)

    assert started.status_code == 202
    assert task.json()["status"] == "failed"
    assert "OPENAI_API_KEY" in task.json()["error"]
    assert (tmp_path / "data" / "index-tasks" / f"{task_id}.json").is_file()


def test_metadata_backup_contains_schema_and_projects(tmp_path: Path) -> None:
    import zipfile

    headers = {"X-ClairDoc-Key": "test-secret"}
    with make_client(tmp_path) as client:
        project = client.post("/api/v1/projects", headers=headers, json={"name": "Archives"})
        response = client.post("/api/v1/maintenance/backups", headers=headers)

    backup = Path(response.json()["path"])
    with zipfile.ZipFile(backup) as archive:
        names = set(archive.namelist())

    assert response.status_code == 200
    assert backup.is_file()
    assert "schema.json" in names
    assert f"projects/{project.json()['id']}.json" in names


def test_document_library_lists_completed_documents_before_indexing(tmp_path: Path) -> None:
    headers = {"X-ClairDoc-Key": "test-secret"}
    with make_client(tmp_path) as client:
        project = client.post("/api/v1/projects", headers=headers, json={"name": "Archives"})
        project_id = project.json()["id"]
        uploaded = client.post(
            f"/api/v1/document/jobs?project_id={project_id}",
            headers=headers,
            files={"file": ("facture.txt", "Facture de test", "text/plain")},
        )
        job_id = uploaded.json()["id"]
        deadline = monotonic() + 2
        while monotonic() < deadline:
            job = client.get(f"/api/v1/ocr/jobs/{job_id}", headers=headers)
            if job.json()["status"] == "completed":
                break
            sleep(0.01)
        library = client.get(
            f"/api/v1/projects/{project_id}/documents",
            headers=headers,
        )

    assert library.status_code == 200
    assert library.json()["documents"][0]["name"] == "facture.txt"
    assert library.json()["documents"][0]["category"] == "À indexer"
    assert library.json()["relationships"] == []
