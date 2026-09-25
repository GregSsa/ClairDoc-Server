from pathlib import Path

from clairdoc_server.models import ProjectCreate
from clairdoc_server.organization import OrganizationService
from clairdoc_server.storage import LocalStorage


def test_plan_builds_category_year_and_unique_names(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "data")
    storage.initialize()
    project = storage.create_project(ProjectCreate(name="Archives"))
    document = {
        "job_id": "d6c7963e-88ad-4458-8971-fc778ed67d39",
        "document_name": "scan.pdf",
        "source_relative_path": "anciens/scan.pdf",
        "metadata": {
            "category": "Factures",
            "document_type": "Facture",
            "date": "2026-03-14",
            "organization": "ACME SARL",
        },
        "chunks": [],
    }
    storage.write_index(project.id, {"documents": [document, document]})

    plan = OrganizationService(storage).build_plan(project.id)

    assert plan.entries[0].suggested_path.startswith("Factures/2026/")
    assert plan.entries[1].suggested_path.endswith("_2.pdf")
