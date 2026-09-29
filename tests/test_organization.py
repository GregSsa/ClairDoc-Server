from pathlib import Path

from clairdoc_server.models import JobStatus, ProjectCreate
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
    assert plan.entries[0].suggested_path.endswith("/scan.pdf")
    assert plan.entries[1].suggested_path.endswith("_2.pdf")
    renamed = OrganizationService(storage).build_plan(
        project.id,
        rename_files=True,
        names={document["job_id"]: "Facture_ACME_mars_2026"},
    )
    assert renamed.entries[0].suggested_path.endswith("/Facture_ACME_mars_2026.pdf")


def test_plan_respects_depth_and_normalizes_filename_dates(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "data")
    storage.initialize()
    project = storage.create_project(ProjectCreate(name="Archives"))
    storage.write_index(project.id, {"documents": [{
        "job_id": "d6c7963e-88ad-4458-8971-fc778ed67d39",
        "document_name": "Facture_2026-03-14.pdf",
        "source_relative_path": "old/Facture_2026-03-14.pdf",
        "metadata": {"category": "Factures", "date": "2026-03-14"},
        "chunks": [],
    }]})
    service = OrganizationService(storage)

    one = service.build_plan(project.id, rename_files=True, max_depth=1)
    assert one.entries[0].suggested_path == "Factures/Facture_14-03-2026.pdf"
    flat = service.build_plan(project.id, rename_files=True, organize=False)
    assert flat.entries[0].suggested_path == "Facture_14-03-2026.pdf"
    preserved = service.build_plan(project.id, rename_files=False, max_depth=None)
    assert preserved.entries[0].suggested_path.endswith("/Facture_2026-03-14.pdf")


def test_plan_limits_direct_child_folders(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "data")
    storage.initialize()
    project = storage.create_project(ProjectCreate(name="Archives"))
    categories = ["Factures", "Contrats", "Courriers"]
    documents = [{
        "job_id": f"d6c7963e-88ad-4458-8971-fc778ed67d3{index}",
        "document_name": f"document{index}.pdf",
        "metadata": {"category": category, "date": f"202{index}-03-14"},
        "chunks": [],
    } for index, category in enumerate(categories)]
    storage.write_index(project.id, {"documents": documents})

    plan = OrganizationService(storage).build_plan(project.id, max_children=2)

    root_children = {entry.suggested_path.split("/")[0] for entry in plan.entries}
    assert len(root_children) <= 2
    for category in root_children:
        years = {entry.suggested_path.split("/")[1] for entry in plan.entries
                 if entry.suggested_path.startswith(f"{category}/")}
        assert len(years) <= 2


def test_plan_keeps_failed_imports_visible_for_review(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "data")
    storage.initialize()
    project = storage.create_project(ProjectCreate(name="Archives"))
    failed = storage.create_job("scan001.pdf", project.id, "old/scan001.pdf")
    failed.status = JobStatus.FAILED
    storage.save_job(failed)
    storage.write_index(project.id, {"documents": []})

    plan = OrganizationService(storage).build_plan(project.id, rename_files=True)

    assert len(plan.entries) == 1
    assert plan.entries[0].original_filename == "scan001.pdf"
    assert "À vérifier" in plan.entries[0].reason
