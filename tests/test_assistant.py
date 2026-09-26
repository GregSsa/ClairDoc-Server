from pathlib import Path
from uuid import UUID

import pytest

from clairdoc_server.assistant import AssistantService
from clairdoc_server.config import Settings
from clairdoc_server.models import ConversationMessage, ProjectCreate
from clairdoc_server.rag import RagService
from clairdoc_server.storage import LocalStorage


def make_service(tmp_path: Path) -> tuple[LocalStorage, AssistantService, str, str]:
    root = tmp_path / "documents"
    root.mkdir()
    (root / "note.txt").write_text("Contenu du projet", encoding="utf-8")
    storage = LocalStorage(tmp_path / "data")
    storage.initialize()
    project = storage.create_project(ProjectCreate(name="Archives", source_root=str(root)))
    job = storage.create_job("note.txt", project.id, "note.txt")
    storage.write_index(
        project.id,
        {
            "embedding_model": "test",
            "embedding_dimensions": 2,
            "documents": [
                {
                    "job_id": str(job.id),
                    "document_name": "note.txt",
                    "source_relative_path": "note.txt",
                    "metadata": {"category": "Autres"},
                    "chunks": [],
                }
            ],
        },
    )
    settings = Settings(data_dir=tmp_path / "data", api_key="test-secret")
    service = AssistantService(storage, RagService(storage, settings))
    return storage, service, str(project.id), str(job.id)


def test_read_pdf_ocr_by_name_without_source_access(tmp_path: Path) -> None:
    storage, service, project_id, _ = make_service(tmp_path)
    project_uuid = UUID(project_id)
    job = storage.create_job("e001.pdf", project_uuid, "archive/e001.pdf")
    storage.text_path(job.id).write_text("Attestation fiscale 2025", encoding="utf-8")
    result, action = service._execute_tool(
        project_uuid, "read_project_document", {"document": "e001", "offset": 0, "limit": 11}, False
    )
    assert result["text"] == "Attestation"
    assert result["next_offset"] == 11
    assert action.status == "completed"
    other = storage.create_project(ProjectCreate(name="Autre"))
    with pytest.raises(ValueError):
        service._read_document(other.id, {"document": str(job.id)})


def test_rename_requires_permission_and_preserves_extension(tmp_path: Path) -> None:
    storage, service, project_id, job_id = make_service(tmp_path)
    project_uuid = UUID(project_id)
    args = {"job_id": job_id, "new_name": "memo.txt"}
    result, _ = service._execute_tool(project_uuid, "rename_document", args, False)
    assert result["requires_confirmation"]
    for name in ["../outside.txt", "memo.pdf", "sub/memo.txt"]:
        with pytest.raises(ValueError):
            service._rename_document(project_uuid, {**args, "new_name": name})
    result, _ = service._execute_tool(project_uuid, "rename_document", args, True)
    assert result["ok"]
    assert (tmp_path / "documents" / "memo.txt").is_file()
    assert not (tmp_path / "documents" / "note.txt").exists()
    assert storage.get_job(UUID(job_id)).original_filename == "memo.txt"
    assert storage.read_index(project_uuid)["documents"][0]["document_name"] == "memo.txt"


def test_conversations_and_memory_are_persisted(tmp_path: Path) -> None:
    storage, _, project_id, _ = make_service(tmp_path)
    project_uuid = UUID(project_id)
    conversation = storage.create_conversation(project_uuid, "Contrats")
    conversation.messages.append(ConversationMessage(role="user", content="Bonjour"))
    storage.save_conversation(conversation)

    restored = storage.get_conversation(project_uuid, conversation.id)
    assert restored.title == "Contrats"
    assert restored.messages[0].content == "Bonjour"
    assert storage.read_project_memory(project_uuid).content.startswith("# Archives")


def test_write_tools_require_permission_and_stay_in_project(tmp_path: Path) -> None:
    storage, service, project_id, job_id = make_service(tmp_path)
    project_uuid = UUID(project_id)
    denied, action = service._execute_tool(
        project_uuid,
        "set_document_category",
        {"job_id": job_id, "category": "Important"},
        False,
    )
    assert denied["requires_confirmation"] is True
    assert action.status == "confirmation_required"

    result, action = service._execute_tool(
        project_uuid,
        "set_document_category",
        {"job_id": job_id, "category": "Important"},
        True,
    )
    assert result["ok"] is True
    assert action.status == "completed"
    document = storage.read_index(project_uuid)["documents"][0]
    assert document["metadata"]["category"] == "Important"

    try:
        service._execute_tool(
            project_uuid,
            "copy_document",
            {"job_id": job_id, "destination": "../outside.txt"},
            True,
        )
    except ValueError as error:
        assert "sort du dossier" in str(error)
    else:
        raise AssertionError("La traversée de dossier aurait dû être refusée")


def test_read_tools_are_limited_to_project_text_files(tmp_path: Path) -> None:
    _, service, project_id, _ = make_service(tmp_path)
    project_uuid = UUID(project_id)

    result, action = service._execute_tool(
        project_uuid,
        "read_project_text_file",
        {"relative_path": "note.txt"},
        False,
    )
    assert result["content"] == "Contenu du projet"
    assert action.status == "completed"

    try:
        service._execute_tool(
            project_uuid,
            "read_project_text_file",
            {"relative_path": "../secret.txt"},
            False,
        )
    except ValueError as error:
        assert "sort du dossier" in str(error)
    else:
        raise AssertionError("La lecture hors projet aurait dû être refusée")
