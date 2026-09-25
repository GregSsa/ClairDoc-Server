from pathlib import Path
from uuid import UUID

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
