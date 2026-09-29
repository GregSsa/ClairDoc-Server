import hashlib
from asyncio import run
from pathlib import Path
from types import SimpleNamespace
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
    job.content_sha256 = hashlib.sha256(b"Contenu du projet").hexdigest()
    storage.save_job(job)
    storage.text_path(job.id).write_text("Contenu du projet", encoding="utf-8")
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

    job = storage.get_job(UUID(job_id))
    job.content_sha256 = "0" * 64
    storage.save_job(job)
    with pytest.raises(ValueError, match="ne correspond pas"):
        service._project_root(project_uuid)
    job.content_sha256 = hashlib.sha256(b"Contenu du projet").hexdigest()
    storage.save_job(job)

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
        assert "invalide" in str(error)
    else:
        raise AssertionError("La lecture hors projet aurait dû être refusée")


def test_local_file_action_waits_for_confirmation_then_updates_index(tmp_path: Path) -> None:
    storage, service, project_id, job_id = make_service(tmp_path)
    project_uuid = UUID(project_id)
    result, action = service._execute_tool(
        project_uuid,
        "rename_document",
        {"job_id": job_id, "new_name": "memo.txt"},
        False,
        defer_file_actions=True,
    )
    assert result["requires_local_confirmation"] is True
    assert action.status == "pending_local"
    assert (tmp_path / "documents" / "note.txt").is_file()

    conversation = storage.create_conversation(project_uuid, "Renommage")
    conversation.messages.append(
        ConversationMessage(role="assistant", content="Proposition", actions=[action])
    )
    storage.save_conversation(conversation)
    draft = service.draft_for_project(project_uuid)
    assert len(draft.actions) == 1
    assert draft.actions[0].destination_relative_path == "memo.txt"
    with pytest.raises(ValueError, match="extension"):
        service._execute_tool(
            project_uuid,
            "move_document",
            {"job_id": job_id, "destination": "ailleurs.pdf"},
            False,
            defer_file_actions=True,
        )
    _, next_action = service._execute_tool(
        project_uuid,
        "move_document",
        {"job_id": job_id, "destination": "ailleurs/memo.txt"},
        False,
        defer_file_actions=True,
    )
    assert next_action.source_relative_path == "memo.txt"
    conversation.messages.append(
        ConversationMessage(role="assistant", content="Deuxième proposition", actions=[next_action])
    )
    storage.save_conversation(conversation)
    assert [
        item.source_relative_path for item in service.draft_for_project(project_uuid).actions
    ] == [
        "note.txt",
        "memo.txt",
    ]
    listed, _ = service._execute_tool(project_uuid, "list_project_documents", {}, False)
    assert listed["documents"][0]["path"] == "ailleurs/memo.txt"
    found, _ = service._execute_tool(
        project_uuid, "search_project_files", {"query": "ailleurs"}, False
    )
    assert found["files"] == ["ailleurs/memo.txt"]
    read, _ = service._execute_tool(
        project_uuid, "read_project_document", {"document": "memo.txt"}, False
    )
    assert read["name"] == "memo.txt"
    assert read["text"] == "Contenu du projet"
    with pytest.raises(ValueError, match="Retirez d'abord"):
        service.cancel_local_action(project_uuid, conversation.id, action.id)
    with pytest.raises(ValueError, match="dans l'ordre"):
        service.complete_local_action(project_uuid, conversation.id, next_action.id)
    (tmp_path / "documents" / "note.txt").rename(tmp_path / "documents" / "memo.txt")
    completed = service.complete_local_action(project_uuid, conversation.id, action.id)
    assert completed.status == "completed"
    assert storage.get_job(UUID(job_id)).source_relative_path == "memo.txt"
    assert storage.read_index(project_uuid)["documents"][0]["document_name"] == "memo.txt"
    assert (
        storage.get_conversation(project_uuid, conversation.id).messages[0].actions[0].status
        == "completed"
    )
    with pytest.raises(ValueError, match="déjà traitée"):
        service.complete_local_action(project_uuid, conversation.id, action.id)
    assert len(service.draft_for_project(project_uuid).actions) == 1
    (tmp_path / "documents" / "ailleurs").mkdir()
    (tmp_path / "documents" / "memo.txt").rename(tmp_path / "documents" / "ailleurs" / "memo.txt")
    service.complete_local_action(project_uuid, conversation.id, next_action.id)
    assert storage.get_job(UUID(job_id)).source_relative_path == "ailleurs/memo.txt"
    assert service.draft_for_project(project_uuid).actions == []


def test_draft_action_can_be_cancelled_without_touching_file(tmp_path: Path) -> None:
    storage, service, project_id, job_id = make_service(tmp_path)
    project_uuid = UUID(project_id)
    _, action = service._execute_tool(
        project_uuid,
        "delete_document",
        {"job_id": job_id},
        False,
        defer_file_actions=True,
    )
    conversation = storage.create_conversation(project_uuid, "Corbeille")
    conversation.messages.append(
        ConversationMessage(role="assistant", content="Proposition", actions=[action])
    )
    storage.save_conversation(conversation)
    assert len(service.draft_for_project(project_uuid).actions) == 1
    cancelled = service.cancel_local_action(project_uuid, conversation.id, action.id)
    assert cancelled.status == "cancelled"
    assert service.draft_for_project(project_uuid).actions == []
    assert (tmp_path / "documents" / "note.txt").is_file()


def test_multiple_actions_on_same_file_can_be_staged_in_one_reply(tmp_path: Path) -> None:
    storage, service, project_id, job_id = make_service(tmp_path)
    project_uuid = UUID(project_id)
    _, rename = service._execute_tool(
        project_uuid,
        "rename_document",
        {"job_id": job_id, "new_name": "memo.txt"},
        False,
        defer_file_actions=True,
    )
    _, move = service._execute_tool(
        project_uuid,
        "move_document",
        {"job_id": job_id, "destination": "archives/memo.txt"},
        False,
        defer_file_actions=True,
        staged_actions=[rename],
    )
    _, copy = service._execute_tool(
        project_uuid,
        "copy_document",
        {"job_id": job_id, "destination": "copies/memo.txt"},
        False,
        defer_file_actions=True,
        staged_actions=[rename, move],
    )
    assert [item.source_relative_path for item in (rename, move, copy)] == [
        "note.txt",
        "memo.txt",
        "archives/memo.txt",
    ]
    assert (tmp_path / "documents" / "note.txt").is_file()
    with pytest.raises(ValueError, match="déjà utilisée"):
        service._execute_tool(
            project_uuid,
            "copy_document",
            {"job_id": job_id, "destination": "copies/memo.txt"},
            False,
            defer_file_actions=True,
            staged_actions=[rename, move, copy],
        )
    conversation = storage.create_conversation(project_uuid, "Chaîne")
    conversation.messages.append(
        ConversationMessage(role="assistant", content="Trois étapes", actions=[rename, move, copy])
    )
    storage.save_conversation(conversation)
    assert len(service.draft_for_project(project_uuid).actions) == 3
    with pytest.raises(ValueError, match="Retirez d'abord"):
        service.cancel_local_action(project_uuid, conversation.id, rename.id)
    service.cancel_local_action(project_uuid, conversation.id, copy.id)
    service.cancel_local_action(project_uuid, conversation.id, move.id)
    service.cancel_local_action(project_uuid, conversation.id, rename.id)
    assert service.draft_for_project(project_uuid).actions == []


def test_deleted_file_cannot_be_modified_again_before_validation(tmp_path: Path) -> None:
    _, service, project_id, job_id = make_service(tmp_path)
    project_uuid = UUID(project_id)
    _, deletion = service._execute_tool(
        project_uuid,
        "delete_document",
        {"job_id": job_id},
        False,
        defer_file_actions=True,
    )
    with pytest.raises(ValueError, match="corbeille"):
        service._execute_tool(
            project_uuid,
            "rename_document",
            {"job_id": job_id, "new_name": "memo.txt"},
            False,
            defer_file_actions=True,
            staged_actions=[deletion],
        )


def test_local_move_can_finish_before_project_is_indexed(tmp_path: Path) -> None:
    storage, service, project_id, job_id = make_service(tmp_path)
    project_uuid = UUID(project_id)
    storage.index_path(project_uuid).unlink()
    _, action = service._execute_tool(
        project_uuid,
        "move_document",
        {"job_id": job_id, "destination": "archives/note.txt"},
        False,
        defer_file_actions=True,
    )
    conversation = storage.create_conversation(project_uuid, "Déplacement")
    conversation.messages.append(
        ConversationMessage(role="assistant", content="Proposition", actions=[action])
    )
    storage.save_conversation(conversation)
    destination = tmp_path / "documents" / "archives" / "note.txt"
    destination.parent.mkdir()
    (tmp_path / "documents" / "note.txt").rename(destination)
    service.complete_local_action(project_uuid, conversation.id, action.id)
    assert storage.get_job(UUID(job_id)).source_relative_path == "archives/note.txt"


def test_assistant_does_not_load_document_content_before_tool_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, service, project_id, _ = make_service(tmp_path)
    project_uuid = UUID(project_id)
    conversation = storage.create_conversation(project_uuid, "Question")
    calls: list[dict[str, object]] = []

    class Responses:
        async def create(self, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(output=[], output_text="Bonjour.")

    monkeypatch.setattr(service.rag, "_client", lambda: SimpleNamespace(responses=Responses()))

    async def unexpected_search(*_args: object) -> None:
        raise AssertionError("La recherche ne doit pas être lancée automatiquement.")

    monkeypatch.setattr(service, "_select_context", unexpected_search)
    answer = run(service.ask(project_uuid, conversation.id, "Bonjour à toi", None, False))
    assert answer.answer == "Bonjour."
    assert answer.citations == []
    assert calls[0]["input"][-1]["content"] == "Bonjour à toi"
    assert "Contenu du projet" not in calls[0]["instructions"]


def test_assistant_can_stage_then_request_global_validation_in_one_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, service, project_id, job_id = make_service(tmp_path)
    project_uuid = UUID(project_id)
    conversation = storage.create_conversation(project_uuid, "Classement")
    outputs = [
        SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="function_call",
                    name="move_document",
                    call_id="call-1",
                    arguments=f'{{"job_id":"{job_id}","destination":"archives/note.txt"}}',
                )
            ],
            output_text="",
        ),
        SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="function_call",
                    name="request_apply_pending_changes",
                    call_id="call-2",
                    arguments="{}",
                )
            ],
            output_text="",
        ),
        SimpleNamespace(output=[], output_text="Le brouillon est prêt à valider."),
    ]

    class Responses:
        async def create(self, **_kwargs: object) -> SimpleNamespace:
            return outputs.pop(0)

    monkeypatch.setattr(service.rag, "_client", lambda: SimpleNamespace(responses=Responses()))
    answer = run(
        service.ask(
            project_uuid, conversation.id, "Déplace note.txt puis valide le brouillon", None, False
        )
    )
    assert [action.status for action in answer.actions] == ["pending_local", "validation_requested"]
    assert len(service.draft_for_project(project_uuid).actions) == 1
    assert (tmp_path / "documents" / "note.txt").is_file()
