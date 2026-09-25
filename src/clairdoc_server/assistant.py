import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from .models import AskResponse, AssistantAction, Citation, ConversationMessage
from .rag import cosine_similarity, keyword_similarity
from .storage import LocalStorage

ASSISTANT_INSTRUCTIONS = """Tu es l'assistant spécialisé d'un projet ClairDoc.
Utilise la mémoire du projet, l'historique de la conversation et les extraits fournis.
N'invente jamais le contenu d'un document. Cite les sources avec [1], [2], etc.
Tu peux utiliser les outils pour consulter ou organiser le projet.
N'exécute une action d'écriture que si la demande de l'utilisateur est explicite et si l'outil
l'autorise. Explique brièvement chaque action réellement effectuée. Réponds en français."""


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "list_project_documents",
        "description": "Liste les documents connus avec leur identifiant, chemin et catégorie.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "search_project_files",
        "description": "Recherche des fichiers par nom dans le dossier source du projet.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "read_project_text_file",
        "description": "Lit un fichier texte situé dans le dossier source du projet.",
        "parameters": {
            "type": "object",
            "properties": {"relative_path": {"type": "string"}},
            "required": ["relative_path"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "set_document_category",
        "description": "Attribue une catégorie à un document indexé.",
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "category": {"type": "string"},
            },
            "required": ["job_id", "category"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "add_document_relationship",
        "description": "Ajoute un lien explicite entre deux documents indexés.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_job_id": {"type": "string"},
                "target_job_id": {"type": "string"},
                "kind": {"type": "string"},
                "label": {"type": "string"},
            },
            "required": ["source_job_id", "target_job_id", "kind", "label"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "update_project_memory",
        "description": "Met à jour le README mémoire du projet avec des informations durables.",
        "parameters": {
            "type": "object",
            "properties": {"content": {"type": "string"}},
            "required": ["content"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "copy_document",
        "description": "Copie un document dans un autre chemin du dossier source du projet.",
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "destination": {"type": "string"},
            },
            "required": ["job_id", "destination"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "move_document",
        "description": (
            "Déplace un document dans le dossier source et actualise son chemin ClairDoc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "destination": {"type": "string"},
            },
            "required": ["job_id", "destination"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "delete_document",
        "description": "Place un document dans la corbeille locale .clairdoc/trash du projet.",
        "parameters": {
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


class AssistantService:
    def __init__(self, storage: LocalStorage, rag: Any) -> None:
        self.storage = storage
        self.rag = rag

    async def ask(
        self,
        project_id: UUID,
        conversation_id: UUID,
        question: str,
        top_k: int | None,
        allow_write_actions: bool,
    ) -> AskResponse:
        project = self.storage.get_project(project_id)
        conversation = self.storage.get_conversation(project_id, conversation_id)
        selected = await self._select_context(project_id, question, top_k)
        citations = self._citations(selected)
        context = self._format_context(selected)
        memory = self.storage.read_project_memory(project_id).content
        transcript = [
            {"role": message.role, "content": message.content}
            for message in conversation.messages[-20:]
        ]
        transcript.append(
            {
                "role": "user",
                "content": f"{question}\n\nExtraits documentaires :\n{context}",
            }
        )
        instructions = (
            f"{ASSISTANT_INSTRUCTIONS}\n\nProjet : {project.name}\n"
            f"Mémoire projet :\n{memory[:20000]}\n\n"
            f"Actions d'écriture autorisées pour ce tour : {allow_write_actions}."
        )
        client = self.rag._client()
        model = self.storage.get_llm_model(self.rag.settings.llm_model)
        response = await client.responses.create(
            model=model,
            instructions=instructions,
            input=transcript,
            tools=TOOLS,
            store=False,
        )
        running_input: list[Any] = list(transcript)
        actions: list[AssistantAction] = []
        for _ in range(6):
            calls = [item for item in response.output if item.type == "function_call"]
            if not calls:
                break
            running_input.extend(response.output)
            for call in calls:
                try:
                    arguments = json.loads(call.arguments)
                    result, action = self._execute_tool(
                        project_id, call.name, arguments, allow_write_actions
                    )
                except (ValueError, OSError, KeyError) as exc:
                    result = {"ok": False, "error": str(exc)}
                    action = AssistantAction(tool=call.name, status="failed", summary=str(exc))
                actions.append(action)
                running_input.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(result, ensure_ascii=False),
                    }
                )
            response = await client.responses.create(
                model=model,
                instructions=instructions,
                input=running_input,
                tools=TOOLS,
                store=False,
            )

        answer = response.output_text or "Je n'ai pas pu produire de réponse."
        if not conversation.messages:
            conversation.title = question.strip()[:80]
        conversation.messages.extend(
            [
                ConversationMessage(role="user", content=question),
                ConversationMessage(
                    role="assistant", content=answer, citations=citations, actions=actions
                ),
            ]
        )
        self.storage.save_conversation(conversation)
        return AskResponse(answer=answer, citations=citations, model=model, actions=actions)

    async def _select_context(
        self, project_id: UUID, question: str, top_k: int | None
    ) -> list[tuple[float, dict[str, Any], dict[str, Any]]]:
        try:
            index = self.storage.read_index(project_id)
        except FileNotFoundError:
            return []
        query = await self.rag._client().embeddings.create(
            model=str(index["embedding_model"]),
            input=question,
            dimensions=int(index["embedding_dimensions"]),
        )
        query_vector = query.data[0].embedding
        ranked: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for document in index.get("documents", []):
            metadata_text = " ".join(str(value) for value in document.get("metadata", {}).values())
            for chunk in document.get("chunks", []):
                vector = (cosine_similarity(query_vector, chunk["embedding"]) + 1) / 2
                keyword = keyword_similarity(question, f"{chunk['text']} {metadata_text}")
                ranked.append((0.75 * vector + 0.25 * keyword, document, chunk))
        selected = sorted(ranked, key=lambda item: item[0], reverse=True)[
            : top_k or self.rag.settings.rag_top_k
        ]
        return selected

    @staticmethod
    def _format_context(selected: list[tuple[float, dict[str, Any], dict[str, Any]]]) -> str:
        parts = []
        for number, (_, document, chunk) in enumerate(selected, 1):
            page = f", page {chunk['page_number']}" if chunk.get("page_number") else ""
            parts.append(
                f"[{number}] {document['document_name']}{page} "
                f"(job_id={document['job_id']})\n{chunk['text']}"
            )
        return "\n\n".join(parts) or "Aucun extrait indexé pour cette question."

    @staticmethod
    def _citations(selected: list[tuple[float, dict[str, Any], dict[str, Any]]]) -> list[Citation]:
        return [
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

    def _execute_tool(
        self, project_id: UUID, name: str, arguments: dict[str, Any], allow_write: bool
    ) -> tuple[dict[str, Any], AssistantAction]:
        if name == "list_project_documents":
            result = self._list_documents(project_id)
            return result, AssistantAction(
                tool=name,
                status="completed",
                summary=f"{len(result['documents'])} document(s) listé(s).",
            )
        if name == "search_project_files":
            result = self._search_files(project_id, str(arguments["query"]))
            return result, AssistantAction(
                tool=name,
                status="completed",
                summary=f"{len(result['files'])} fichier(s) trouvé(s).",
            )
        if name == "read_project_text_file":
            result = self._read_text_file(project_id, str(arguments["relative_path"]))
            return result, AssistantAction(
                tool=name, status="completed", summary="Fichier texte consulté."
            )
        if not allow_write:
            summary = "Action non exécutée : autorisation requise dans l'interface."
            return {"ok": False, "requires_confirmation": True}, AssistantAction(
                tool=name, status="confirmation_required", summary=summary
            )
        handlers = {
            "set_document_category": self._set_category,
            "add_document_relationship": self._add_relationship,
            "update_project_memory": self._update_memory,
            "copy_document": self._copy_document,
            "move_document": self._move_document,
            "delete_document": self._delete_document,
        }
        if name not in handlers:
            raise ValueError(f"Outil inconnu : {name}")
        result = handlers[name](project_id, arguments)
        summary = str(result.get("summary") or "Action exécutée.")
        self.storage.log_action(
            project_id, {"tool": name, "arguments": arguments, "result": result}
        )
        return result, AssistantAction(tool=name, status="completed", summary=summary)

    def _index_document(
        self, project_id: UUID, job_id: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        index = self.storage.read_index(project_id)
        document = next(
            (item for item in index.get("documents", []) if str(item.get("job_id")) == job_id),
            None,
        )
        if not document:
            raise ValueError("Document indexé introuvable.")
        return index, document

    def _set_category(self, project_id: UUID, arguments: dict[str, Any]) -> dict[str, Any]:
        category = str(arguments["category"]).strip()[:120]
        if not category:
            raise ValueError("La catégorie est vide.")
        index, document = self._index_document(project_id, str(arguments["job_id"]))
        document.setdefault("metadata", {})["category"] = category
        self.storage.write_index(project_id, index)
        self.refresh_memory(project_id)
        return {"ok": True, "summary": f"Catégorie définie sur « {category} »."}

    def _add_relationship(self, project_id: UUID, arguments: dict[str, Any]) -> dict[str, Any]:
        index, _ = self._index_document(project_id, str(arguments["source_job_id"]))
        self._index_document(project_id, str(arguments["target_job_id"]))
        relationship = {
            "source_job_id": str(arguments["source_job_id"]),
            "target_job_id": str(arguments["target_job_id"]),
            "kind": str(arguments["kind"]).strip()[:80],
            "label": str(arguments["label"]).strip()[:200],
        }
        relationships = index.setdefault("relationships", [])
        if relationship not in relationships:
            relationships.append(relationship)
            self.storage.write_index(project_id, index)
        return {"ok": True, "summary": "Lien ajouté entre les deux documents."}

    def _update_memory(self, project_id: UUID, arguments: dict[str, Any]) -> dict[str, Any]:
        content = str(arguments["content"]).strip()
        if not content or len(content) > 20000:
            raise ValueError("La mémoire doit contenir entre 1 et 20 000 caractères.")
        self.storage.write_project_memory(project_id, content)
        return {"ok": True, "summary": "Mémoire du projet mise à jour."}

    def _project_root(self, project_id: UUID) -> Path:
        project = self.storage.get_project(project_id)
        if not project.source_root:
            raise ValueError("Le projet n'a pas de dossier source.")
        root = Path(project.source_root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError("Le dossier source du projet est inaccessible sur le serveur.")
        return root

    @staticmethod
    def _safe_destination(root: Path, relative: str) -> Path:
        candidate = (root / relative).resolve()
        if candidate == root or root not in candidate.parents:
            raise ValueError("Le chemin demandé sort du dossier du projet.")
        return candidate

    def _document_source(self, project_id: UUID, job_id: str) -> tuple[Any, Path, Path]:
        root = self._project_root(project_id)
        job = self.storage.get_job(UUID(job_id))
        if job.project_id != project_id:
            raise ValueError("Ce document n'appartient pas au projet.")
        source = self._safe_destination(root, job.source_relative_path or job.original_filename)
        if not source.is_file():
            raise ValueError("Le fichier source n'existe plus dans le dossier du projet.")
        return job, root, source

    def _copy_document(self, project_id: UUID, arguments: dict[str, Any]) -> dict[str, Any]:
        _, root, source = self._document_source(project_id, str(arguments["job_id"]))
        destination = self._safe_destination(root, str(arguments["destination"]))
        if destination.exists():
            raise ValueError("La destination existe déjà.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return {"ok": True, "summary": f"Document copié vers {destination.relative_to(root)}."}

    def _move_document(self, project_id: UUID, arguments: dict[str, Any]) -> dict[str, Any]:
        job, root, source = self._document_source(project_id, str(arguments["job_id"]))
        destination = self._safe_destination(root, str(arguments["destination"]))
        if destination.exists():
            raise ValueError("La destination existe déjà.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(source, destination)
        job.source_relative_path = destination.relative_to(root).as_posix()
        job.original_filename = destination.name
        self.storage.save_job(job)
        index, document = self._index_document(project_id, str(job.id))
        document["source_relative_path"] = job.source_relative_path
        document["document_name"] = job.original_filename
        self.storage.write_index(project_id, index)
        return {"ok": True, "summary": f"Document déplacé vers {job.source_relative_path}."}

    def _delete_document(self, project_id: UUID, arguments: dict[str, Any]) -> dict[str, Any]:
        job, root, source = self._document_source(project_id, str(arguments["job_id"]))
        trash = root / ".clairdoc" / "trash"
        trash.mkdir(parents=True, exist_ok=True)
        destination = trash / f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S-%f')}-{source.name}"
        shutil.move(source, destination)
        index = self.storage.read_index(project_id)
        index["documents"] = [
            item for item in index.get("documents", []) if str(item.get("job_id")) != str(job.id)
        ]
        index["relationships"] = [
            item
            for item in index.get("relationships", [])
            if str(job.id) not in {str(item.get("source_job_id")), str(item.get("target_job_id"))}
        ]
        self.storage.write_index(project_id, index)
        self.storage.delete_job(job.id)
        self.refresh_memory(project_id)
        return {"ok": True, "summary": f"Document placé dans {destination.relative_to(root)}."}

    def _search_files(self, project_id: UUID, query: str) -> dict[str, Any]:
        root = self._project_root(project_id)
        needle = query.casefold().strip()
        files = []
        for path in root.rglob("*"):
            if ".clairdoc" in path.parts or not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if not needle or needle in relative.casefold():
                files.append(relative)
            if len(files) >= 50:
                break
        return {"ok": True, "files": files}

    def _list_documents(self, project_id: UUID) -> dict[str, Any]:
        try:
            index = self.storage.read_index(project_id)
        except FileNotFoundError:
            index = {"documents": []}
        documents = [
            {
                "job_id": str(item.get("job_id")),
                "name": str(item.get("document_name")),
                "path": str(item.get("source_relative_path")),
                "category": str(item.get("metadata", {}).get("category") or "Autres"),
            }
            for item in index.get("documents", [])
        ]
        return {"ok": True, "documents": documents[:500]}

    def _read_text_file(self, project_id: UUID, relative_path: str) -> dict[str, Any]:
        root = self._project_root(project_id)
        path = self._safe_destination(root, relative_path)
        allowed = {".txt", ".md", ".csv", ".tsv", ".log", ".json", ".xml", ".yaml", ".yml"}
        if not path.is_file() or path.suffix.lower() not in allowed:
            raise ValueError("Ce fichier texte n'est pas lisible par l'assistant.")
        content = path.read_text(encoding="utf-8", errors="replace")[:20000]
        return {"ok": True, "path": path.relative_to(root).as_posix(), "content": content}

    def refresh_memory(self, project_id: UUID) -> None:
        project = self.storage.get_project(project_id)
        try:
            index = self.storage.read_index(project_id)
        except FileNotFoundError:
            index = {"documents": []}
        documents = list(index.get("documents", []))
        categories: dict[str, int] = {}
        lines = []
        for document in documents:
            metadata = document.get("metadata", {})
            category = str(metadata.get("category") or "Autres")
            categories[category] = categories.get(category, 0) + 1
            details = [category]
            if metadata.get("date"):
                details.append(str(metadata["date"]))
            if metadata.get("organization"):
                details.append(str(metadata["organization"]))
            lines.append(f"- {document.get('document_name')} — {' · '.join(details)}")
        content = (
            f"# {project.name}\n\n## Rôle du projet\n"
            f"{project.description or 'Projet documentaire ClairDoc.'}\n\n"
            f"## Dossier source\n{project.source_root or 'Non défini'}\n\n"
            f"## Vue d'ensemble\n{len(documents)} document(s) indexé(s). "
            + ", ".join(f"{name}: {count}" for name, count in sorted(categories.items()))
            + "\n\n## Documents\n"
            + ("\n".join(lines[:200]) or "Aucun document indexé.")
            + "\n"
        )
        self.storage.write_project_memory(project_id, content)
