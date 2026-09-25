import re
import unicodedata
from pathlib import Path, PurePosixPath
from uuid import UUID

from .models import OrganizationEntry, OrganizationPlan
from .storage import LocalStorage, RecordNotFoundError


def _safe_component(value: str, fallback: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    normalized = re.sub(r"[^\w.-]+", "_", normalized, flags=re.UNICODE).strip("._-")
    return normalized[:80] or fallback


class OrganizationService:
    def __init__(self, storage: LocalStorage) -> None:
        self.storage = storage

    def build_plan(self, project_id: UUID) -> OrganizationPlan:
        self.storage.get_project(project_id)
        index = self.storage.read_index(project_id)
        used_paths: set[str] = set()
        entries: list[OrganizationEntry] = []
        for document in index.get("documents", []):
            metadata = document.get("metadata", {})
            category = str(metadata.get("category") or "Autres")
            document_date = metadata.get("date")
            year = str(document_date)[:4] if document_date else "Date_inconnue"
            organization = metadata.get("organization")
            original = str(document["document_name"])
            extension = Path(original).suffix.lower()
            stem_parts = [
                str(document_date or "sans_date"),
                str(metadata.get("document_type") or category),
            ]
            if organization:
                stem_parts.append(str(organization))
            stem_parts.append(Path(original).stem)
            stem = _safe_component("_".join(stem_parts), "document")
            folder = PurePosixPath(
                _safe_component(category, "Autres"),
                _safe_component(year, "Date_inconnue"),
            )
            candidate = str(folder / f"{stem}{extension}")
            suffix = 2
            while candidate.casefold() in used_paths:
                candidate = str(folder / f"{stem}_{suffix}{extension}")
                suffix += 1
            used_paths.add(candidate.casefold())
            entries.append(
                OrganizationEntry(
                    job_id=UUID(str(document["job_id"])),
                    original_filename=original,
                    source_relative_path=str(document.get("source_relative_path") or original),
                    suggested_path=candidate,
                    category=category,
                    document_date=document_date,
                    organization=organization,
                    reason=f"Classé dans {category} à partir du contenu détecté.",
                )
            )
        if not entries:
            raise RecordNotFoundError("index vide")
        plan = OrganizationPlan(project_id=project_id, entries=entries)
        self.storage.write_plan(project_id, plan.model_dump(mode="json"))
        return plan
