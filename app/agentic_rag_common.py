from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from PIL import Image


MAX_EVIDENCE_FOR_LLM_PROMPT = 50
REFERENCED_GALLERY_ELEM_CLASS = "referenced-images-scroll-gallery"

AnswerGenerator = Callable[[str, list["Evidence"], str], str]
LLMTextGenerator = Callable[[str], str]


@dataclass
class Evidence:
    id: str
    image_id: Any = None
    file_name: str = ""
    caption: str = ""
    search_mode: str = ""
    source_query: str = ""
    source_tool: str = ""
    distance: Any = None
    image: Image.Image | None = None


@dataclass
class SufficiencyDecision:
    status: str
    reason: str
    missing_aspects: list[str] = field(default_factory=list)


class EvidencePool:
    """検索方式ごとの候補を、IDを保ちながら重複排除して集約する。"""

    def __init__(self):
        self._evidence: list[Evidence] = []
        self._seen: set[str] = set()

    def add_many(self, results: Iterable[dict], source_query: str, source_tool: str):
        for result in results or []:
            evidence = self._from_result(result, source_query, source_tool)
            if evidence.id in self._seen:
                continue
            self._seen.add(evidence.id)
            self._evidence.append(evidence)

    def all(self) -> list[Evidence]:
        return list(self._evidence)

    @staticmethod
    def _from_result(result: dict, source_query: str, source_tool: str) -> Evidence:
        stable_id = str(result.get("image_id") or result.get("file_name") or f"{source_tool}:{len(str(result))}")
        image = result.get("image")
        return Evidence(
            id=stable_id,
            image_id=result.get("image_id"),
            file_name=result.get("file_name", stable_id),
            caption=result.get("caption", "") or "",
            search_mode=result.get("search_mode", ""),
            source_query=source_query,
            source_tool=source_tool,
            distance=result.get("distance"),
            image=image if isinstance(image, Image.Image) else None,
        )


def format_documents(selected: list[Evidence]) -> str:
    return "\n\n".join(
        f"{index}. {item.file_name}\n{item.caption or '（キャプションなし）'}"
        for index, item in enumerate(selected, start=1)
    )

