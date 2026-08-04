"""Duplicate detection helpers."""

from __future__ import annotations


class DuplicateService:
    def group_by_hash(self, files: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
        groups: dict[str, list[dict[str, object]]] = {}
        for item in files:
            digest = str(item.get("content_hash") or "")
            if digest:
                groups.setdefault(digest, []).append(item)
        return {digest: group for digest, group in groups.items() if len(group) > 1}
