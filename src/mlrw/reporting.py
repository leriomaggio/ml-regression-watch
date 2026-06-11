"""Shared Markdown rendering helpers."""

from __future__ import annotations

from collections.abc import Sequence


def markdown_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a GitHub-flavoured Markdown table.

    Columns are not padded to equal width; GitHub renders the table regardless and
    keeping the source compact avoids noisy diffs when values change.
    """
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)
