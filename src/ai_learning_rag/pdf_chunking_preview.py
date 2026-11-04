"""Human-readable rendering for structure-aware PDF chunking results."""

from pathlib import Path

from ai_learning_rag.pdf_chunking import PdfChunkingResult
from ai_learning_rag.pdf_schema import ParsedDocument


def build_markdown_preview(
    document: ParsedDocument,
    result: PdfChunkingResult,
    *,
    preview_characters: int = 1200,
) -> str:
    """Render a human-readable review of chunk content and traceability."""

    lines = [
        f"# PDF chunking preview: {Path(document.source).name}",
        "",
        f"- Source: `{document.source}`",
        f"- Parser: `{document.parser_name} {document.parser_version}`",
        f"- Pages: {document.page_count}",
        f"- Status: `{result.status.value}`",
        f"- Chunks: {len(result.chunks)}",
        f"- Diagnostics: {len(result.diagnostics)}",
        "",
        "## Diagnostics",
        "",
    ]
    if result.diagnostics:
        lines.extend(
            (
                "| Severity | Code | Pages | Source blocks | Message |",
                "| --- | --- | --- | --- | --- |",
            )
        )
        for diagnostic in result.diagnostics:
            lines.append(
                "| "
                + " | ".join(
                    (
                        diagnostic.severity.value,
                        diagnostic.code.value,
                        ", ".join(map(str, diagnostic.page_numbers)) or "-",
                        ", ".join(diagnostic.source_block_ids) or "-",
                        _escape_table_cell(diagnostic.message),
                    )
                )
                + " |"
            )
    else:
        lines.append("No diagnostics.")

    lines.extend(("", "## Chunks", ""))
    for chunk in result.chunks:
        section_path = " > ".join(chunk.section_path) or "(root)"
        source_summary = ", ".join(
            f"p{region.page_number}:{region.block_id}"
            f"[{region.source_text_start}:{region.source_text_end}]"
            for region in chunk.source_regions
        )
        content = chunk.content
        if len(content) > preview_characters:
            content = content[:preview_characters] + "\n… [preview truncated; JSON is complete]"
        lines.extend(
            (
                f"### {chunk.chunk_index}. {chunk.content_type.value}",
                "",
                f"- Chunk ID: `{chunk.id}`",
                f"- Pages: {chunk.page_start}–{chunk.page_end}",
                f"- Section: {section_path}",
                f"- Sources: `{source_summary}`",
                f"- Warnings: {len(chunk.warnings)}",
                "",
                "```text",
                content.replace("```", "` ` `"),
                "```",
                "",
            )
        )
    return "\n".join(lines)


def _escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
