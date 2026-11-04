from ai_learning_rag.pdf_chunking import PdfChunkingConfig, chunk_pdf_document
from ai_learning_rag.pdf_chunking_preview import build_markdown_preview
from ai_learning_rag.pdf_schema import (
    BlockType,
    BoundingBox,
    ExtractionMethod,
    PageKind,
    ParsedBlock,
    ParsedDocument,
    ParsedPage,
    ParseStatus,
)


def test_markdown_preview_exposes_content_and_source_offsets() -> None:
    block = ParsedBlock(
        id="body-1",
        page_number=1,
        block_type=BlockType.PARAGRAPH,
        text="Traceable chunk content.",
        bbox=BoundingBox(10, 20, 200, 40),
        reading_order=0,
        extraction_method=ExtractionMethod.NATIVE,
    )
    document = ParsedDocument(
        document_id="preview-doc",
        source="file:///preview.pdf",
        file_sha256="a" * 64,
        parser_name="preview-parser",
        parser_version="1.0",
        pages=(
            ParsedPage(
                page_number=1,
                width=595,
                height=842,
                kind=PageKind.NATIVE_TEXT,
                status=ParseStatus.PARSED,
                blocks=(block,),
                text_character_count=len(block.text),
            ),
        ),
    )
    result = chunk_pdf_document(document, PdfChunkingConfig())

    preview = build_markdown_preview(document, result)

    assert "### 0. body_text" in preview
    assert "p1:body-1[0:24]" in preview
    assert "Traceable chunk content." in preview
    assert result.chunks[0].id in preview
