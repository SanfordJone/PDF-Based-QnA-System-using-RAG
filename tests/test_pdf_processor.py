import pytest

from pdf_processor import PDFError, PDFProcessor


def test_chunks_overlap():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = PDFProcessor().chunk_text(text, chunk_size=100, chunk_overlap=30)
    assert len(chunks) > 1
    # the start of each chunk should repeat words from the end of the previous one
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.split()[0] in prev.split()


def test_chunks_cover_all_text_and_respect_size():
    text = " ".join(f"w{i}" for i in range(400))
    chunks = PDFProcessor().chunk_text(text, chunk_size=120, chunk_overlap=20)
    assert all(len(c) <= 120 for c in chunks)
    assert chunks[0].startswith("w0")
    assert chunks[-1].endswith("w399")


def test_chunking_text_without_spaces_terminates():
    chunks = PDFProcessor().chunk_text("x" * 2500, chunk_size=1000, chunk_overlap=200)
    assert len(chunks) == 3


def test_empty_text_gives_no_chunks():
    assert PDFProcessor().chunk_text("") == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_chunk_settings_rejected(size, overlap):
    with pytest.raises(ValueError):
        PDFProcessor().chunk_text("some text", chunk_size=size, chunk_overlap=overlap)


def test_extract_pages_keeps_page_numbers(sample_pdf):
    pages = PDFProcessor().extract_pages(sample_pdf)
    assert [p for p, _ in pages] == [1, 2]
    assert "Refunds" in pages[0][1]
    assert "Parramatta" in pages[1][1]


def test_rejects_non_pdf():
    with pytest.raises(PDFError):
        PDFProcessor().extract_pages(b"hello, not a pdf")


def test_rejects_empty_file():
    with pytest.raises(PDFError):
        PDFProcessor().extract_pages(b"")


def test_rejects_oversized_file(sample_pdf):
    with pytest.raises(PDFError):
        PDFProcessor(max_file_bytes=100).extract_pages(sample_pdf)


def test_rejects_too_many_pages(sample_pdf):
    with pytest.raises(PDFError):
        PDFProcessor(max_pages=1).extract_pages(sample_pdf)
