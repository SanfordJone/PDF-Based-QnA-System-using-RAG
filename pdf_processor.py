import io

from pypdf import PdfReader

# Limits on uploaded files, so a very large or hostile PDF cannot exhaust memory
MAX_FILE_BYTES = 20 * 1024 * 1024  # 20 MB
MAX_PAGES = 300


class PDFError(ValueError):
    """Raised when an uploaded file cannot be processed safely."""


class PDFProcessor:
    """
    Class to handle PDF operations including text extraction and chunking
    """

    def __init__(self, max_file_bytes=MAX_FILE_BYTES, max_pages=MAX_PAGES):
        self.max_file_bytes = max_file_bytes
        self.max_pages = max_pages

    def extract_pages(self, pdf_bytes):
        """
        Extract text from each page of a PDF

        Args:
            pdf_bytes (bytes): Raw PDF content

        Returns:
            list: One (page_number, text) tuple per page that has text
        """
        if not pdf_bytes:
            raise PDFError("The file is empty.")
        if len(pdf_bytes) > self.max_file_bytes:
            raise PDFError(f"File is larger than {self.max_file_bytes // (1024 * 1024)} MB.")
        if not pdf_bytes.startswith(b"%PDF"):
            raise PDFError("This does not look like a PDF file.")

        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
        except Exception as e:
            raise PDFError(f"Could not read the PDF: {e}") from e

        if reader.is_encrypted:
            raise PDFError("Password protected PDFs are not supported.")
        if len(reader.pages) > self.max_pages:
            raise PDFError(f"PDF has more than {self.max_pages} pages.")

        pages = []
        for number, page in enumerate(reader.pages, start=1):
            # extract_text() can return None or "" for image-only pages
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((number, text))
        return pages

    def chunk_text(self, text, chunk_size=1000, chunk_overlap=200):
        """
        Split text into overlapping chunks for processing

        Args:
            text (str): Text to chunk
            chunk_size (int): Maximum size of each chunk
            chunk_overlap (int): Overlap between chunks

        Returns:
            list: List of text chunks
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be at least 0 and smaller than chunk_size")
        if not text:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)

            # If not at the end of text, break at the last space to avoid splitting words
            if end < text_length:
                last_space = text.rfind(" ", start, end)
                if last_space > start:
                    end = last_space

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            # Step back by the overlap so neighbouring chunks share context,
            # but always move forward so the loop cannot get stuck
            start = max(end - chunk_overlap, start + 1)

        return chunks
