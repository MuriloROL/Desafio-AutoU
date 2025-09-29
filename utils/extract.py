from typing import Optional
from werkzeug.datastructures import FileStorage

# Using pypdf for PDF text extraction
from pypdf import PdfReader


def extract_text_from_pdf(file_storage: FileStorage) -> Optional[str]:
    """Extract text from a PDF uploaded via Flask FileStorage.

    The file stream is read directly without saving to disk.
    """
    try:
        reader = PdfReader(file_storage)
        parts = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            parts.append(txt)
        return "\n\n".join(parts).strip()
    except Exception:
        # Fallback: try reading bytes into PdfReader
        try:
            file_storage.stream.seek(0)
            reader = PdfReader(file_storage.stream)
            parts = []
            for page in reader.pages:
                txt = page.extract_text() or ""
                parts.append(txt)
            return "\n\n".join(parts).strip()
        except Exception:
            return None
