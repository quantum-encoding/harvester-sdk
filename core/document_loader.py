import os

def load_text_file(path):
    with open(path, encoding="utf-8") as f:
        return f.read()

def load_md_file(path):
    return load_text_file(path)

def load_pdf_file(path):
    # Requires: pip install PyPDF2
    from PyPDF2 import PdfReader
    text = []
    with open(path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text.append(page.extract_text() or "")
    return '\n'.join(text)

def load_docx_file(path):
    # Requires: pip install python-docx
    from docx import Document
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def load_html_file(path):
    # Strips tags, keeps readable text
    from html.parser import HTMLParser
    class TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.text = []
        def handle_data(self, data):
            self.text.append(data)
    parser = TextExtractor()
    with open(path, encoding="utf-8") as f:
        parser.feed(f.read())
    return "".join(parser.text)

DOCUMENT_LOADERS = {
    ".py": load_text_file,
    ".js": load_text_file,
    ".ts": load_text_file,
    ".tsx": load_text_file,
    ".java": load_text_file,
    ".go": load_text_file,
    ".rs": load_text_file,
    ".md": load_md_file,
    ".txt": load_text_file,
    ".pdf": load_pdf_file,
    ".docx": load_docx_file,
    ".html": load_html_file,
    ".htm": load_html_file,
}

def load_document(path):
    ext = os.path.splitext(path)[1].lower()
    loader = DOCUMENT_LOADERS.get(ext)
    if not loader:
        raise RuntimeError(f"Unsupported document type: {ext}")
    return loader(path)