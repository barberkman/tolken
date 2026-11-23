"""Reading PDF, Word and image files for a model: their text, and images as PNG in base64.

tolken.py uses these.
"""
import base64
import io
import zipfile
from xml.etree import ElementTree

import fitz  # PyMuPDF

MAX_SIDE = 1280         # pixels; images and scanned PDF pages are sent no larger
PDF_PAGE_IMAGES = 10    # at most this many scanned pages of a PDF are sent as images
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"  # the namespace of a .docx's XML


def read_pdf(data):
    """The text of a PDF's pages, and the pages without text (scans) as PNG images in base64."""
    texts, images, skipped = [], [], 0
    with fitz.open(stream=data, filetype="pdf") as doc:
        for number, page in enumerate(doc, 1):
            text = page.get_text().strip()
            if text:
                texts.append(f"[page {number}]\n{text}")
            elif len(images) < PDF_PAGE_IMAGES:
                zoom = MAX_SIDE / max(page.rect.width, page.rect.height)
                png = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom)).tobytes("png")
                images.append({"page": number, "data": base64.b64encode(png).decode()})
            else:
                skipped += 1
    note = f"{skipped} more scanned page(s) were left out" if skipped else ""
    return {"text": "\n\n".join(texts), "images": images, "note": note}


def paragraph(p):
    """A Word paragraph's text: its w:t runs, with tabs and line breaks."""
    return "".join(n.text or "" if n.tag == W + "t" else "\t" if n.tag == W + "tab" else "\n"
                   for n in p.iter() if n.tag in (W + "t", W + "tab", W + "br", W + "cr"))


def read_docx(data):
    """The text of a Word document (.docx): a line per paragraph, and a line per table row with its cells split by |."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            body = ElementTree.fromstring(archive.read("word/document.xml")).find(W + "body")
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError):
        raise ValueError("not a Word (.docx) document")
    lines = []
    for block in body:
        if block.tag == W + "tbl":
            lines += [" | ".join(" ".join(paragraph(p) for p in cell.iter(W + "p")) for cell in row.findall(W + "tc"))
                      for row in block.findall(W + "tr")]
        else:  # a paragraph, or a content control (w:sdt) holding some
            lines += [paragraph(p) for p in block.iter(W + "p")]
    return {"text": "\n".join(lines).strip()}


def read_image(data):
    """An image file's bytes as a PNG in base64, shrunk to MAX_SIDE pixels on its longer side when larger.

    Raises ValueError when the bytes are not an image PyMuPDF can read.
    """
    try:
        image = fitz.Pixmap(data)
        if max(image.width, image.height) > MAX_SIDE:
            with fitz.open(stream=data) as doc:  # an image opens as a one-page document, measured at 96 dpi
                rect = doc[0].rect
                zoom = MAX_SIDE / max(rect.width, rect.height)
                image = doc[0].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        if image.n - image.alpha > 3:  # CMYK, which PNG cannot hold
            image = fitz.Pixmap(fitz.csRGB, image)
        return base64.b64encode(image.tobytes("png")).decode()
    except (fitz.mupdf.FzErrorBase, RuntimeError, ValueError) as error:  # PyMuPDF's errors
        raise ValueError(f"not an image that can be read ({error})") from error
