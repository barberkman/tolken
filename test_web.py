"""Tests for web.py. Run: python -m unittest test_web

No model server is needed; the models are fakes.
"""
import base64
import io
import threading
import time
import unittest
import zipfile
from unittest import mock

import fitz

import documents
import harness
import web


class Fake:
    """A model that answers with fixed pieces of text, or fails with error."""

    def __init__(self, label="fake", size=3000, vision=True, pieces=("Hel", "lo\n"), error=None):
        self.label, self.size, self.vision, self.pieces, self.error = label, size, vision, pieces, error
        self.detail, self.url, self.sent, self.calls = "", "http://fake", None, []

    def count(self, messages):
        return sum(len(message["content"]) for message in messages)

    def enter(self):
        self.calls.append("enter")

    def leave(self):
        self.calls.append("leave")

    def chat(self, messages, ui=None, session=None):
        self.sent = messages
        for piece in self.pieces:
            ui.text(piece)
        if self.error:
            raise self.error
        return "".join(self.pieces)


def pdf(*pages):
    """A PDF whose pages hold these texts; "" makes a page with only a drawing (like a scan)."""
    doc = fitz.open()
    for text in pages:
        page = doc.new_page(width=600, height=800)
        if text:
            page.insert_text((50, 100), text)
        else:
            page.draw_rect(fitz.Rect(50, 50, 300, 300), color=(0, 0, 0), fill=(0.5, 0.5, 0.5))
    return doc.tobytes()


class ReadPdf(unittest.TestCase):
    def test_text_pages_give_text_and_pages_without_text_give_images(self):
        result = web.read_pdf(pdf("hello pdf", ""))
        self.assertEqual(result["text"], "[page 1]\nhello pdf")
        self.assertEqual([image["page"] for image in result["images"]], [2])
        png = base64.b64decode(result["images"][0]["data"])
        self.assertTrue(png.startswith(b"\x89PNG"))
        image = fitz.Pixmap(png)
        self.assertEqual(max(image.width, image.height), documents.MAX_SIDE)
        self.assertEqual(result["note"], "")

    def test_scanned_pages_beyond_the_limit_are_left_out_with_a_note(self):
        with mock.patch.object(documents, "PDF_PAGE_IMAGES", 2):
            result = web.read_pdf(pdf("", "", "", "text"))
        self.assertEqual([image["page"] for image in result["images"]], [1, 2])
        self.assertEqual(result["text"], "[page 4]\ntext")
        self.assertIn("1 more", result["note"])


def docx(body):
    """A .docx whose document body is this XML."""
    xml = ('<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
           f'<w:body>{body}</w:body></w:document>')
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("word/document.xml", xml)
    return data.getvalue()


class ReadDocx(unittest.TestCase):
    def test_paragraphs_give_lines_with_tabs_breaks_and_links(self):
        result = web.read_docx(docx(
            "<w:p><w:r><w:t>Hello</w:t></w:r><w:r><w:tab/><w:t>world</w:t></w:r></w:p>"
            "<w:p><w:r><w:t>one</w:t><w:br/><w:t>two</w:t></w:r>"
            "<w:hyperlink><w:r><w:t> link</w:t></w:r></w:hyperlink></w:p>"))
        self.assertEqual(result["text"], "Hello\tworld\none\ntwo link")

    def test_tables_give_a_line_per_row_in_document_order(self):
        cell = "<w:tc><w:p><w:r><w:t>{}</w:t></w:r></w:p></w:tc>"
        row = "<w:tr>" + cell + cell + "</w:tr>"
        result = web.read_docx(docx(
            "<w:p><w:r><w:t>before</w:t></w:r></w:p>"
            "<w:tbl>" + row.format("a", "b") + row.format("c", "d") + "</w:tbl>"
            "<w:p><w:r><w:t>after</w:t></w:r></w:p>"))
        self.assertEqual(result["text"], "before\na | b\nc | d\nafter")

    def test_broken_files_refused(self):
        empty_zip = b"PK\x05\x06" + b"\0" * 18  # a zip without word/document.xml
        for data in (b"not a docx", empty_zip, docx("<w:p>")):  # the last has broken XML
            with self.assertRaises(ValueError):
                web.read_docx(data)


def exchange(*texts):
    """Alternating user and assistant messages with these contents."""
    return [{"role": ("user", "assistant")[i % 2], "content": text} for i, text in enumerate(texts)]


class Fit(unittest.TestCase):
    # Fake counts characters; the window leaves size - MAX_OUT - MARGIN of them for the prompt.
    def setUp(self):
        patch = mock.patch.object(harness, "MAX_OUT", 0)  # so the answer's share of these tiny windows stays 0
        patch.start()
        self.addCleanup(patch.stop)

    def model(self, room):
        return Fake(size=room + harness.MAX_OUT + harness.MARGIN)

    def test_everything_kept_when_it_fits(self):
        messages = exchange("a" * 10, "b" * 10, "c" * 10)
        self.assertEqual(web.fit(messages, self.model(30)), (messages, 30 + harness.MARGIN))

    def test_oldest_exchanges_dropped_first(self):
        messages = exchange("a" * 10, "b" * 10, "c" * 10, "d" * 10, "e" * 10)
        fitted, _ = web.fit(messages, self.model(30))
        self.assertEqual(fitted, messages[2:])
        fitted, _ = web.fit(messages, self.model(25))
        self.assertEqual(fitted, messages[4:])

    def test_images_count_toward_the_size(self):
        messages = exchange("a", "b", "c")
        messages[0]["images"] = ["x"]
        fitted, _ = web.fit(messages, self.model(10))
        self.assertEqual(fitted, messages[2:])

    def test_newest_message_alone_too_long(self):
        fitted, used = web.fit(exchange("a", "b", "c" * 50), self.model(20))
        self.assertIsNone(fitted)
        self.assertEqual(used, 50 + harness.MARGIN)


class Endless(Fake):
    """A model that answers until its call is ended."""

    def __init__(self):
        super().__init__()
        self.ended = threading.Event()

    def chat(self, messages, ui=None, session=None):
        while not session.stopped:
            ui.text("x")
            time.sleep(0.01)
        self.ended.set()


class Server(unittest.TestCase):
    def setUp(self):
        self.client = web.app.test_client()
        patcher = mock.patch.object(web, "current", None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_page_is_served(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"ASCII_PROMPT", response.data)
        response.close()

    def test_chat_streams_the_answer(self):
        web.current = Fake()
        messages = exchange("question")
        response = self.client.post("/chat", json={"messages": messages})
        self.assertEqual((response.status_code, response.data), (200, b"Hello\n"))
        self.assertEqual(response.headers["X-Dropped"], "0")
        self.assertEqual(web.current.sent, messages)

    def test_error_during_the_answer_follows_a_nul(self):
        web.current = Fake(error=OSError("the answer was cut off"))
        response = self.client.post("/chat", json={"messages": exchange("question")})
        self.assertEqual(response.data, b"Hello\n\0the answer was cut off")

    def test_closing_the_answer_ends_the_call(self):
        web.current = model = Endless()
        response = self.client.post("/chat", json={"messages": exchange("question")}, buffered=False)
        self.assertEqual(next(response.iter_encoded()), b"x")
        response.close()  # what Stop in the page amounts to
        self.assertTrue(model.ended.wait(2))

    def test_images_refused_by_a_model_that_cannot_read_them(self):
        web.current = Fake(vision=False)
        messages = exchange("what is this?")
        messages[0]["images"] = ["aGk="]
        response = self.client.post("/chat", json={"messages": messages})
        self.assertEqual(response.status_code, 400)
        self.assertIn("can't read images", response.json["error"])
        self.assertIsNone(web.current.sent)

    def test_too_long_message_refused(self):
        web.current = Fake(size=harness.MARGIN + 70)  # 3 characters are left for the prompt
        response = self.client.post("/chat", json={"messages": exchange("x" * 100)})
        self.assertEqual(response.status_code, 400)
        self.assertIn("too long", response.json["error"])

    def test_switching_leaves_the_old_model_and_enters_the_new(self):
        old, new = Fake("old"), Fake("new", size=4096)
        web.current = old
        with mock.patch.object(harness, "models", return_value=[old, new]),              mock.patch.object(web, "vision", return_value=True):
            response = self.client.post("/model", json={"label": "new"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json, {"label": "new", "size": 4096, "vision": True})
            self.assertEqual((old.calls, new.calls), (["leave"], ["enter"]))
            self.assertIs(web.current, new)
            response = self.client.get("/models")
            self.assertEqual([m["current"] for m in response.json], [False, True])

    def test_start_loads_the_model_the_settings_name_else_the_first_listed(self):
        first, second = Fake("ollama · first"), Fake("ollama · second")
        first.name, second.name = "first", "second"
        with mock.patch.object(harness, "models", return_value=[first, second]), \
                mock.patch.object(web, "vision", return_value=False), mock.patch("builtins.print"):
            web.start()
            self.assertIs(web.current, first)
            with mock.patch.object(harness, "MODEL", "second"):
                web.start()
            self.assertIs(web.current, second)
            self.assertEqual((first.calls, second.calls), (["enter", "leave"], ["enter"]))
            response = self.client.get("/models")
            self.assertEqual([m["current"] for m in response.json], [False, True])  # what the page opens on

    def test_unknown_model_refused(self):
        with mock.patch.object(harness, "models", return_value=[]):
            response = self.client.post("/model", json={"label": "nope"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("nope", response.json["error"])

    def test_pdf_route(self):
        response = self.client.post("/pdf", data=pdf("hello"))
        self.assertEqual(response.json["text"], "[page 1]\nhello")

    def test_broken_pdf_refused(self):
        response = self.client.post("/pdf", data=b"not a pdf")
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json)

    def test_docx_route(self):
        response = self.client.post("/docx", data=docx("<w:p><w:r><w:t>hello</w:t></w:r></w:p>"))
        self.assertEqual(response.json["text"], "hello")

    def test_broken_docx_refused(self):
        response = self.client.post("/docx", data=b"not a docx")
        self.assertEqual(response.status_code, 400)
        self.assertIn("Word", response.json["error"])


if __name__ == "__main__":
    unittest.main()
