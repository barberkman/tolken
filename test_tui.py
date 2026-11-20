"""Headless tests for tui.py, using Textual's pilot. Run: python -m unittest test_tui

The model is replaced by canned answers; no llama-server is needed.
"""
import os
import re
import tempfile
import threading
import time
import unittest
from unittest import mock

from textual import events

import tolken
import tui


def write(path, text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


class FakeModel:
    """Stands in for a model class: records what the app does with it."""

    def __init__(self, name, fail=False):
        self.name, self.label, self.detail, self.size, self.url = name, f"fake · {name}", "1.0 GB", 10_000, "fake://"
        self.fail, self.events = fail, []

    def enter(self):
        if self.fail:
            raise OSError("no such model")
        self.events.append("enter")

    def leave(self):
        self.events.append("leave")

    def count(self, messages):
        return 10

    def chat(self, messages, ui, session):
        self.events.append("chat")
        ui.text("from " + self.name)
        ui.text("\n")
        return "from " + self.name


class AppTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(tmp.name)
        self.answers = []  # canned model answers, one per call
        self.prompts = []  # the last user message of every call
        self.hold = None   # an Event; while it is unset the fake model stalls after its first words

        def chat(messages, url, ui, session, size=None, think=None):
            self.prompts.append(messages[-1]["content"])
            answer = self.answers.pop(0)
            ui.text(answer[:7])
            while self.hold and not self.hold.is_set():
                if session.stopped:
                    return answer[:7]
                time.sleep(0.01)
            ui.text(answer[7:])
            ui.timings({"cache_n": 100, "prompt_n": 20, "predicted_n": 5, "predicted_ms": 250.0})
            ui.text("\n")
            return answer

        self.shift = False  # what tui.shift_held() reports, instead of the real keyboard
        patches = [mock.patch.object(tolken, "chat", chat),
                   mock.patch.object(tolken, "count", lambda messages, url: sum(len(m["content"]) for m in messages) // 4),
                   mock.patch.object(tui, "shift_held", lambda: self.shift)]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.app = tui.TolkenApp(8192, "demo.gguf")

    def items(self):
        """The text of every item in the conversation area; for an answer, the markdown it shows."""
        return [(widget.text if isinstance(widget, tui.Answer) else getattr(widget.content, "plain", widget.content))
                .rstrip() for widget in self.app.conversation.children]

    def status(self):
        return str(self.app.query_one("#status").content)

    async def submit(self, pilot, text):
        with self.app.prompt.prevent(tui.TextArea.Changed):  # put in place without opening the @ suggestions
            self.app.prompt.value = text
        await pilot.press("enter")

    async def until(self, pilot, condition):
        deadline = time.monotonic() + 5
        while not condition():
            self.assertLess(time.monotonic(), deadline, f"timed out; conversation: {self.items()}")
            await pilot.pause(0.02)
        await pilot.pause(0.15)  # let the refresh timer run once more

    async def test_question_answer_and_status(self):
        self.answers = ["Hello there, friend."]
        async with self.app.run_test() as pilot:
            header = str(self.app.query_one("#header").content)
            self.assertRegex(header, r"^tolken · .+ · llama\.cpp · demo · 8,192 tokens$")
            self.assertLessEqual(len(header), 100)
            self.assertTrue(self.status().startswith("ready │ 0/8,192 (0%)"))
            await self.submit(pilot, "zzz hi")
            await self.until(pilot, lambda: not self.app.busy)
            items = self.items()
            self.assertEqual(items[0], "> zzz hi")
            self.assertRegex(items[1], r"^no code \| \d+/8192 tokens$")
            self.assertEqual(items[2], "Hello there, friend.")
            self.assertRegex(items[3], r"^\d+\.\ds · 1 call · prompt 120 \(100 cached\) · answer 5 · 20 tok/s$")
            self.assertTrue(self.status().startswith("ready │ 125/8,192 (1%) │ enter send"))

    async def test_summary_adds_up_prompt_time_and_leaves_out_an_unknown_cache(self):
        figures = [{"cache_n": None, "prompt_n": 900, "prompt_ms": 1500.0, "predicted_n": 5, "predicted_ms": 250.0},
                   {"cache_n": None, "prompt_n": 1000, "prompt_ms": 200.0, "predicted_n": 5, "predicted_ms": 250.0}]

        def chat(messages, url, ui, session, size=None, think=None):  # like Ollama, which does not say what was cached
            ui.timings(figures[len(self.prompts)])
            self.prompts.append(messages[-1]["content"])
            return "NEED: a.py" if len(self.prompts) == 1 else "done"

        write("a.py", "x\n")
        with mock.patch.object(tolken, "chat", chat):
            async with self.app.run_test() as pilot:
                await self.submit(pilot, "zzz hi")
                await self.until(pilot, lambda: not self.app.busy)
                self.assertRegex(self.items()[-1],
                                 r"^\d+\.\ds · 2 calls · prompt 1,000 · prompt time 1\.7s · answer 10 · 20 tok/s$")
                self.assertTrue(self.status().startswith("ready │ 1,005/8,192"))

    async def test_file_requests_show_each_call(self):
        write("a.py", "body of a\n")
        self.answers = ["NEED: a.py", "It prints a body."]
        async with self.app.run_test() as pilot:
            await self.submit(pilot, "zzz what is in there")
            await self.until(pilot, lambda: not self.app.busy)
            items = self.items()
            self.assertEqual(items[2], "NEED: a.py")
            self.assertRegex(items[3], r"^\+ a\.py \| \d+/8192 tokens$")  # added to the conversation
            self.assertEqual(items[4], "It prints a body.")
            self.assertIn("2 calls", items[5])
            self.assertIn("body of a", self.prompts[1])

    async def test_a_phrase_shows_above_the_input_while_working(self):
        self.answers = ["A long answer that never ends"]
        self.hold = threading.Event()
        async with self.app.run_test() as pilot:
            flavor = self.app.query_one("#flavor")
            self.assertFalse(flavor.display)
            await self.submit(pilot, "zzz first")
            await self.until(pilot, lambda: "A long" in self.items())
            self.assertTrue(flavor.display)
            line = re.fullmatch(r"✦ (.+…) \(↓ 1 token\)", str(flavor.content))  # one piece is about a token
            self.assertIn(line[1], tui.PHRASES)  # and too soon for a speed
            self.app.call_first -= 2  # as if the first piece came two seconds before the last
            await pilot.pause(0.15)
            self.assertRegex(str(flavor.content), r"\(↓ 1 token · \d+ tok/s\)$")
            self.assertTrue(self.status().startswith("answering · "))  # the status bar keeps its text
            self.hold.set()
            await self.until(pilot, lambda: not self.app.busy)
            self.assertFalse(flavor.display)

    async def test_escape_cancels_and_the_next_message_works(self):
        self.answers = ["A long answer that never ends", "Second answer."]
        self.hold = threading.Event()
        async with self.app.run_test() as pilot:
            await self.submit(pilot, "zzz first")
            await self.until(pilot, lambda: "A long" in self.items())
            self.assertRegex(self.status(), r"^answering · call 1 · \d+\.\ds │ .* │ esc cancel$")
            await self.submit(pilot, "typed while busy")  # refused; the text stays in the input
            self.assertEqual(self.app.query_one(tui.Prompt).value, "typed while busy")
            self.assertEqual(len(self.prompts), 1)
            self.assertTrue(self.status().endswith("│ still working · esc cancels"))
            await pilot.press("escape")
            await self.until(pilot, lambda: not self.app.busy)
            self.assertIn("cancelled", self.items())
            self.assertNotIn("A long answer that never ends", self.items())
            self.hold = None
            await self.submit(pilot, "zzz second")
            await self.until(pilot, lambda: not self.app.busy)
            self.assertIn("Second answer.", self.items())

    BLOCK = "{0}\n<<<<<<< SEARCH\n{1}\n=======\n{2}\n>>>>>>> REPLACE\n"

    def test_diff_card(self):
        old = "".join(f"line {i}\n" for i in range(1, 13))
        new = old.replace("line 2\n", "line two\n").replace("line 11\n", "")
        title, body = tui.diff_card("src/a.py", old, new)
        self.assertEqual(title, "edit src/a.py · +1 −2")
        self.assertEqual(body.plain, " 1   line 1\n 2 - line 2\n 2 + line two\n 3   line 3\n 4   line 4\n   ⋮\n"
                                     " 9   line 9\n10   line 10\n11 - line 11\n11   line 12")
        title, body = tui.diff_card("new.txt", "", "hello\nthere\n")
        self.assertEqual((title, body.plain), ("new file new.txt · +2 −0", "1 + hello\n2 + there"))

    def test_without_blocks_removes_only_that_files_blocks(self):
        answer = ("First:\n\n```python\n" + self.BLOCK.format("a.py", "x", "y") + "```\n\nthen\n"
                  + self.BLOCK.format("`b.py`", "p", "q") + "Done.")
        self.assertEqual(tui.without_blocks(answer, "a.py"),
                         "First:\n\nthen\n" + self.BLOCK.format("`b.py`", "p", "q") + "Done.")
        self.assertEqual(tui.without_blocks(tui.without_blocks(answer, "a.py"), "b.py"), "First:\n\nthen\n\nDone.")

    def test_without_blocks_removes_the_blocks_that_follow_a_files_block(self):
        more = "<<<<<<< SEARCH\np\n=======\nq\n>>>>>>> REPLACE\n"
        answer = "First:\n" + self.BLOCK.format("a.py", "x", "y") + more + "Done."
        self.assertEqual(tui.without_blocks(answer, "a.py"), "First:\n\nDone.")

    async def run_edit(self, key):
        write("a.txt", "one\ntwo\n")
        self.answers = ["Change it:\n\n```\n" + self.BLOCK.format("a.txt", "two", "2") + "```\n", "DONE"]
        async with self.app.run_test() as pilot:
            prompt = self.app.prompt
            await self.submit(pilot, "zzz change two")
            await self.until(pilot, lambda: self.app.asking)
            self.assertFalse(self.app.query_one("#flavor").display)  # the model is not working while it waits
            self.assertEqual(self.status().split(" │ ")[0], "apply edit?")
            self.assertTrue(self.status().endswith("│ " + tui.EDIT_KEYS))
            self.assertEqual(str(prompt.border_title), "Apply this change to a.txt?")
            self.assertEqual(prompt.placeholder, tui.EDIT_KEYS)
            self.assertEqual(self.items()[2:], ["Change it:", "1   one\n2 - two\n2 + 2"])  # no raw block; the card
            self.assertEqual(str(self.app.query(".card").first().border_title), "edit a.txt · +1 −1")
            await pilot.press("y", "x")  # other keys do not answer the question
            await pilot.pause(0.2)
            self.assertTrue(self.app.asking)
            await pilot.press(key)
            await self.until(pilot, lambda: not self.app.busy)
            self.assertEqual((prompt.placeholder, prompt.value, prompt.disabled), (tui.PLACEHOLDER, "", False))
            self.assertTrue(prompt.has_focus)
            return self.items()

    async def test_edit_is_applied_on_enter(self):
        self.assertIn("✓ applied a.txt", await self.run_edit("enter"))
        with open("a.txt", encoding="utf-8") as f:
            self.assertEqual(f.read(), "one\n2\n")

    async def test_edit_is_skipped_on_escape(self):
        self.assertIn("skipped a.txt", await self.run_edit("escape"))
        with open("a.txt", encoding="utf-8") as f:
            self.assertEqual(f.read(), "one\ntwo\n")

    async def test_each_file_is_asked_about_separately(self):
        write("a.txt", "one\n")
        write("b.txt", "two\n")
        self.answers = [self.BLOCK.format("a.txt", "one", "1") + "\n" + self.BLOCK.format("b.txt", "two", "2")]
        async with self.app.run_test() as pilot:
            await self.submit(pilot, "zzz change both")
            await self.until(pilot, lambda: self.app.asking)
            self.assertEqual(str(self.app.prompt.border_title), "Apply this change to a.txt?")
            await pilot.press("enter")
            await self.until(pilot, lambda: self.app.asking and "b.txt" in str(self.app.prompt.border_title))
            await pilot.press("escape")
            await self.until(pilot, lambda: not self.app.busy)
            self.assertEqual(self.items()[2:6], ["1 - one\n1 + 1", "✓ applied a.txt", "1 - two\n1 + 2", "skipped b.txt"])
            with open("a.txt", encoding="utf-8") as f, open("b.txt", encoding="utf-8") as g:
                self.assertEqual((f.read(), g.read()), ("1\n", "two\n"))
            self.assertEqual(self.prompts, ["(No file contents have been shown yet.)\n\nzzz change both"])

    async def test_shift_tab_switches_between_asking_and_making_changes_at_once(self):
        async with self.app.run_test() as pilot:
            prompt = self.app.prompt
            self.assertEqual((str(prompt.border_subtitle), prompt.has_class("auto")), ("ask · shift+tab", False))
            await pilot.press("shift+tab")
            self.assertEqual((self.app.session.approve, str(prompt.border_subtitle), prompt.has_class("auto")),
                             ("auto", "auto · shift+tab", True))
            await pilot.press("shift+tab")
            self.assertEqual((self.app.session.approve, str(prompt.border_subtitle), prompt.has_class("auto")),
                             ("ask", "ask · shift+tab", False))

    async def test_in_auto_mode_changes_are_shown_and_made_without_a_key(self):
        write("a.txt", "one\n")
        write("b.txt", "two\n")
        self.answers = [self.BLOCK.format("a.txt", "one", "1") + "DELETE: b.txt\n", "DONE"]
        async with self.app.run_test() as pilot:
            await pilot.press("shift+tab")
            await self.submit(pilot, "zzz change it")
            await self.until(pilot, lambda: not self.app.busy)
            items = self.items()
            self.assertLess(items.index("✓ applied a.txt"), items.index("✓ deleted b.txt"))
            self.assertEqual([card.content.plain for card in self.app.query(".card")], ["1 - one\n1 + 1", "1 - two"])
            with open("a.txt", encoding="utf-8") as f:
                self.assertEqual((f.read(), os.path.exists("b.txt")), ("1\n", False))

    async def test_shift_tab_applies_the_waiting_change_and_makes_the_rest_at_once(self):
        write("a.txt", "one\n")
        write("b.txt", "two\n")
        self.answers = [self.BLOCK.format("a.txt", "one", "1") + "\n" + self.BLOCK.format("b.txt", "two", "2"), "DONE"]
        async with self.app.run_test() as pilot:
            await self.submit(pilot, "zzz change both")
            await self.until(pilot, lambda: self.app.asking)
            await pilot.press("shift+tab")
            await self.until(pilot, lambda: not self.app.busy)
            self.assertEqual((self.app.session.approve, str(self.app.prompt.border_subtitle)),
                             ("auto", "auto · shift+tab"))
            self.assertIn("✓ applied b.txt", self.items())  # not asked about
            with open("a.txt", encoding="utf-8") as f, open("b.txt", encoding="utf-8") as g:
                self.assertEqual((f.read(), g.read()), ("1\n", "2\n"))

    async def test_the_settings_can_start_in_auto_mode(self):
        with mock.patch.object(tolken, "APPROVE", "auto"):
            self.app = tui.TolkenApp(8192, "demo.gguf")
        async with self.app.run_test():
            self.assertEqual(str(self.app.prompt.border_subtitle), "auto · shift+tab")

    async def test_a_command_waits_for_enter_and_its_output_follows(self):
        self.answers = ['Testing.\nRUN: python -c "print(42)"\n', "Done."]
        async with self.app.run_test() as pilot:
            await self.submit(pilot, "zzz test it")
            await self.until(pilot, lambda: self.app.asking)
            self.assertEqual((str(self.app.prompt.border_title), self.app.prompt.placeholder),
                             ("Run this command?", tui.RUN_KEYS))
            card = self.app.query(".card").last()
            self.assertEqual((str(card.border_title), card.content.plain), ("run", 'python -c "print(42)"'))
            await pilot.press("enter")
            await self.until(pilot, lambda: not self.app.busy)
            card = self.app.query(".card").last()
            self.assertRegex(str(card.border_title), r'^python -c "print\(42\)" · exit 0 · \d+\.\ds$')
            self.assertEqual(card.content.plain, "42")
            self.assertTrue(any(item.startswith('✓ ran python -c "print(42)": exit 0') for item in self.items()))
            self.assertIn("42", self.prompts[1])

    async def test_a_refused_command_ends_the_message(self):
        self.answers = ['RUN: python -c "print(42)"\n', "never"]
        async with self.app.run_test() as pilot:
            await self.submit(pilot, "zzz test it")
            await self.until(pilot, lambda: self.app.asking)
            await pilot.press("escape")
            await self.until(pilot, lambda: not self.app.busy)
            self.assertIn('did not run python -c "print(42)"', self.items())
            self.assertEqual(len(self.prompts), 1)

    def test_edit_blocks_outside_a_fence_are_shown_in_one(self):
        block = "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE"
        self.assertEqual(tui.displayed(f"Here.\nFILE: a.py\n{block}\nDone."),
                         f"Here.\nFILE: a.py\n\n~~~~\n{block}\n~~~~\nDone.")
        fenced = f"FILE: a.py\n```python\n{block}\n```\nDone."
        self.assertEqual(tui.displayed(fenced), fenced.replace("FILE: a.py\n", "FILE: a.py\n\n"))
        # A block still arriving stays in an open fence, and the text only grows as it arrives.
        text = f"FILE: README.md\n<<<<<<< SEARCH\n```\nold\n```\n=======\nnew\n>>>>>>> REPLACE\nNEED: b.py\nok"
        shown = [tui.displayed(text[:end]) for end in range(len(text) + 1)]
        self.assertTrue(shown[text.index("```") + 1].endswith("~~~~\n<<<<<<< SEARCH\n`"))
        self.assertEqual(shown[-1], "FILE: README.md\n\n~~~~\n<<<<<<< SEARCH\n```\nold\n```\n=======\nnew\n"
                                    ">>>>>>> REPLACE\n~~~~\nNEED: b.py\n\nok")
        grows = [later.startswith(earlier) for earlier, later in zip(shown, shown[1:])]
        self.assertEqual(grows.count(False), 1)  # only where the "<<<<<<< SEARCH" line ends

    async def test_a_dropped_file_is_attached_and_sent_as_a_pin(self):
        write("my spec.pdf", "x")
        path = os.path.abspath("my spec.pdf")
        async with self.app.run_test() as pilot:
            prompt = self.app.prompt
            prompt.post_message(events.Paste(f'"{path}"'))
            await pilot.pause()
            self.assertEqual(prompt.value, "[file my spec.pdf] ")
            prompt.insert("what is in it?")
            self.assertEqual(prompt.take(), ("[file my spec.pdf] what is in it?", f'@"{path}" what is in it?'))

    async def test_a_quoted_path_typed_in_is_attached(self):
        # What Terminator types for an image pasted with Ctrl+V, or for files dropped: PowerShell-quoted paths.
        for name in ("x.png", "b.txt", "it's.png"):
            write(name, "x")
        x, b, quote = (os.path.abspath(name) for name in ("x.png", "b.txt", "it's.png"))
        async with self.app.run_test() as pilot:
            prompt = self.app.prompt
            for character in f"'{x}' '{b}' '{quote.replace(chr(39), chr(39) * 2)}' ":
                prompt.insert(character)
                await pilot.pause()
            self.assertEqual(prompt.value, "[image x.png] [file b.txt] [image it's.png] ")
            prompt.insert("what is it?")
            self.assertEqual(prompt.take()[1], f'@"{x}" @"{b}" @"{quote}" what is it?')
            for text in ("'README.md' ", f"'{os.path.abspath('missing.png')}' ", f"'{x}"):
                prompt.insert(text)
                await pilot.pause()
                self.assertEqual(prompt.value, text)
                prompt.value = ""

    async def test_alt_v_attaches_copied_files_and_images_and_pastes_text(self):
        write("a.txt", "x\n")
        held = [("files", [os.path.abspath("a.txt")]), ("image", b"\x89PNG fake"), ("text", "hello"), None]
        with mock.patch.object(tui, "read_clipboard", lambda: held.pop(0)):
            async with self.app.run_test() as pilot:
                prompt = self.app.prompt
                await pilot.press("alt+v")
                self.assertEqual(prompt.value, "[file a.txt] ")
                await pilot.press("alt+v")
                self.assertRegex(prompt.value, r"^\[file a\.txt\] \[image clipboard-\d{8}-\d{6}\.png\] $")
                saved = next(value[0] for marker, value in prompt.pastes.items() if marker.startswith("[image"))
                self.addCleanup(os.remove, saved)
                with open(saved, "rb") as f:
                    self.assertEqual(f.read(), b"\x89PNG fake")
                self.assertEqual(os.path.dirname(saved), os.path.join(tempfile.gettempdir(), "tolken"))
                await pilot.press("alt+v")
                self.assertTrue(prompt.value.endswith("] hello"))
                await pilot.press("alt+v")
                await pilot.pause()
                self.assertTrue(self.status().endswith("│ the clipboard holds nothing to attach")
                                or self.status().endswith("│ alt+v works on Windows only"))

    def test_a_clipboard_bitmap_becomes_a_png(self):
        import struct
        import fitz
        width, height = 3, 2  # 24 bits a pixel, rows padded to 4 bytes, bottom row first
        row = b"\x00\x00\xff" * width + b"\x00" * ((4 - width * 3 % 4) % 4)
        dib = struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, len(row) * height, 0, 0, 0, 0) + row * height
        image = fitz.Pixmap(tui.bitmap_png(dib))
        self.assertEqual((image.width, image.height), (width, height))
        self.assertEqual(image.pixel(0, 0)[:3], (255, 0, 0))  # stored as blue, green, red

    def test_delete_card_shows_how_the_file_begins(self):
        title, body = tui.diff_card("a.txt", "one\ntwo\n", None)
        self.assertEqual((title, body.plain), ("delete a.txt · 2 lines", "1 - one\n2 - two"))
        title, body = tui.diff_card("big.txt", "".join(f"line {i}\n" for i in range(1, 21)), None)
        self.assertEqual(title, "delete big.txt · 20 lines")
        self.assertEqual(body.plain.splitlines()[-2:], ["12 - line 12", "   ⋮ 8 more lines"])

    async def test_a_file_is_deleted_only_after_enter(self):
        for key, kept in (("escape", True), ("enter", False)):
            with self.subTest(key=key):
                write("a.txt", "one\ntwo\n")
                self.app = tui.TolkenApp(8192, "demo.gguf")
                self.answers = ["It is not needed.\nDELETE: a.txt\n", "DONE"]
                async with self.app.run_test() as pilot:
                    await self.submit(pilot, "zzz remove a.txt")
                    await self.until(pilot, lambda: self.app.asking)
                    self.assertEqual(str(self.app.prompt.border_title), "Delete a.txt?")
                    self.assertEqual(self.status().split(" │ ")[0], "delete file?")
                    self.assertTrue(self.status().endswith("│ " + tui.DELETE_KEYS))
                    card = self.app.query(".card").first()
                    self.assertEqual((str(card.border_title), card.content.plain),
                                     ("delete a.txt · 2 lines", "1 - one\n2 - two"))
                    await pilot.press(key)
                    await self.until(pilot, lambda: not self.app.busy)
                    self.assertIn("kept a.txt" if kept else "✓ deleted a.txt", self.items())
                    self.assertEqual(os.path.exists("a.txt"), kept)
                    self.assertEqual(self.app.prompt.placeholder, tui.PLACEHOLDER)

    async def test_an_edit_that_cannot_be_applied_says_so(self):
        write("a.txt", "one\n")
        self.answers = [self.BLOCK.format("a.txt", "missing", "x"), "I cannot find that line."]
        async with self.app.run_test() as pilot:
            await self.submit(pilot, "zzz change it")
            await self.until(pilot, lambda: not self.app.busy)
            items = self.items()
            failure = "could not apply an edit to a.txt: its SEARCH text is not in the file"
            self.assertIn("<<<<<<< SEARCH", items[2])  # the model's text stays, since no card replaced it
            self.assertEqual(items[3], "✗ " + failure)
            self.assertEqual(items[5], "I cannot find that line.")  # the model was told, and answered again
            # The file is shown with the failure, so its text can be copied exactly.
            self.assertEqual(self.prompts[1], "--- a.txt (lines 1-1 of 1) ---\none\n\n" + failure + "\n" + tolken.GO_ON)
            self.assertEqual(len(self.app.query(".card")), 0)

    async def test_multi_line_paste_is_sent_whole(self):
        self.answers = ["ok"]
        async with self.app.run_test() as pilot:
            prompt = self.app.query_one(tui.Prompt)
            prompt.post_message(events.Paste("zzz one"))
            await pilot.pause()
            self.assertEqual(prompt.value, "zzz one")
            prompt.post_message(events.Paste("error: line 1\n  at line 2\nline 3\n"))
            await pilot.pause()
            self.assertEqual(prompt.value, "zzz one[pasted 3 lines #1]")
            await pilot.press("enter")
            await self.until(pilot, lambda: not self.app.busy)
            self.assertEqual(self.items()[0], "> zzz one[pasted 3 lines #1]")
            self.assertTrue(self.prompts[0].endswith("zzz one\nerror: line 1\n  at line 2\nline 3"))

    async def type(self, pilot, text):
        await pilot.press(*("space" if character == " " else character for character in text))

    def test_matching_puts_file_name_matches_first(self):
        files = ["a.py", "src/cart.cpp", "src/cart.h", "src/notes.md", "cart/readme.md"]
        self.assertEqual(tui.matching(files, ""), ["a.py", "src/cart.h", "src/cart.cpp", "src/notes.md", "cart/readme.md"])
        self.assertEqual(tui.matching(files, "CA"), ["src/cart.h", "src/cart.cpp", "cart/readme.md"])
        self.assertEqual(tui.matching(files, "src\\ca"), ["src/cart.h", "src/cart.cpp"])
        self.assertEqual(tui.matching(files, "", limit=2), ["a.py", "src/cart.h"])

    async def test_at_sign_suggests_files_and_tab_inserts_one(self):
        for path in ("a.py", "src/cart.cpp", "src/cart.h", "src/notes.md", "node_modules/x.js"):
            write(path, f"body of {path}\n")
        self.answers = ["ok"]
        async with self.app.run_test() as pilot:
            prompt, suggestions = self.app.prompt, self.app.suggestions
            await self.type(pilot, "look at ")
            self.assertFalse(suggestions.display)
            await self.type(pilot, "@")
            self.assertTrue(suggestions.display)
            self.assertEqual(self.app.matches, ["a.py", "src/cart.h", "src/cart.cpp", "src/notes.md"])
            await self.type(pilot, "ca")
            self.assertEqual(suggestions.content.plain, "› src/cart.h\n  src/cart.cpp")
            await pilot.press("down", "down", "up")
            self.assertEqual(suggestions.content.plain, "  src/cart.h\n› src/cart.cpp")
            await pilot.press("tab")
            await pilot.pause()
            self.assertEqual(prompt.value, "look at @src/cart.cpp ")
            self.assertFalse(suggestions.display)
            await pilot.press("enter")
            await self.until(pilot, lambda: not self.app.busy)
            self.assertIn("--- src/cart.cpp (lines 1-1 of 1) ---\nbody of src/cart.cpp", self.prompts[0])

    async def test_enter_picks_a_suggestion_and_escape_closes_them(self):
        write("a.py", "x\n")
        self.hold = threading.Event()
        async with self.app.run_test() as pilot:
            await self.type(pilot, "@a")
            await pilot.press("enter")
            await pilot.pause()
            self.assertEqual(self.app.prompt.value, "@a.py ")
            self.assertEqual(self.prompts, [])  # Enter chose the file; nothing was sent
            await self.type(pilot, "@")
            self.assertTrue(self.app.suggestions.display)
            await pilot.press("escape")
            self.assertFalse(self.app.suggestions.display)
            self.assertEqual(self.app.prompt.value, "@a.py @")

    async def test_shift_enter_and_ctrl_j_start_a_new_line(self):
        self.answers = ["ok"]
        async with self.app.run_test() as pilot:
            prompt = self.app.prompt
            self.assertEqual(prompt.size.height, 1)
            await self.type(pilot, "zzz one")
            await pilot.press("shift+enter")
            await self.type(pilot, "two")
            await pilot.press("ctrl+j")
            await self.type(pilot, "three")
            await pilot.pause()
            self.assertEqual(prompt.value, "zzz one\ntwo\nthree")
            self.assertEqual(prompt.size.height, 3)  # the box grows with the text
            self.assertEqual(self.prompts, [])  # nothing was sent
            await pilot.press("up")  # inside a multi-line text, Up moves the cursor, not the history
            self.assertEqual((prompt.value, prompt.cursor_location[0]), ("zzz one\ntwo\nthree", 1))
            await pilot.press("enter")
            await self.until(pilot, lambda: not self.app.busy)
            self.assertTrue(self.prompts[0].endswith("\n\nzzz one\ntwo\nthree"))
            self.assertEqual(self.items()[0], "> zzz one\ntwo\nthree")
            self.assertEqual((prompt.value, prompt.size.height), ("", 1))
            await pilot.press("up")  # the whole message comes back from the history
            self.assertEqual(prompt.value, "zzz one\ntwo\nthree")
            await pilot.press("up")  # the cursor is on its last line, so Up first moves within it
            self.assertEqual((prompt.value, prompt.cursor_location[0]), ("zzz one\ntwo\nthree", 1))

    async def test_enter_with_shift_held_down_starts_a_new_line(self):
        self.answers = ["ok"]
        async with self.app.run_test() as pilot:
            await self.type(pilot, "zzz one")
            self.shift = True  # the terminal reports a plain Enter, but the Shift key is down
            await pilot.press("enter")
            self.shift = False
            await self.type(pilot, "two")
            await pilot.pause()
            self.assertEqual((self.app.prompt.value, self.prompts), ("zzz one\ntwo", []))
            await pilot.press("enter")
            await self.until(pilot, lambda: not self.app.busy)
            self.assertTrue(self.prompts[0].endswith("\n\nzzz one\ntwo"))

    async def test_up_and_down_recall_earlier_messages(self):
        write("a.py", "x\n")
        self.answers = ["one", "two", "three"]
        async with self.app.run_test() as pilot:
            prompt = self.app.prompt

            async def press(key, expected):
                await pilot.press(key)
                await pilot.pause()
                self.assertEqual(prompt.value, expected)
                self.assertEqual(prompt.cursor_location, prompt.document.end)

            await press("up", "")  # nothing has been sent yet
            for text in ("zzz first", "zzz look at @a.py", "zzz look at @a.py"):  # the repeat is stored once
                await self.submit(pilot, text)
                await self.until(pilot, lambda: not self.app.busy)
            await self.type(pilot, "draft")
            await press("down", "draft")  # Down does nothing before Up
            await press("up", "zzz look at @a.py")
            self.assertFalse(self.app.suggestions.display)  # the recalled @file does not open the file list
            await press("up", "zzz first")
            await press("up", "zzz first")
            await press("down", "zzz look at @a.py")
            await press("down", "draft")
            await press("down", "draft")

    async def test_a_recalled_message_keeps_its_pasted_text(self):
        self.answers = ["one", "two"]
        async with self.app.run_test() as pilot:
            prompt = self.app.prompt
            await self.type(pilot, "zzz ")
            prompt.post_message(events.Paste("line 1\nline 2\n"))
            await pilot.pause()
            await pilot.press("enter")
            await self.until(pilot, lambda: not self.app.busy)
            await pilot.press("up")
            self.assertEqual(prompt.value, "zzz [pasted 2 lines #1]")
            await pilot.press("enter")
            await self.until(pilot, lambda: not self.app.busy)
            self.assertTrue(self.prompts[0].endswith("zzz \nline 1\nline 2"))
            self.assertEqual(self.prompts[1], "zzz \nline 1\nline 2")  # the paste was sent again, whole

    async def test_model_list_switches_the_model(self):
        small = FakeModel("small")
        with mock.patch.object(tolken, "models", lambda: [tolken.Llama(8192, "demo.gguf"), small]):
            async with self.app.run_test() as pilot:
                await self.submit(pilot, "/model")
                await self.until(pilot, lambda: self.app.models is not None)
                self.assertEqual(self.app.suggestions.content.plain,
                                 "› llama.cpp · demo · 8,192 tokens   (current)\n  fake · small · 1.0 GB")
                self.assertEqual(self.status().split(" │ ")[0], "choose a model")
                self.assertTrue(self.status().endswith("│ ↑↓ choose · enter select · esc close"))
                await pilot.press("escape")
                self.assertFalse(self.app.suggestions.display)
                self.assertEqual((self.app.models, self.status().split(" │ ")[0]), (None, "ready"))
                await self.submit(pilot, "/model")
                await self.until(pilot, lambda: self.app.models is not None)
                await pilot.press("down", "enter")
                await self.until(pilot, lambda: self.app.session.model is small and not self.app.busy)
                self.assertTrue(str(self.app.header.content).endswith(" · fake · small · 10,000 tokens"))
                self.assertIn("model: fake · small · 10,000 tokens", self.items())
                self.assertTrue(self.status().startswith("ready │ 0/10,000 (0%)"))
                self.assertEqual(self.app.models, None)
                await self.submit(pilot, "zzz hi")
                await self.until(pilot, lambda: "from small" in self.items())
                self.assertEqual(small.events, ["enter", "chat"])
                self.assertEqual(self.prompts, [])  # llama-server's model was not asked

    async def test_the_model_the_settings_name_is_loaded_at_startup(self):
        small = FakeModel("small")
        with mock.patch.object(tolken, "MODEL", "small"), mock.patch.object(tolken, "models", lambda: [small]):
            async with self.app.run_test() as pilot:
                await self.until(pilot, lambda: self.app.session.model is small and not self.app.busy)
                self.assertTrue(str(self.app.header.content).endswith(" · fake · small · 10,000 tokens"))

    async def test_thinking_is_shown_before_the_answer(self):
        def chat(messages, url, ui, session, size=None, think=None):
            ui.thinking("Let me ")
            ui.thinking("see.")
            ui.text("The answer.")
            ui.text("\n")
            return "The answer."

        with mock.patch.object(tolken, "chat", chat):
            async with self.app.run_test() as pilot:
                await self.submit(pilot, "zzz hi")
                await self.until(pilot, lambda: not self.app.busy)
                self.assertEqual(self.items()[2:4], ["Let me see.", "The answer."])
                self.assertEqual(len(self.app.query(".thinking")), 1)
                self.assertEqual(self.app.session.history[-1][1], "The answer.")  # the thinking is not carried forward

    async def test_model_by_name_unknown_name_and_failed_load(self):
        small, broken = FakeModel("small"), FakeModel("broken", fail=True)
        found = [tolken.Llama(8192, "demo.gguf"), small, broken]
        with mock.patch.object(tolken, "models", lambda: found):
            async with self.app.run_test() as pilot:
                await self.submit(pilot, "/model sma")
                await self.until(pilot, lambda: self.app.session.model is small and not self.app.busy)
                await self.submit(pilot, "/model nope")
                await self.until(pilot, lambda: "✗ no single model matches 'nope'" in self.items())
                await self.submit(pilot, "/model broken")
                await self.until(pilot, lambda: "✗ could not load fake · broken: no such model" in self.items())
                self.assertIs(self.app.session.model, small)
                self.assertTrue(str(self.app.header.content).endswith(" · fake · small · 10,000 tokens"))
        with mock.patch.object(tolken, "models", lambda: []):
            self.app = tui.TolkenApp(8192, "demo.gguf")
            async with self.app.run_test() as pilot:
                await self.submit(pilot, "/model")
                await self.until(pilot, lambda: "✗ no model server answers" in self.items())

    async def test_long_answers_stay_in_view_until_scrolled_up(self):
        self.answers = ["\n".join(f"line {i}" for i in range(80)), "short"]
        async with self.app.run_test() as pilot:
            box = self.app.conversation
            await self.submit(pilot, "zzz long")
            await self.until(pilot, lambda: not self.app.busy and box.max_scroll_y > 0)  # the answer is drawn
            self.assertEqual(box.scroll_y, box.max_scroll_y)
            await pilot.press("pageup")
            await pilot.pause(0.3)
            self.assertLess(box.scroll_y, box.max_scroll_y)
            await self.submit(pilot, "zzz again")  # a new message jumps back to the end
            await self.until(pilot, lambda: not self.app.busy)
            self.assertEqual(box.scroll_y, box.max_scroll_y)

    def test_copy_uses_the_first_clipboard_command_that_works(self):
        runs = []

        def run(command, **options):
            runs.append((command, options["input"]))
            if command[0] == "clip.exe":
                raise FileNotFoundError

        with mock.patch.object(tui.os, "name", "posix"):  # on Windows itself, copy() uses the Win32 API instead
            with mock.patch.object(tui.subprocess, "run", run):
                self.assertTrue(tui.copy("héllo"))
            self.assertEqual(runs, [(["clip.exe"], "héllo".encode("utf-16-le")), (["pbcopy"], "héllo".encode("utf-8"))])
            with mock.patch.object(tui.subprocess, "run", side_effect=OSError):
                self.assertFalse(tui.copy("x"))

    async def test_ctrl_c_copies_a_selection_and_twice_without_one_exits(self):
        copied = []
        with mock.patch.object(tui, "copy", copied.append):
            async with self.app.run_test() as pilot:
                await pilot.press("ctrl+c")
                await pilot.pause()
                self.assertTrue(self.status().endswith("│ press ctrl+c again to exit"))
                self.assertEqual(len(self.app.query("Toast")), 0)  # no pop-up notice
                self.app.quit_until = self.app.hint_until = 0.0  # as if two seconds had passed
                await pilot.press("ctrl+c")
                await pilot.pause()
                self.assertTrue(self.app.is_running)  # a late second press only shows the hint again
                self.app.quit_until = 0.0
                await self.type(pilot, "hello")
                await pilot.press("shift+home", "ctrl+c", "ctrl+c")  # text selected in the input: copies, never exits
                await pilot.pause()
                self.assertEqual(copied, ["hello", "hello"])
                self.assertTrue(self.status().endswith("│ copied"))
                self.app.prompt.value = ""
                with mock.patch.object(self.app.screen, "get_selected_text", lambda: "from the conversation"):
                    await pilot.press("ctrl+c")
                    await pilot.pause()
                self.assertEqual(copied[-1], "from the conversation")
                self.assertTrue(self.app.is_running)
                await pilot.press("ctrl+c", "ctrl+c")  # nothing selected: twice exits
                await pilot.pause()
                self.assertFalse(self.app.is_running)

    async def test_clear_and_exit(self):
        self.answers = ["ok"]
        async with self.app.run_test() as pilot:
            await self.submit(pilot, "zzz hi")
            await self.until(pilot, lambda: not self.app.busy)
            self.assertTrue(self.app.session.history)
            await self.submit(pilot, "/clear")
            await pilot.pause()
            self.assertEqual(self.items(), [])
            self.assertEqual(self.app.session.history, [])
            self.assertTrue(self.status().startswith("ready │ 0/8,192"))
            await self.submit(pilot, "/exit")
            await pilot.pause()
            self.assertFalse(self.app.is_running)

    async def test_an_answer_is_shown_as_markdown_that_can_be_copied(self):
        self.answers = ["## Plan\n\nUse **bold** and `code`:\n- one\n- two\n\nfirst line\nsecond line\n"
                        "Returns Optional<User>."]
        async with self.app.run_test() as pilot:
            await self.submit(pilot, "zzz explain")
            await self.until(pilot, lambda: not self.app.busy)
            answer = self.app.query_one(tui.Answer)
            self.assertEqual(answer.source, answer.text)  # caught up
            self.assertEqual([type(block).__name__ for block in answer.children],
                             ["MarkdownH2", "MarkdownParagraph", "MarkdownBulletList", "MarkdownParagraph"])
            answer.text_select_all()
            await pilot.pause()
            selected = self.app.screen.get_selected_text()
            self.assertIn("Use bold and code:", selected)
            self.assertIn("first line\nsecond line\nReturns Optional<User>.", selected)

    async def test_undo_puts_back_the_last_changes(self):
        write("a.txt", "one\n")
        self.answers = [self.BLOCK.format("a.txt", "one", "1"), "DONE"]
        async with self.app.run_test() as pilot:
            await pilot.press("shift+tab")
            await self.submit(pilot, "zzz change it")
            await self.until(pilot, lambda: not self.app.busy)
            await self.submit(pilot, "/undo")
            await pilot.pause()
            self.assertEqual(self.items()[-1], "put back a.txt")
            with open("a.txt", encoding="utf-8") as f:
                self.assertEqual(f.read(), "one\n")
            await self.submit(pilot, "/undo")
            await pilot.pause()
            self.assertEqual(self.items()[-1], "nothing to undo")


if __name__ == "__main__":
    unittest.main()
