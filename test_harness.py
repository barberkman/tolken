"""Tests for harness.py. Run: python -m unittest test_harness

No llama-server is needed; the server functions run against a fake one.
"""
import contextlib
import http.server
import io
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from unittest import mock

import harness


def write(path, text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


class Out(io.StringIO):
    """Stands in for sys.stdout, which main() reconfigures."""

    def reconfigure(self, **_):
        pass


class InTempDir(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(tmp.name)
        out = contextlib.redirect_stdout(io.StringIO())
        self.printed = out.__enter__()
        self.addCleanup(out.__exit__, None, None, None)


class Pins(InTempDir):
    def test_whole_file_and_range(self):
        write("src/a.py", "x\n")
        self.assertEqual(harness.parse_pins("see @src/a.py and @src/a.py:2-5."),
                         [["src/a.py", 1, None], ["src/a.py", 2, 5]])

    def test_absolute_path_with_range(self):
        write("a.py", "x\n")
        [(path, start, end)] = harness.parse_pins("@" + os.path.abspath("a.py") + ":10-20")
        self.assertTrue(os.path.samefile(path, "a.py"))
        self.assertEqual((start, end), (10, 20))

    def test_mention_that_is_not_a_file_is_ignored(self):
        self.assertEqual(harness.parse_pins("use @property or mail me@example.com"), [])

    def test_snippet_clamps_range(self):
        write("a.py", "one\ntwo\nthree\n")
        self.assertEqual(harness.snippet(["a.py", 2, 99]), "--- a.py (lines 2-3 of 3) ---\ntwo\nthree")
        self.assertEqual(harness.label(["a.py", 1, None]), "a.py")
        self.assertEqual(harness.label(["a.py", 2, 9]), "a.py:2-9")


class Search(InTempDir):
    def test_identifier_file_ranks_best_and_skipped_dirs_are_not_read(self):
        write("src/auth.py", "import os\n\ndef check_token(token):\n    return token\n")
        write("src/other.py", "# token handling notes\n")
        write("node_modules/x.js", "function check_token() {}\n")
        write(".git/y", "check_token\n")
        found = harness.search("what does check_token do with the token?", 10_000)
        self.assertEqual(found[-1], ["src/auth.py", 1, 4])
        self.assertEqual({entry[0] for entry in found}, {"src/auth.py", "src/other.py"})

    def test_too_common_word_is_dropped(self):
        write("big.py", "widget = 1\n" * (harness.COMMON + 1))
        write("tree.py", "def render_widget_tree():\n    pass\n")
        self.assertEqual(harness.search("widget", 10_000), [])
        self.assertEqual(harness.search("widget render_widget_tree", 10_000), [["tree.py", 1, 2]])

    def test_nearby_hits_merge_and_budget_limits(self):
        write("a.py", "\n".join(f"line{i}" for i in range(1, 201)) + "\n")
        self.assertEqual(harness.search("line50 line60 line150", 100_000),
                         [["a.py", 140, 170], ["a.py", 40, 80]])
        self.assertEqual(harness.search("line50 line60 line150", 50), [["a.py", 40, 80]])

    def test_no_usable_words(self):
        self.assertEqual(harness.search("why does it not work?", 10_000), [])


class Project(InTempDir):
    def test_map_lists_text_files(self):
        write("README.md", "# Demo\n")
        write("src/a.py", "a\n")
        write("node_modules/x.js", "x\n")
        write("logo.bin", "a\0b")
        name = os.path.basename(os.getcwd())
        self.assertEqual(harness.project_map(1000), f"Project folder: {name}\nFiles:\nREADME.md\nsrc/a.py")

    def test_map_collapses_deep_directories_to_fit(self):
        for i in range(30):
            write(f"src/ui/widget_{i:02}.cpp", "x\n")
        write("src/main.cpp", "x\n")
        write("top.txt", "x\n")
        self.assertTrue(harness.project_map(60).endswith("Files:\ntop.txt\nsrc/main.cpp\nsrc/ui/ (30 files)"))
        self.assertTrue(harness.project_map(30).endswith("Files:\ntop.txt\nsrc/ (31 files)"))

    def test_map_collapses_one_folder_at_a_time_largest_first(self):
        for i in range(30):
            write(f"app/big/file_{i:02}.py", "x\n")
        write("app/small/one.py", "x\n")
        write("app/small/two.py", "x\n")
        # The large folder is collapsed; the small one beside it, at the same depth, still fits whole.
        self.assertTrue(harness.project_map(80).endswith(
            "Files:\napp/big/ (30 files)\napp/small/one.py\napp/small/two.py"))

    def test_search_and_requests_use_the_file_list_they_are_given(self):
        write("a.py", "zzz_marker = 1\n")
        self.assertEqual(harness.search("where is zzz_marker", 10_000, []), [])
        self.assertEqual(harness.search("where is zzz_marker", 10_000, ["a.py"]), [["a.py", 1, 1]])
        self.assertEqual(harness.requested("NEED: a.py", []), [])
        self.assertEqual(harness.requested("NEED: a.py", ["a.py"]), [["a.py", 1, None]])

    def test_whether_a_file_is_text_is_remembered_until_it_changes(self):
        write("a.dat", "text\n")
        self.assertTrue(harness.is_text("a.dat"))
        with mock.patch("builtins.open", side_effect=AssertionError("opened again")):
            self.assertTrue(harness.is_text("a.dat"))  # unchanged: not opened again
        write("a.dat", "bin\0ary\n")
        self.assertFalse(harness.is_text("a.dat"))

    @unittest.skipUnless(shutil.which("git"), "git is not installed")
    def test_in_a_git_work_tree_ignored_files_are_left_out(self):
        subprocess.run(["git", "init", "-q"], check=True)
        write(".gitignore", "secret.txt\n")
        for path in ("secret.txt", "kept.txt", "sub/b.py", "sub/a.py", "build/out.txt", ".hidden/x.txt"):
            write(path, "x\n")
        self.assertEqual(list(harness.project_files()), [".gitignore", "kept.txt", "sub/a.py", "sub/b.py"])
        with mock.patch.object(harness, "git_files", lambda: None):  # outside git: every file, as before
            self.assertIn("secret.txt", list(harness.project_files()))

    def test_heads_share_the_budget(self):
        for name in ("a.py", "b.py"):
            write(name, "\n".join(f"line{i}" for i in range(1, 501)) + "\n")
        write("small.py", "s\n")
        write("data.bin", "a\0b")
        self.assertEqual(harness.heads(["a.py", "b.py"], 3_000), [["a.py", 1, 150], ["b.py", 1, 150]])
        self.assertEqual(harness.heads(["small.py", "data.bin", "missing.py"], 3_000), [["small.py", 1, None]])

    def test_requested_files(self):
        for path in ("main.py", "src/Cart.cpp", "src/cart.h", "docs/a.md", "docs/b.md"):
            write(path, "x\n")
        self.assertEqual(harness.requested("NEED: main.py, `cart.cpp` docs/"), [
            ["main.py", 1, None], ["src/Cart.cpp", 1, None], ["docs/a.md", 1, None], ["docs/b.md", 1, None]])
        self.assertEqual(harness.requested("Let me look.\nNEED: src/cart.h:5-9\nNEED: main.py\n"),
                         [["src/cart.h", 5, 9], ["main.py", 1, None]])
        write("logo.bin", "a\0b")
        self.assertEqual(harness.requested("NEED: nothing.txt ../outside.txt logo.bin"), [])
        self.assertEqual(harness.requested("You need to pin a file."), [])
        folder = os.path.basename(os.getcwd())  # a name the prompt shows as "Project folder:"
        self.assertEqual(harness.requested(f"NEED: {folder}/main.py"), [["main.py", 1, None]])

    def test_within_drops_from_the_front(self):
        write("a.txt", "a" * 100 + "\n")
        write("b.txt", "b\n")
        entries = [["a.txt", 1, None], ["b.txt", 1, None]]
        self.assertEqual(harness.within(entries, 10_000), entries)
        self.assertEqual(harness.within(entries, 60), [["b.txt", 1, None]])
        self.assertEqual(harness.within(entries, 5), [])


class Includes(InTempDir):
    def test_follows_quoted_includes_one_level(self):
        write("src/widget.cpp", '#include "util.h"\n#include "widget.h"\n#include <vector>\n'
                                '#include "util/helpers.h"\n#include "missing.h"\n  #  include "widget.h"\n')
        write("src/widget.h", '#include "deeper.h"\n')
        write("src/util.h", "u\n")
        write("src/deeper.h", "d\n")
        write("include/util/helpers.h", "h\n")
        self.assertEqual(harness.includes([["src/widget.cpp", 5, 9]]), [
            ["src/util.h", 1, None], ["include/util/helpers.h", 1, None], ["src/widget.h", 1, None]])

    def test_pinned_files_are_not_repeated(self):
        write("a.cpp", '#include "a.h"\n')
        write("a.h", "x\n")
        self.assertEqual(harness.includes([["a.cpp", 1, None], ["a.h", 1, None]]), [])


class Edits(InTempDir):
    def test_parse_plain_fenced_and_several(self):
        answer = (
            "Change it:\n\nsrc/a.py\n<<<<<<< SEARCH\nold\nlines\n=======\nnew\n>>>>>>> REPLACE\n"
            "and\n`src/a.py`\n```python\n<<<<<<< SEARCH\nx = 1\n=======\n>>>>>>> REPLACE\n```\n"
            "new.txt\n<<<<<<< SEARCH\n=======\nhello\n>>>>>>> REPLACE\n")
        self.assertEqual(harness.parse_edits(answer), [
            ("src/a.py", "old\nlines", "new"),
            ("src/a.py", "x = 1", ""),
            ("new.txt", "", "hello")])

    def test_block_path_drops_labels_and_marks(self):
        for line in ("src/a.py", "FILE: src/a.py", "**File path:** `src/a.py`", "FILE_PATH: src\\a.py",
                     "### path = src/a.py", "`src/a.py`:"):
            self.assertEqual(harness.block_path(line), "src/a.py", line)
        for line in ("FILE: <path>", "FILE:", "``` "):
            self.assertEqual(harness.block_path(line), "", line)

    def test_block_path_drops_a_first_folder_named_like_the_project(self):
        folder = os.path.basename(os.getcwd())
        self.assertEqual(harness.block_path(f"FILE: {folder}/roman.py"), "roman.py")
        self.assertEqual(harness.block_path(f"FILE: {folder}"), folder)
        os.mkdir(folder)  # now the project has such a folder, so the path means it
        self.assertEqual(harness.block_path(f"FILE: {folder}/roman.py"), f"{folder}/roman.py")

    def test_edits_outside_the_project_are_refused(self):
        notes, ui = [], mock.Mock()
        ui.note = lambda text, kind="info": notes.append((kind, text))
        ui.confirm.return_value = True
        outside = os.path.join(os.path.dirname(os.getcwd()), "zzz_outside.py").replace("\\", "/")
        inner = os.path.abspath("inner.py").replace("\\", "/")
        new = "FILE: {}\n<<<<<<< SEARCH\n=======\nx\n>>>>>>> REPLACE\n"
        changed, problems, _ = harness.Session(ui).apply(new.format("../up.py") + new.format(outside)
                                                         + new.format(inner))
        self.assertEqual(problems, ["could not edit ../up.py: it is outside the project folder",
                                    f"could not edit {outside}: it is outside the project folder"])
        self.assertFalse(os.path.exists("../up.py") or os.path.exists(outside))
        self.assertEqual((changed, read_bytes("inner.py")), ({inner: "applied"}, b"x\n"))

    def test_edit_without_a_real_path_is_refused(self):
        notes, ui = [], mock.Mock()
        ui.note = lambda text, kind="info": notes.append((kind, text))
        answer = "FILE: <path>\n<<<<<<< SEARCH\n=======\nhello\n>>>>>>> REPLACE\n"
        with mock.patch.object(harness, "chat", lambda *_: answer), \
                mock.patch.object(harness, "count", lambda *_: 10):
            harness.Session(ui, 8192).send("zzz")
        self.assertIn(("bad", "could not apply an edit: the model did not say which file it is for"), notes)
        ui.confirm.assert_not_called()
        self.assertEqual(os.listdir("."), [])

    def test_apply_exact(self):
        self.assertEqual(harness.apply_edit("a\nb\nc", "b", "B\nB2"), "a\nB\nB2\nc")

    def test_apply_ignoring_indentation(self):
        self.assertEqual(harness.apply_edit("if x:\n    y = 1\n", "y = 1", "    y = 2"),
                         "if x:\n    y = 2\n")

    def test_apply_ignoring_indentation_indents_the_new_lines_as_found(self):
        code = "def f():\n    if x:\n        a = 1\n    return a\n"
        self.assertEqual(harness.apply_edit(code, "if x:\n    a = 1", "if x:\n    a = 10\n\n    b = 2"),
                         "def f():\n    if x:\n        a = 10\n\n        b = 2\n    return a\n")
        self.assertEqual(harness.apply_edit("a = 1\n", "    a = 1", "    a = 2\n    b = 3"), "a = 2\nb = 3\n")

    def test_a_failed_edit_says_why(self):
        self.assertEqual(harness.edit_problem("a\nb\na\n", "a"),
                         "its SEARCH text matches 2 places; add lines around it so that it matches one")
        self.assertEqual(harness.edit_problem("a\nb\n", "missing"), "its SEARCH text is not in the file")
        code = "\n".join(f"line {i}" for i in range(1, 21)) + "\ndef total(items):\n    return sum(items)\n"
        self.assertEqual(harness.edit_problem(code, "def totals(items):\n    return sum(item)"),
                         "its SEARCH text is not in the file; the nearest lines (19-23) are:\n"
                         "line 19\nline 20\ndef total(items):\n    return sum(items)\n")

    def test_apply_deletion_removes_the_lines(self):
        self.assertEqual(harness.apply_edit("a\nb\nc", "b", ""), "a\nc")

    def test_apply_unique_part_of_a_line(self):
        self.assertEqual(harness.apply_edit("x = foo(1)\n", "foo(1)", "bar(2)"), "x = bar(2)\n")

    def test_apply_rejects_ambiguous_and_missing(self):
        self.assertIsNone(harness.apply_edit("a\nb\na\n", "a", "z"))
        self.assertIsNone(harness.apply_edit("a\n  b\nb\n", "b ", "z"))
        self.assertIsNone(harness.apply_edit("a\nb\n", "missing", "z"))

    def test_edit_file_keeps_crlf(self):
        write("a.txt", "one\r\ntwo\r\n")
        with mock.patch("builtins.input", return_value="y"):
            harness.edit_file("a.txt", [("two", "2")])
        self.assertEqual(read_bytes("a.txt"), b"one\r\n2\r\n")

    def test_edit_file_keeps_lf_and_applies_blocks_in_order(self):
        write("a.txt", "one\ntwo\n")
        with mock.patch("builtins.input", return_value="y"):
            harness.edit_file("a.txt", [("one", "1"), ("1", "uno")])
        self.assertEqual(read_bytes("a.txt"), b"uno\ntwo\n")

    def test_declined_or_unmatched_edit_leaves_file_alone(self):
        write("a.txt", "one\r\ntwo\r\n")
        with mock.patch("builtins.input", return_value="n"):
            harness.edit_file("a.txt", [("two", "2")])
        with mock.patch("builtins.input", side_effect=AssertionError("must not ask")):
            harness.edit_file("a.txt", [("three", "3"), ("", "x")])
            harness.edit_file("missing.txt", [("a", "b")])
        self.assertEqual(read_bytes("a.txt"), b"one\r\ntwo\r\n")
        self.assertFalse(os.path.exists("missing.txt"))

    def test_create_file(self):
        with mock.patch("builtins.input", return_value="y"):
            harness.edit_file("new/b.txt", [("", "hello")])
        self.assertEqual(read_bytes("new/b.txt"), b"hello\n")

    def test_deletions_are_read_from_delete_lines(self):
        answer = ("I will remove them.\nDELETE: src/a.py, `tests\\b.py`\nDELETE: <path>\nDELETE: src/a.py\n"
                  "Do not DELETE: c.py\n")
        self.assertEqual(harness.deletions(answer), ["src/a.py", "tests/b.py"])
        self.assertEqual(harness.deletions("Nothing needs deleting."), [])

    def test_delete_file_asks_first_and_stays_inside_the_project(self):
        notes, ui = [], mock.Mock()
        ui.note = lambda text, kind="info": notes.append((kind, text))
        write("a.txt", "one\r\ntwo\r\n")
        write("b.txt", "keep\n")
        os.makedirs("folder")
        outside = "outside_" + os.path.basename(os.getcwd()) + ".txt"
        write(os.path.join("..", outside), "x\n")
        self.addCleanup(os.remove, os.path.join(os.path.dirname(os.getcwd()), outside))
        ui.confirm.return_value = True
        harness.delete_file("a.txt", ui)
        ui.confirm.assert_called_once_with("a.txt", "one\ntwo\n", None)
        self.assertFalse(os.path.exists("a.txt"))
        ui.confirm.return_value = False
        harness.delete_file("b.txt", ui)
        self.assertTrue(os.path.exists("b.txt"))
        ui.confirm.reset_mock()
        for path in ("missing.txt", "folder", "../" + outside):
            harness.delete_file(path, ui)
        ui.confirm.assert_not_called()
        self.assertTrue(os.path.exists(os.path.join("..", outside)))
        self.assertEqual(notes, [
            ("good", "deleted a.txt"), ("info", "kept b.txt"),
            ("bad", "could not delete missing.txt: there is no such file"),
            ("bad", "could not delete folder: there is no such file"),
            ("bad", f"could not delete ../{outside}: it is outside the project folder")])

    def test_the_plain_prompt_asks_before_deleting(self):
        write("a.txt", "one\ntwo\n")
        with mock.patch("builtins.input", return_value="n") as asked:
            harness.delete_file("a.txt")
        asked.assert_called_once_with("Delete a.txt? [y/N] ")
        self.assertIn("a.txt: 2 lines", self.printed.getvalue())
        self.assertTrue(os.path.exists("a.txt"))
        with mock.patch("builtins.input", return_value="y"):
            harness.delete_file("a.txt")
        self.assertFalse(os.path.exists("a.txt"))

    def test_the_model_can_delete_and_edit_in_one_answer(self):
        write("old.txt", "x\n")
        write("a.txt", "one\n")
        ui = mock.Mock()
        ui.confirm.return_value = True
        answer = "FILE: a.txt\n<<<<<<< SEARCH\none\n=======\n1\n>>>>>>> REPLACE\nDELETE: old.txt\n"
        with mock.patch.object(harness, "chat", lambda *_: answer), \
                mock.patch.object(harness, "count", lambda *_: 10):
            harness.Session(ui, 8192).send("zzz tidy up")
        self.assertEqual((read_bytes("a.txt"), os.path.exists("old.txt")), (b"1\n", False))
        self.assertIn("DELETE: <path>", harness.RULES)

    def test_in_auto_mode_every_change_is_made_and_shown_without_asking(self):
        write("old.txt", "a\n")
        write("gone.txt", "b\n")
        ui = mock.Mock()
        session = harness.Session(ui)
        session.approve = "auto"
        answer = ("FILE: old.txt\n<<<<<<< SEARCH\na\n=======\nA\n>>>>>>> REPLACE\n"
                  "FILE: new.txt\n<<<<<<< SEARCH\n=======\nfresh\n>>>>>>> REPLACE\n"
                  "DELETE: gone.txt\n")
        self.assertEqual(session.apply(answer),
                         ({"old.txt": "applied", "new.txt": "applied", "gone.txt": "deleted"}, [], []))
        ui.confirm.assert_not_called()
        self.assertEqual([call.args for call in ui.show.call_args_list],
                         [("old.txt", "a\n", "A\n"), ("new.txt", "", "fresh\n"), ("gone.txt", "b\n", None)])
        self.assertEqual((read_bytes("old.txt"), read_bytes("new.txt"), os.path.exists("gone.txt")),
                         (b"A\n", b"fresh\n", False))

    def test_the_plain_prompt_shows_a_change_it_makes_at_once(self):
        write("a.txt", "one\n")
        out = io.StringIO()
        with mock.patch("builtins.input", side_effect=AssertionError("asked")), contextlib.redirect_stdout(out):
            self.assertEqual(harness.edit_file("a.txt", [("one", "two")], harness.PLAIN, auto=True), (True, []))
        self.assertIn("-one\n+two", out.getvalue())
        self.assertEqual(read_bytes("a.txt"), b"two\n")

    def test_an_empty_block_for_an_existing_file_changes_nothing(self):
        write("tests/a.cpp", "int main() {}\n")
        ui = mock.Mock()
        ui.confirm.return_value = True
        empty = "FILE: tests/a.cpp\n<<<<<<< SEARCH\n=======\n>>>>>>> REPLACE\n"
        told = ("could not apply an edit to tests/a.cpp: an empty block changes nothing; "
                "to delete the file, write DELETE: tests/a.cpp")
        self.assertEqual(harness.Session(ui).apply(empty), ({}, [told], []))
        ui.confirm.assert_not_called()
        self.assertTrue(os.path.exists("tests/a.cpp"))
        # Beside a DELETE: line for the same file, only the DELETE: line deletes it.
        self.assertEqual(harness.Session(ui).apply(empty + "DELETE: tests/a.cpp\n"),
                         ({"tests/a.cpp": "deleted"}, [told], []))
        ui.confirm.assert_called_once_with("tests/a.cpp", "int main() {}\n", None)


class Undo(InTempDir):
    def session(self, *answers):
        """A session in "auto" whose model gives answers in turn; each send() takes the next one."""
        session, queue = harness.Session(mock.Mock()), list(answers)
        session.approve = "auto"
        session.model = FakeModel("one")
        session.model.chat = lambda *_, **__: queue.pop(0)
        return session

    def test_undo_puts_back_edited_created_and_deleted_files(self):
        write("a.txt", "one\r\n")
        write("gone.txt", "bye\n")
        session = self.session("FILE: a.txt\n<<<<<<< SEARCH\none\n=======\n1\n>>>>>>> REPLACE\n"
                               "FILE: new/deep/b.txt\n<<<<<<< SEARCH\n=======\nfresh\n>>>>>>> REPLACE\n"
                               "DELETE: gone.txt\n", "ok")
        session.send("zzz change")
        self.assertEqual((read_bytes("a.txt"), os.path.exists("gone.txt")), (b"1\r\n", False))
        session.send("zzz a question that changes nothing")  # undo still covers the message before
        self.assertEqual(session.undo(), ["put back a.txt", "removed new/deep/b.txt, which was new",
                                          "restored gone.txt, which was deleted"])
        self.assertEqual((read_bytes("a.txt"), read_bytes("gone.txt"), os.path.exists("new")),
                         (b"one\r\n", b"bye\n", False))
        self.assertEqual(session.untold, ["the user undid your changes to: a.txt, new/deep/b.txt, gone.txt"])
        self.assertEqual(session.undo(), ["nothing to undo"])

    def test_undo_keeps_a_file_changed_since(self):
        write("a.txt", "one\n")
        session = self.session("FILE: a.txt\n<<<<<<< SEARCH\none\n=======\n1\n>>>>>>> REPLACE\n")
        session.send("zzz change")
        write("a.txt", "mine\n")
        self.assertEqual(session.undo(), ["kept a.txt: it has changed since"])
        self.assertEqual((read_bytes("a.txt"), session.untold), (b"mine\n", []))

    def test_clear_forgets_what_undo_would_put_back(self):
        write("a.txt", "one\n")
        session = self.session("FILE: a.txt\n<<<<<<< SEARCH\none\n=======\n1\n>>>>>>> REPLACE\n")
        session.send("zzz change")
        session.clear()
        self.assertEqual(session.undo(), ["nothing to undo"])


def pdf_bytes(*pages):
    """A PDF whose pages hold these texts; "" makes a page without text, like a scan."""
    import fitz
    doc = fitz.open()
    for text in pages:
        page = doc.new_page(width=600, height=800)
        if text:
            page.insert_text((50, 100), text)
        else:
            page.draw_rect(fitz.Rect(50, 50, 300, 300), color=(0, 0, 0), fill=(0.5, 0.5, 0.5))
    return doc.tobytes()


def docx_bytes(*paragraphs):
    import zipfile
    body = "".join(f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>" for text in paragraphs)
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("word/document.xml", '<w:document xmlns:w="http://schemas.openxmlformats.org/'
                                              f'wordprocessingml/2006/main"><w:body>{body}</w:body></w:document>')
    return data.getvalue()


def png_bytes(width=20, height=10):
    import fitz
    image = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, width, height), False)
    image.clear_with(120)
    return image.tobytes("png")


def write_bytes(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


class Documents(InTempDir):
    def test_pdf_and_word_files_are_listed_and_shown_as_their_text(self):
        write_bytes("spec.pdf", pdf_bytes("first page", "second page"))
        write_bytes("notes.docx", docx_bytes("one", "two"))
        write_bytes("shot.png", png_bytes())
        write_bytes("data.bin", b"a\0b")
        write("a.py", "x\n")
        self.assertEqual(harness.readable_files(), ["a.py", "notes.docx", "shot.png", "spec.pdf"])
        self.assertEqual(harness.snippet(["notes.docx", 1, None]), "--- notes.docx (lines 1-2 of 2) ---\none\ntwo")
        self.assertEqual(harness.snippet(["spec.pdf", 1, 2]), "--- spec.pdf (lines 1-2 of 5) ---\n[page 1]\nfirst page")
        with mock.patch("documents.read_pdf", side_effect=AssertionError("read again")):
            harness.snippet(["spec.pdf", 1, None])  # unchanged: remembered
        self.assertEqual(harness.includes([["spec.pdf", 1, None]]), [])
        self.assertEqual(harness.requested("NEED: spec.pdf"), [["spec.pdf", 1, None]])

    def test_a_scanned_pdf_sends_its_pages_without_text_as_images(self):
        write_bytes("scan.pdf", pdf_bytes("", "text"))
        text, images = harness.contents("scan.pdf")
        self.assertEqual(len(images), 1)
        self.assertIn("[1 page(s) without text, such as scans, are sent as images]", text)
        self.assertEqual(harness.images_of(["scan.pdf", 1, None]), images)

    def test_a_file_that_cannot_be_read_says_so(self):
        write_bytes("broken.pdf", b"not a pdf")
        self.assertIn("[this file could not be read:", harness.snippet(["broken.pdf", 1, None]))

    def test_paths_with_spaces_are_pinned_in_quotes(self):
        write("my docs/a b.txt", "x\n")
        write("c.txt", "y\n")
        self.assertEqual(harness.parse_pins('see @"my docs/a b.txt" and @c.txt.'),
                         [["my docs/a b.txt", 1, None], ["c.txt", 1, None]])

    def test_an_image_goes_with_the_message_to_a_model_that_can_see(self):
        write_bytes("shot.png", png_bytes())
        model, notes, ui = FakeModel("eye"), [], mock.Mock()
        model.vision = True
        ui.note = lambda text, kind="info": notes.append((kind, text))
        session = harness.Session(ui)
        session.use(model)
        session.send("zzz what is in @shot.png")
        first = model.seen[0][1]
        self.assertIn("--- shot.png (an image of ", first["content"])
        self.assertEqual(len(first["images"]), 1)
        session.send("zzz and now?")  # the image is in the conversation already: not sent again
        self.assertNotIn("images", model.seen[1][-1])
        self.assertEqual([kind for kind, _ in notes if kind == "warn"], [])

    def test_an_image_is_left_out_for_a_model_that_cannot_see(self):
        write_bytes("shot.png", png_bytes())
        model, notes, ui = FakeModel("blind"), [], mock.Mock()
        ui.note = lambda text, kind="info": notes.append((kind, text))
        session = harness.Session(ui)
        session.use(model)
        session.send("zzz what is in @shot.png")
        self.assertNotIn("images", model.seen[0][1])
        self.assertNotIn("shot.png (an image", model.seen[0][1]["content"])
        self.assertIn(("warn", "left out shot.png: fake · blind cannot see images"), notes)

    def test_ollama_reading_more_than_was_meant_gives_a_warning(self):
        class Response:
            def __init__(self, prompt_n):
                self.lines = [json.dumps({"message": {"content": "ok"}}).encode(),
                              json.dumps({"done": True, "prompt_eval_count": prompt_n}).encode()]

            def __iter__(self):
                return iter(self.lines)

            def close(self):
                pass

        model, ui = harness.Ollama("big"), mock.Mock()
        model.size = 8192  # 8,192 less max_out's 2,000 are meant for the prompt
        for prompt_n in (6192, 6193):
            with mock.patch.object(harness, "post", lambda *_: Response(prompt_n)):
                self.assertEqual(model.chat([{"role": "user", "content": "x" * 100}], ui), "ok")
        ui.note.assert_called_once_with("Ollama read 6,193 prompt tokens, more than the 6,192 meant for it; "
                                        "it may have cut the start of the prompt", "warn")

    def test_ollama_counts_an_image_as_many_tokens(self):
        model = harness.Ollama("eye")
        messages = [{"role": "user", "content": "x" * 30, "images": ["a", "b"]}]
        self.assertEqual(model.count(messages), 10 + 2 * harness.IMAGE_TOKENS)


class Commands(InTempDir):
    def test_run_lines_the_allowlist_and_shell_syntax(self):
        answer = "Let me test.\nRUN: `python -m pytest -q`\nRUN:\nRUN: ctest --test-dir build\n"
        self.assertEqual(harness.commands(answer), ["python -m pytest -q", "ctest --test-dir build"])
        with mock.patch.object(harness, "RUN", ["python -m pytest", "cmake --build", ""]):
            self.assertTrue(harness.listed("python -m pytest -q tests/test_a.py"))
            self.assertTrue(harness.listed("python  -m   pytest"))
            self.assertFalse(harness.listed("python -m pytest_cov"))  # word by word, not letter by letter
            self.assertFalse(harness.listed("python -c print(1)"))
            self.assertFalse(harness.listed("cmake"))
        self.assertTrue(harness.SHELL.search("python -m pytest && del x"))
        self.assertTrue(harness.SHELL.search("python -m pytest > out.txt"))
        self.assertTrue(harness.SHELL.search("build.cmd & echo x"))  # a batch file runs in cmd.exe
        self.assertTrue(harness.SHELL.search("build.cmd ^& echo x"))
        self.assertTrue(harness.SHELL.search("build.cmd %PATH%"))
        self.assertFalse(harness.SHELL.search('python -m pytest -k "a and b"'))
        self.assertIn("These commands are allowed (arguments may follow them): python -m pytest, python -m "
                      "unittest.", harness.RULES)

    def test_a_command_runs_in_the_project_folder_without_a_shell(self):
        write("here.txt", "x\n")
        code, _, output = harness.run_command('python -c "import os; print(sorted(os.listdir()))"')
        self.assertEqual((code, output.strip()), (0, "['here.txt']"))
        code, _, output = harness.run_command('python -c "import sys; sys.stderr.write(\'err\'); sys.exit(3)"')
        self.assertEqual((code, output.strip()), (3, "err"))  # the error output too
        with self.assertRaises(OSError):
            harness.run_command("no_such_program_zzz --version")

    def test_a_command_that_takes_too_long_is_stopped(self):
        with mock.patch.object(harness, "RUN_TIMEOUT", 0.5):
            code, seconds, _ = harness.run_command('python -c "import time; time.sleep(30)"')
        self.assertIsNone(code)
        self.assertLess(seconds, 10)

    def test_cancel_stops_a_running_command(self):
        session, result = harness.Session(mock.Mock()), []
        thread = threading.Thread(target=lambda: result.append(
            harness.run_command('python -c "import time; time.sleep(30)"', session)))
        thread.start()
        deadline = time.time() + 10
        while session.process is None and time.time() < deadline:
            time.sleep(0.01)
        started = time.time()
        session.cancel()
        thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertLess(time.time() - started, 5)
        self.assertNotEqual(result[0][0], 0)
        self.assertIsNone(session.process)

    def test_clip_keeps_the_start_and_the_end(self):
        text = "\n".join(f"line {i}" for i in range(1, 201))
        clipped = harness.clip(text, 200)
        self.assertTrue(clipped.startswith("line 1\n"))
        self.assertTrue(clipped.endswith("\nline 200"))
        self.assertIn("lines left out", clipped)
        self.assertLess(len(clipped), 240)
        self.assertEqual(harness.clip("short", 200), "short")

    def send(self, answers, approve, allowed=True):
        """Send one message to a session whose model gives answers in turn; return the prompts and the ui."""
        prompts, ui = [], mock.Mock()
        ui.allow.return_value = allowed

        def chat(messages, *_):
            prompts.append(messages[-1]["content"])
            return answers.pop(0)

        session = harness.Session(ui)
        session.approve = approve
        with mock.patch.object(harness, "chat", chat), mock.patch.object(harness, "count", lambda *_: 10), \
                mock.patch.object(harness, "RUN", ["python -c"]):
            session.send("zzz")
        return prompts, ui

    def test_the_edits_come_first_and_the_output_goes_back_to_the_model(self):
        create = "FILE: a.py\n<<<<<<< SEARCH\n=======\nx = 1\n>>>>>>> REPLACE\n"
        prompts, ui = self.send([create + "RUN: python -c \"print(open('a.py').read())\"\n", "done"], "auto")
        self.assertEqual(len(prompts), 2)
        self.assertIn("applied a.py\nran python -c \"print(open('a.py').read())\": exit 0 in ", prompts[1])
        self.assertIn("x = 1", prompts[1].split("ran python")[1])  # it ran after the file was made
        ui.allow.assert_not_called()  # an allowed command in auto
        ui.ran.assert_called_once()

    def test_in_ask_mode_a_command_waits_and_refusing_ends_the_message(self):
        prompts, ui = self.send(['RUN: python -c "print(1)"\n', "never"], "ask", allowed=False)
        ui.allow.assert_called_once_with('python -c "print(1)"')
        self.assertEqual(len(prompts), 1)
        ui.ran.assert_not_called()

    def test_a_command_not_allowed_waits_even_in_auto(self):
        prompts, ui = self.send(["RUN: pip list\n", "never"], "auto", allowed=False)
        ui.allow.assert_called_once_with("pip list")
        self.assertEqual(len(prompts), 1)

    def test_shell_syntax_is_refused_and_the_model_told(self):
        prompts, ui = self.send(['RUN: python -c "print(1)" && echo hi\n', "ok"], "auto")
        ui.allow.assert_not_called()
        self.assertIn("shell syntax", prompts[1])

    def test_an_allowed_command_with_a_single_ampersand_is_refused(self):
        prompts, ui = self.send(['RUN: python -c "print(1)" & echo hi\n', "ok"], "auto")
        ui.allow.assert_not_called()
        ui.ran.assert_not_called()
        self.assertIn("could not run", prompts[1])

    def test_two_answers_that_only_run_end_the_message(self):
        prompts, _ = self.send(['RUN: python -c "print(1)"\n', 'RUN: python -c "print(2)"\n', "never"], "auto")
        self.assertEqual(len(prompts), 2)


class Prompt(InTempDir):
    def setUp(self):
        super().setUp()
        write("a.py", "\n".join(f"line{i}" for i in range(1, 101)) + "\n")
        write("b.py", "bee\n")
        self.tokens = lambda messages: sum(len(m["content"]) for m in messages) // 4  # one token per 4 characters
        self.base = self.tokens(harness.build([], "q", [])) + harness.MARGIN  # smallest prompt

    def test_code_comes_right_after_the_system_prompt(self):
        first = harness.build([["b.py", 1, None]], "q1", [])
        self.assertEqual([m["role"] for m in first], ["system", "user"])
        self.assertEqual(first[1]["content"], "--- b.py (lines 1-1 of 1) ---\nbee\n\nq1")
        later = harness.build([["b.py", 1, None]], "q2", [("q1", "a1")])
        self.assertEqual([m["content"] for m in later[1:]],
                         ["--- b.py (lines 1-1 of 1) ---\nbee\n\nq1", "a1", "q2"])
        longer = harness.build([["b.py", 1, None]], "q3", [("q1", "a1"), ("q2", "a2")])
        self.assertEqual([m["content"] for m in longer[1:]],
                         ["--- b.py (lines 1-1 of 1) ---\nbee\n\nq1", "a1", "q2", "a2", "q3"])
        self.assertEqual(harness.build([], "q", [])[1]["content"], harness.NO_CODE + "\n\nq")
        self.assertEqual(harness.build([], "q", [], "MAP")[0]["content"], harness.RULES + "\n\nMAP")

    def test_fit_keeps_everything_when_it_fits(self):
        ctx = [["a.py", 1, None], ["b.py", 1, None]]
        messages, used, history = harness.fit(ctx, "q", [("q0", "a0")], 10_000, self.tokens)
        self.assertEqual(len(ctx), 2)
        self.assertEqual(history, [("q0", "a0")])
        self.assertEqual(used, self.tokens(messages) + harness.MARGIN)

    def test_fit_drops_least_important_first(self):
        ctx = [["a.py", 1, None], ["b.py", 1, None]]
        _, used, _ = harness.fit(ctx, "q", [], self.base + 20, self.tokens)
        self.assertEqual(ctx, [["b.py", 1, None]])
        self.assertLessEqual(used, self.base + 20)

    def test_fit_shortens_a_single_large_entry(self):
        ctx = [["a.py", 1, None]]
        _, used, _ = harness.fit(ctx, "q", [], self.base + 60, self.tokens)
        self.assertEqual(ctx[0][:2], ["a.py", 1])
        self.assertTrue(1 <= ctx[0][2] < 100)
        self.assertLessEqual(used, self.base + 60)

    def test_fit_drops_earlier_exchanges_last_oldest_first_then_gives_up(self):
        ctx = [["b.py", 1, None]]
        _, _, history = harness.fit(ctx, "q", [("q0", "a" * 400), ("q1", "a1")], self.base + 12, self.tokens)
        self.assertEqual((ctx, history), ([], [("q1", "a1")]))
        _, _, history = harness.fit([], "q", [("q0", "a" * 400)], self.base + 5, self.tokens)
        self.assertEqual(history, [])
        self.assertIsNone(harness.fit([], "q" * 400, [], self.base + 5, self.tokens))

    def test_recent_keeps_the_newest_exchanges_that_fit(self):
        history = [("q1", "a" * 50), ("q2", "b" * 20), ("q3", "c" * 30)]
        self.assertEqual(harness.recent(history, 1000), history)
        self.assertEqual(harness.recent(history, 60), history[1:])  # 32 + 22 fit, then 52 more do not
        self.assertEqual(harness.recent(history, 10), [("q3", "c" * 10)])  # the newest, cut to fit
        self.assertEqual(harness.recent([], 10), [])


class Rounds(InTempDir):
    """The file-request loop in main(), with the model replaced by canned answers."""

    EDIT = "FILE: {0}\n<<<<<<< SEARCH\n{1}\n=======\n{2}\n>>>>>>> REPLACE\n"

    def run_main(self, answers, n_ctx=8192, replies=(), inputs=None, during=None):
        """Send one message (or the messages and replies in inputs); return the prompt text of every model call.

        replies answer the plain prompt's questions about edits, in turn. The last
        user message of every call is kept in self.sent, and the messages of every
        call in self.prompts. during(number) runs while call number is answered.
        """
        calls, self.sent, self.prompts = [], [], []

        def chat(messages, *_):
            calls.append("\n\n".join(m["content"] for m in messages[1:]))
            self.sent.append(messages[-1]["content"])
            self.prompts.append([dict(m) for m in messages])
            if during:
                during(len(calls))
            if isinstance(answers[len(calls) - 1], Exception):
                raise answers[len(calls) - 1]
            return answers[len(calls) - 1]

        info = {"default_generation_settings": {"n_ctx": n_ctx}, "model_path": "demo.gguf"}
        with mock.patch.object(harness, "chat", chat), \
                mock.patch.object(harness, "props", lambda url: info), \
                mock.patch.object(harness, "SERVERS", ["http://localhost:8080"]), \
                mock.patch.object(harness, "count", lambda messages, url: sum(len(m["content"]) for m in messages) // 4), \
                mock.patch("builtins.input", side_effect=[*(inputs or ["zzz question", *replies]), EOFError]), \
                mock.patch.object(harness.sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="utf-8")):
            harness.main()
        return calls

    def continues(self, number):
        """Whether call number's prompt is the one before, its answer, and one new message."""
        before, now = self.prompts[number - 2], self.prompts[number - 1]
        return len(now) == len(before) + 2 and now[:len(before)] == before and now[-2]["role"] == "assistant"

    def test_requests_continue_while_the_files_fit(self):
        for name in ("a.py", "b.py"):
            write(name, f"body of {name}\n")
        calls = self.run_main(["NEED: a.py", "NEED: b.py", "done"])
        self.assertEqual(len(calls), 3)
        self.assertIn(harness.NO_CODE, calls[0])
        self.assertIn("body of a.py", calls[1])
        self.assertNotIn("body of b.py", calls[1])
        self.assertIn("body of a.py", calls[2])
        self.assertIn("body of b.py", calls[2])

    def test_requests_are_not_capped(self):
        for i in range(7):
            write(f"f{i}.py", "x\n")
        self.assertEqual(len(self.run_main([f"NEED: f{i}.py" for i in range(7)] + ["done"])), 8)

    def test_after_its_edits_the_model_goes_on_when_it_asks(self):
        write("a.py", "old a\n")
        calls = self.run_main([self.EDIT.format("a.py", "old a", "new a") + "CONTINUE", "done"], replies=["y"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(read_bytes("a.py"), b"new a\n")
        self.assertIn("zzz question", calls[1])
        self.assertTrue(self.continues(2))  # the edited file is shown as it is now, after the answer that changed it
        self.assertEqual(self.sent[1], "--- a.py (lines 1-1 of 1) ---\nnew a\n\napplied a.py\n" + harness.GO_ON)

    def test_after_an_edit_only_the_changed_parts_are_shown_again(self):
        write("a.py", "\n".join(f"line{i}" for i in range(1, 201)) + "\n")
        self.run_main([self.EDIT.format("a.py", "line100", "LINE100") + "CONTINUE",
                       self.EDIT.format("a.py", "line150", "") + "CONTINUE", "done"],
                      inputs=["zzz change @a.py", "y", "y"])
        self.assertTrue(self.continues(2) and self.continues(3))
        self.assertTrue(self.sent[1].startswith(
            "--- a.py (changed parts of its 200 lines; the rest is as shown before) ---\nlines 90-110 (were 90-110):\n"
            + "\n".join(f"line{i}" for i in range(90, 100)) + "\nLINE100\nline101\n"))
        self.assertNotIn("line50\n", self.sent[1])
        self.assertIn("lines 140-159 (were 140-160):\n", self.sent[2])  # a deleted line

    def test_a_file_found_by_search_is_shown_whole_after_an_edit_lengthens_it(self):
        write("a.py", "def price():\n    return 1\n")
        edit = self.EDIT.format("a.py", "    return 1", "    tax = 2\n    return 1 + tax")
        self.run_main([edit + "CONTINUE", "done", "ok"], inputs=["zzz what is the price", "y", "zzz again"])
        self.assertIn("--- a.py (lines 1-2 of 2) ---", self.sent[0])  # found by the search, as a line range
        self.assertIn("--- a.py (lines 1-3 of 3) ---\ndef price():\n    tax = 2\n    return 1 + tax", self.sent[1])
        self.assertNotIn("a.py", self.sent[2])  # shown whole already

    def test_a_file_edited_at_the_end_of_a_message_is_shown_whole_with_the_next(self):
        write("a.py", "def price():\n    return 1\n")
        edit = self.EDIT.format("a.py", "    return 1", "    tax = 2\n    return 1 + tax")
        self.run_main([edit, "ok"], inputs=["zzz what is the price", "y", "zzz again"])
        self.assertIn("--- a.py (lines 1-3 of 3) ---\ndef price():\n    tax = 2\n    return 1 + tax", self.sent[1])

    def test_a_small_file_that_changed_is_shown_whole(self):
        write("b.py", "one\ntwo\nthree\n")
        self.run_main([self.EDIT.format("b.py", "two", "2") + "CONTINUE", "done"], inputs=["zzz change @b.py", "y"])
        self.assertTrue(self.sent[1].startswith("--- b.py (lines 1-3 of 3) ---\none\n2\nthree\n"))

    def test_edits_beside_a_file_request_are_applied(self):
        write("a.py", "old a\n")
        write("b.py", "body of b\n")
        calls = self.run_main([self.EDIT.format("a.py", "old a", "new a") + "NEED: b.py", "done"], replies=["y"])
        self.assertEqual(read_bytes("a.py"), b"new a\n")
        self.assertEqual(len(calls), 2)
        self.assertIn("body of b", calls[1])

    def test_edits_without_continue_end_the_message(self):
        write("a.py", "old a\n")
        self.assertEqual(len(self.run_main([self.EDIT.format("a.py", "old a", "new a"), "never"], replies=["y"])), 1)
        self.assertEqual(read_bytes("a.py"), b"new a\n")

    def test_declining_an_edit_ends_the_message(self):
        write("a.py", "old a\n")
        calls = self.run_main([self.EDIT.format("a.py", "old a", "new a") + "CONTINUE", "never"], replies=["n"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(read_bytes("a.py"), b"old a\n")

    def test_the_next_message_tells_the_model_what_it_was_not_told(self):
        write("a.py", "old a\n")
        write("b.py", "old b\n")
        both = self.EDIT.format("a.py", "old a", "new a") + self.EDIT.format("b.py", "old b", "new b")
        self.run_main([both, "ok", "ok"], inputs=["zzz one", "y", "n", "zzz two", "zzz three"])
        self.assertEqual(self.sent[1], "What happened after your last answer:\napplied a.py\n"
                                       "the user declined the edit to b.py; it was not applied\n\nzzz two")
        self.assertEqual(self.sent[2], "zzz three")  # told once
        self.assertTrue(self.continues(2))

    def test_what_the_model_was_not_told_waits_for_a_call_that_reaches_it(self):
        write("a.py", "old a\n")
        edit = self.EDIT.format("a.py", "old a", "new a")
        self.run_main([edit, OSError("down"), "ok"], inputs=["zzz one", "n", "zzz two", "zzz three"])
        self.assertTrue(self.sent[1].startswith("What happened after your last answer:\nthe user declined"))
        self.assertTrue(self.sent[2].startswith("What happened after your last answer:\nthe user declined"))

    def test_a_message_whose_edits_all_landed_tells_nothing_more(self):
        write("a.py", "old a\n")
        self.run_main([self.EDIT.format("a.py", "old a", "new a"), "ok"], inputs=["zzz one", "y", "zzz two"])
        self.assertEqual(self.sent[1], "zzz two")

    def test_a_failed_edit_is_retried_once(self):
        write("a.py", "old a\n")
        failing = self.EDIT.format("a.py", "missing", "x")
        calls = self.run_main([failing, failing, "never"])
        self.assertEqual(len(calls), 2)
        self.assertNotIn("old a", calls[0])
        self.assertIn("old a", calls[1])  # the retry shows the file, so its text can be copied exactly
        self.assertIn("could not apply an edit to a.py: its SEARCH text", self.sent[1])
        self.assertEqual(read_bytes("a.py"), b"old a\n")

    def test_a_created_file_is_shown_when_the_model_goes_on(self):
        create = "FILE: new.py\n<<<<<<< SEARCH\n=======\nfresh\n>>>>>>> REPLACE\nCONTINUE\n"
        calls = self.run_main([create, "done"], replies=["y"])
        self.assertEqual(len(calls), 2)
        self.assertIn("--- new.py (lines 1-1 of 1) ---\nfresh", calls[1])

    def test_going_on_without_changes_stops_after_one_more_call(self):
        self.assertEqual(len(self.run_main(["Plan first.\nCONTINUE", "Still planning.\nCONTINUE", "never"])), 2)

    def test_asking_again_for_shown_code_stops(self):
        write("a.py", "x\n")
        self.assertEqual(len(self.run_main(["NEED: a.py", "NEED: a.py", "never"])), 2)

    def test_the_call_that_fills_the_window_is_the_last(self):
        write("big.py", "\n".join(f"line{i}" for i in range(1, 1001)) + "\n")
        write("b.py", "x\n")
        calls = self.run_main(["NEED: big.py", "NEED: b.py", "never"], n_ctx=3200)
        self.assertEqual(len(calls), 2)
        self.assertIn("--- big.py (lines 1-", calls[1])
        self.assertNotIn("line1000", calls[1])

    def test_the_next_call_continues_the_conversation(self):
        write("a.py", "body of a\n")
        calls = self.run_main(["NEED: a.py", "It is a.", "Sure."], inputs=["zzz question", "zzz more"])
        self.assertEqual(len(calls), 3)
        self.assertTrue(self.continues(2) and self.continues(3))
        self.assertEqual(self.prompts[1][-2]["content"], "NEED: a.py")
        self.assertEqual(self.sent[1], "--- a.py (lines 1-1 of 1) ---\nbody of a\n\nzzz question")  # asked again
        self.assertEqual(self.sent[2], "zzz more")  # a.py is not sent twice

    def test_files_added_or_removed_between_messages_are_reported(self):
        write("old.py", "x\n")

        def change(number):
            if number == 1:
                write("new.py", "y\n")
                os.remove("old.py")

        self.run_main(["ok", "ok"], inputs=["zzz one", "zzz two"], during=change)
        self.assertIn("old.py", self.prompts[0][0]["content"])  # the file list
        self.assertTrue(self.continues(2))
        self.assertEqual(self.sent[1], "Files added since the file list above: new.py. "
                                       "Files removed since the file list above: old.py.\n\nzzz two")

    def test_a_file_the_model_creates_is_not_reported_again(self):
        create = "FILE: new.py\n<<<<<<< SEARCH\n=======\nfresh\n>>>>>>> REPLACE\n"
        self.run_main([create, "ok"], inputs=["zzz one", "y", "zzz two"])
        self.assertEqual(self.sent[1], "zzz two")


class FakeModel:
    """Stands in for a model class: records what the session does with it."""

    def __init__(self, name, size=10_000, fail=False, answer=None):
        self.name, self.label, self.detail, self.size, self.url = name, f"fake · {name}", "1.0 GB", size, "fake://"
        self.fail, self.events, self.seen, self.answer = fail, [], [], answer or "from " + name

    def enter(self):
        if self.fail:
            raise OSError("no such model")
        self.events.append("enter")

    def leave(self):
        self.events.append("leave")

    def count(self, messages):
        return 10

    def chat(self, messages, ui=None, session=None):
        self.events.append("chat")
        self.seen.append(messages)
        return self.answer


class Models(InTempDir):
    def setUp(self):
        super().setUp()
        self.notes, self.ui = [], mock.Mock()
        self.ui.note = lambda text, kind="info": self.notes.append((kind, text))

    def test_use_switches_the_model_and_keeps_the_conversation(self):
        session = harness.Session(self.ui, 8192, "demo.gguf")
        self.assertEqual((session.size, session.model.label), (8192, "llama.cpp · demo"))
        session.ctx, session.history = [["a.py", 1, None]], [("q", "a")]
        one, two, broken = FakeModel("one", 10_000), FakeModel("two", 20_000), FakeModel("broken", fail=True)
        self.assertTrue(session.use(one))
        self.assertEqual((session.size, one.events), (10_000, ["enter"]))
        self.assertTrue(session.use(two))
        self.assertEqual((session.size, one.events, two.events), (20_000, ["enter", "leave"], ["enter"]))
        self.assertFalse(session.use(broken))
        self.assertIs(session.model, two)
        self.assertEqual(self.notes, [("info", "model: fake · one · 10,000 tokens"),
                                      ("info", "model: fake · two · 20,000 tokens"),
                                      ("bad", "could not load fake · broken: no such model")])
        self.assertEqual((session.ctx, session.history), ([["a.py", 1, None]], [("q", "a")]))
        session.clear()
        session.send("zzz hello")  # counted and answered by the model in use
        self.assertEqual((session.history, two.events[-1]), ([("zzz hello", "from two")], "chat"))

    def test_a_new_model_or_clear_builds_the_prompt_afresh(self):
        session, one, two = harness.Session(self.ui), FakeModel("one"), FakeModel("two")
        session.use(one)
        session.send("zzz a")
        session.send("zzz b")
        self.assertEqual(len(one.seen[1]), 4)  # the first call's prompt, its answer and the new message
        session.use(one)  # the same model again: its cache still holds the conversation
        session.send("zzz c")
        self.assertEqual(len(one.seen[2]), 6)
        session.use(two)
        session.send("zzz d")
        # Built afresh: the code with the first kept message, then the exchanges, then the new message.
        self.assertEqual(two.seen[0][1]["content"], harness.NO_CODE + "\n\nzzz a")
        self.assertEqual([m["content"] for m in two.seen[0][2:]],
                         ["from one", "zzz b", "from one", "zzz c", "from one", "zzz d"])
        session.clear()
        session.send("zzz e")
        self.assertEqual(len(two.seen[1]), 2)

    def test_an_answer_cut_off_while_thinking_is_asked_again_without_thinking(self):
        asked = []

        class Thinker(FakeModel):
            def chat(self, messages, ui=None, session=None, think=None):
                asked.append((think, len(messages)))
                session.thought = session.cut = think is None  # it thinks until the answer's room runs out
                return "" if think is None else "the answer"

        session = harness.Session(self.ui)
        session.use(Thinker("one", 65536))
        session.send("zzz")
        self.assertEqual(asked, [(None, 2), (False, 2)])  # the same prompt, without thinking
        self.assertEqual(session.history, [("zzz", "the answer")])
        self.assertEqual(session.log[-1]["content"], "the answer")
        self.assertIn(("warn", "the answer reached max_out (2,000 tokens) while the model was thinking; "
                               "asking again without thinking"), self.notes)

    def test_an_answer_cut_off_without_thinking_is_kept_with_a_warning(self):
        asked = []

        class Talker(FakeModel):
            def chat(self, messages, ui=None, session=None, think=None):
                asked.append(think)
                session.cut = True
                return "a long answer"

        session = harness.Session(self.ui)
        session.use(Talker("one", 65536))
        session.send("zzz")
        self.assertEqual((asked, session.history), ([None], [("zzz", "a long answer")]))
        self.assertIn(("warn", "the answer reached max_out (2,000 tokens) and was cut off"), self.notes)

    def test_thinking_past_think_max_ends_the_stream(self):
        class Response:
            fp, closed = None, False

            def __iter__(self):
                chunk = lambda delta: f"data: {json.dumps({'choices': [{'delta': delta}]})}\n".encode()
                return iter([chunk({"reasoning_content": f"t{i} "}) for i in range(5)]
                            + [chunk({"content": "answer"}), b"data: [DONE]\n"])

            def close(self):
                self.closed = True

        for cap, thoughts, answer in ((3, 3, ""), (None, 5, "answer")):
            session, response = harness.Session(self.ui), Response()
            with mock.patch.object(harness, "THINK_MAX", cap):
                self.assertEqual(harness.stream(response, harness.llama_line, self.ui, session), answer)
            self.assertEqual((self.ui.thinking.call_count, session.overthought, response.closed),
                             (thoughts, cap is not None, True))
            self.ui.thinking.reset_mock()

    def test_thinking_past_think_max_is_asked_again_without_thinking(self):
        asked = []

        class Thinker(FakeModel):
            def chat(self, messages, ui=None, session=None, think=None):
                asked.append(think)
                session.thought = session.overthought = think is None or self.name == "stubborn"
                return "" if session.overthought else "the answer"

        for name, history in (("one", [("zzz", "the answer")]), ("stubborn", [])):
            session, asked[:] = harness.Session(self.ui), []
            session.use(Thinker(name, 65536))
            with mock.patch.object(harness, "THINK_MAX", 3000):
                session.send("zzz")
            self.assertEqual((asked, session.history), ([None, False], history))
        self.assertIn(("warn", "the model thought for 3,000 tokens (think_max); asking again without thinking"),
                      self.notes)
        self.assertEqual(self.notes[-1], ("bad", "the model thought for 3,000 tokens again and gave no answer; "
                                                 "raise think_max in settings.json"))

    def test_an_unreadable_stream_ends_the_message_with_a_note(self):
        class Garbled(FakeModel):
            def chat(self, messages, ui=None, session=None, think=None):
                raise json.JSONDecodeError("Expecting value", "data: {oops", 6)

        session = harness.Session(self.ui)
        session.use(Garbled("one"))
        session.send("zzz")
        self.assertEqual(session.log, [])
        self.assertIn(("bad", "the server sent an answer that could not be read: Expecting value: line 1 column 7 "
                              "(char 6)"), self.notes)

    def test_a_conversation_that_no_longer_fits_is_built_afresh(self):
        model = FakeModel("one")
        session = harness.Session(self.ui)
        session.use(model)
        session.send("zzz a")
        session.log[-1]["content"] += "x" * 100_000  # as if the conversation had grown past the window
        with mock.patch.object(model, "count", lambda messages: sum(len(m["content"]) for m in messages) // 4):
            session.send("zzz b")
        self.assertEqual([m["content"] for m in model.seen[1][2:]], ["from one", "zzz b"])
        self.assertEqual(model.seen[1][1]["content"], harness.NO_CODE + "\n\nzzz a")
        self.assertTrue(any(text.startswith("rebuilt: no code | ") for _, text in self.notes))

    def test_a_larger_window_keeps_more_of_the_conversation(self):
        for size, exchanges in ((8192, 1), (65536, 3)):
            model = FakeModel("one", size, answer="x" * 2000)
            session = harness.Session(self.ui)
            session.use(model)
            for question in ("zzz one", "zzz two", "zzz three", "zzz four"):
                session.log = []  # as if the conversation no longer fitted, so the prompt is built afresh
                session.send(question)
            self.assertEqual(len(model.seen[-1]), 2 + 2 * exchanges, size)  # system, the exchanges, the question

    def test_pick_by_number_name_or_unique_part(self):
        found = [harness.Llama(8192, "Bonsai-27B.gguf"), harness.Ollama("bonsai-2:27b"), harness.Ollama("qwen3:8b")]
        self.assertIs(harness.pick(found, "2"), found[1])
        self.assertIs(harness.pick(found, "QWEN"), found[2])
        self.assertIs(harness.pick(found, "bonsai-2:27b"), found[1])  # a full name wins over partial matches
        self.assertIs(harness.pick(found, "llama.cpp"), found[0])
        for wanted in ("bonsai", "llama", "0", "9", "nothing"):  # "llama" is also part of "ollama"
            self.assertIsNone(harness.pick(found, wanted), wanted)

    def test_found_code_is_capped_whatever_the_window(self):
        budgets = []
        with mock.patch.object(harness, "search", lambda question, budget, files: budgets.append(budget) or []), \
                mock.patch.object(harness, "count", lambda *_: 10), \
                mock.patch.object(harness, "chat", lambda *_: "ok"):
            for size in (4000, 8192, 32768):
                harness.Session(self.ui, size).send("zzz")
        self.assertLess(budgets[0], harness.SEARCH_TOKENS * 3)  # a small window is the limit itself
        self.assertEqual(budgets[1:], [harness.SEARCH_TOKENS * 3] * 2)

    def test_plain_prompt_lists_and_switches_models(self):
        two, out = FakeModel("two"), Out()
        info = {"default_generation_settings": {"n_ctx": 8192}, "model_path": "demo.gguf"}
        with mock.patch.object(harness, "models", lambda: [harness.Llama(8192, "demo.gguf"), two]), \
                mock.patch.object(harness, "props", lambda url: info), \
                mock.patch("builtins.input", side_effect=["/model", "/model nope", "/model 2", "zzz hi", EOFError]), \
                mock.patch.object(harness.sys, "stdout", out):
            harness.main()
        printed = out.getvalue()
        self.assertIn("1. llama.cpp · demo · 8,192 tokens  (current)\n2. fake · two · 1.0 GB\n", printed)
        self.assertIn("[no single model matches 'nope']", printed)
        self.assertIn("[model: fake · two · 10,000 tokens]", printed)
        self.assertEqual(two.events, ["enter", "chat"])

    def test_plain_prompt_starts_on_the_model_the_settings_name(self):
        two, out = FakeModel("two"), Out()
        with mock.patch.object(harness, "MODEL", "two"), mock.patch.object(harness, "models", lambda: [two]), \
                mock.patch.object(harness, "props", lambda url: self.fail("llama-server was asked")), \
                mock.patch("builtins.input", side_effect=EOFError), \
                mock.patch.object(harness.sys, "stdout", out):
            harness.main()
        self.assertIn("[model: fake · two · 10,000 tokens]\nfake:// | 10000 tokens", out.getvalue())

    def test_without_a_model_in_the_settings_the_first_listed_is_started(self):
        two, out = FakeModel("two"), Out()
        with mock.patch.object(harness, "models", lambda: [two, FakeModel("three")]), \
                mock.patch("builtins.input", side_effect=EOFError), \
                mock.patch.object(harness.sys, "stdout", out):
            harness.main()
        self.assertIn("[model: fake · two · 10,000 tokens]", out.getvalue())
        self.assertEqual(two.events, ["enter"])

    def test_startup_names_the_model_to_start_with(self):
        found = [FakeModel("two")]
        with mock.patch.object(harness, "models", lambda: found):
            self.assertEqual(harness.startup(), "two")
            with mock.patch.object(harness, "MODEL", "three"):
                self.assertEqual(harness.startup(), "three")
            found.clear()
            self.assertEqual(harness.startup(), "")
        self.assertIn("[no model server answers]", self.printed.getvalue())


class Settings(InTempDir):
    def setUp(self):
        super().setUp()
        for name in ("SERVERS", "MODEL", "CONTEXT", "THINK", "THINK_MAX", "MAX_OUT", "OPTIONS", "APPROVE", "RUN",
                     "RULES"):
            patch = mock.patch.object(harness, name, getattr(harness, name))  # put back what load_settings changes
            patch.start()
            self.addCleanup(patch.stop)

    def test_a_missing_file_is_written_with_the_defaults(self):
        harness.load_settings("settings.json")
        with open("settings.json", encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"servers": ["http://localhost:11434", "http://localhost:8080"],
                                            "model": "", "context": None, "think": False, "think_max": None,
                                            "max_out": 2000,
                                            "options": {"temperature": 0.2}, "approve": "ask",
                                            "run": ["python -m pytest", "python -m unittest"]})
        self.assertIn("[wrote the default settings to settings.json]", self.printed.getvalue())

    def test_the_file_replaces_the_defaults(self):
        write("settings.json", json.dumps({"servers": "http://192.168.1.50:11434/", "think": "high",
                                           "think_max": 4000, "context": 16384, "options": {"top_k": 20},
                                           "tempreature": 1, "approve": "auto", "run": ["ctest --test-dir build"]}))
        harness.load_settings("settings.json")
        self.assertEqual((harness.SERVERS, harness.MODEL, harness.THINK, harness.CONTEXT, harness.MAX_OUT,
                          harness.OPTIONS, harness.APPROVE, harness.RUN),
                         (["http://192.168.1.50:11434"], "", "high", 16384, 2000, {"top_k": 20}, "auto",
                          ["ctest --test-dir build"]))
        self.assertEqual((harness.THINK_MAX, harness.Session().think_max), (4000, 4000))
        self.assertEqual(harness.Session().approve, "auto")
        self.assertIn("These commands are allowed (arguments may follow them): ctest --test-dir build.", harness.RULES)
        self.assertIn("unknown settings ignored: tempreature", self.printed.getvalue())

    def test_a_broken_file_exits_with_the_error(self):
        write("settings.json", '{"think": true,}')
        with self.assertRaises(SystemExit) as caught:
            harness.load_settings("settings.json")
        self.assertRegex(caught.exception.code, r"^settings\.json: .*line 1 column \d+")
        write("settings.json", '["http://localhost:11434"]')
        with self.assertRaises(SystemExit):
            harness.load_settings("settings.json")


class Terminal(InTempDir):
    def test_single_and_multi_line_message(self):
        with mock.patch("builtins.input", return_value="  hi  "):
            self.assertEqual(harness.read_message(), "hi")
        with mock.patch("builtins.input", side_effect=['"""', "a", " b", '"""']):
            self.assertEqual(harness.read_message(), "a\n b")


class FakeServer(http.server.BaseHTTPRequestHandler):
    """Answers like llama-server and, under /api/, like Ollama."""

    chat_request = None
    release = threading.Event()  # ends a stalled stream
    ps = show = None  # what Ollama's /api/ps and /api/show answer; set by Server.setUp
    ollama_requests = []  # the bodies posted to /api/chat

    def log_message(self, *args):
        pass

    def stall(self, first):
        """Send one piece of an answer, then keep the stream open without sending more."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(first)
        self.wfile.flush()
        FakeServer.release.wait(10)

    def ollama_chat(self, body):
        FakeServer.ollama_requests.append(body)
        if not body["messages"]:  # a request to load or unload the model
            self.reply(b'{"done": true}')
        elif body["messages"][-1]["content"] == "stall":
            self.stall(b'{"message": {"content": "Hel"}}\n')
        elif body["messages"][-1]["content"] == "fail":
            self.reply(b'{"error": "model crashed"}\n')
        elif body["messages"][-1]["content"] == "long":  # thinking that runs into the length limit
            lines = [{"message": {"content": "", "thinking": "hmm"}},
                     {"message": {"content": ""}, "done": True, "done_reason": "length", "prompt_eval_count": 7}]
            self.reply("".join(json.dumps(line) + "\n" for line in lines).encode(), "application/x-ndjson")
        else:
            lines = [{"message": {"content": "", "thinking": "let me think"}},
                     {"message": {"content": "Hel"}}, {"message": {"content": "lo"}},
                     {"message": {"content": ""}, "done": True, "prompt_eval_count": 7, "prompt_eval_duration": 30_000_000,
                      "eval_count": 2, "eval_duration": 100_000_000}]
            self.reply("".join(json.dumps(line) + "\n" for line in lines).encode(), "application/x-ndjson")

    def reply(self, body, kind="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            answer = {"models": [{"name": "small:1b", "size": 3_200_000_000}, {"name": "big:27b", "size": 8_300_000_000}]}
        elif self.path == "/api/ps":
            answer = {"models": FakeServer.ps}
        else:
            answer = {"default_generation_settings": {"n_ctx": 4096}, "model_path": "/models/demo.gguf"}
        self.reply(json.dumps(answer).encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            self.reply(json.dumps({"tokens": body["content"].split()}).encode())
            return
        if self.path == "/api/show":
            self.reply(json.dumps(FakeServer.show).encode())
            return
        if self.path == "/api/chat":
            self.ollama_chat(body)
            return
        if body["messages"][-1]["content"] == "stall":
            self.stall(b'data: {"choices": [{"delta": {"content": "Hel"}}]}\n\n')
            return
        if body["messages"][-1]["content"] == "long":  # thinking that runs into the length limit
            FakeServer.chat_request = body
            chunks = [{"choices": [{"delta": {"reasoning_content": "hmm"}}]},
                      {"choices": [{"delta": {}, "finish_reason": "length"}]}]
            stream = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            self.reply(stream.encode(), "text/event-stream")
            return
        FakeServer.chat_request = body
        chunks = [{"choices": [{"delta": {"role": "assistant", "content": None}}]},
                  {"choices": [{"delta": {"reasoning_content": "thinking"}}]},
                  {"choices": [{"delta": {"content": "Hel"}}]},
                  {"choices": [{"delta": {"content": "lo"}}]},
                  {"choices": [], "timings": {"predicted_n": 2}}]
        stream = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        self.reply(stream.encode(), "text/event-stream")


class FakeOllama(FakeServer):
    """Like FakeServer, but answers llama-server's /props with an error, as Ollama does."""

    def do_GET(self):
        if self.path == "/props":
            self.send_error(404)
        else:
            super().do_GET()


class Server(InTempDir):
    def serve(self, handler=FakeServer):
        """Start a fake server and return its URL."""
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}"

    def setUp(self):
        super().setUp()
        self.url = self.serve()
        FakeServer.ps = [{"name": "other:1b", "context_length": 999}, {"name": "small:1b", "context_length": 32768}]
        FakeServer.show = {"capabilities": ["completion", "thinking"], "parameters": "temperature 1"}
        FakeServer.ollama_requests = []

    def test_models_come_server_by_server_each_asked_which_it_is(self):
        ollama, dead = self.serve(FakeOllama), "http://127.0.0.1:1"
        with mock.patch.object(harness, "SERVERS", [ollama, dead, self.url]):  # the server that is not running is left out
            self.assertEqual([(model.label, model.detail, model.url) for model in harness.models()],
                             [("ollama · small:1b", "3.2 GB", ollama), ("ollama · big:27b", "8.3 GB", ollama),
                              ("llama.cpp · demo", "4,096 tokens", self.url)])
        with mock.patch.object(harness, "SERVERS", [self.url, ollama]):
            self.assertEqual([model.label for model in harness.models()],
                             ["llama.cpp · demo", "ollama · small:1b", "ollama · big:27b"])
        with mock.patch.object(harness, "SERVERS", [dead]):
            self.assertEqual(harness.models(), [])

    def test_models_of_several_servers_of_each_kind(self):
        ollama, other, second = self.serve(FakeOllama), self.serve(FakeOllama), self.serve()
        with mock.patch.object(harness, "SERVERS", [ollama, other, self.url, second]):
            found = harness.models()
        self.assertEqual([model.url for model in found], [ollama] * 2 + [other] * 2 + [self.url, second])
        found[-1].enter()
        self.assertEqual(found[-1].url, second)
        self.assertEqual(harness.Ollama("qwen3:8b", url="http://192.168.1.50:11434").label,
                         "ollama@192.168.1.50 · qwen3:8b")
        self.assertEqual(harness.Llama(8192, "demo.gguf", "http://192.168.1.50:8080").label,
                         "llama.cpp@192.168.1.50 · demo")

    def test_ollama_model_loads_reports_its_size_and_unloads(self):
        model = harness.Ollama("small:1b", url=self.url)
        model.enter()
        self.assertEqual((model.size, model.thinks), (32768, True))
        options = {**harness.OPTIONS, "num_predict": harness.answer_room(4096)}  # the size before loading
        self.assertEqual(FakeServer.ollama_requests, [{"model": "small:1b", "messages": [], "options": options}])
        FakeServer.ps = [{"name": "small:1b"}]  # an Ollama that does not report the size in use
        FakeServer.show = {"capabilities": ["completion"], "parameters": "num_ctx    2048\ntemperature 1"}
        model.enter()
        self.assertEqual((model.size, model.thinks), (2048, False))
        FakeServer.show = {}
        model.enter()
        self.assertEqual(model.size, 4096)
        model.leave()
        self.assertEqual(FakeServer.ollama_requests[-1], {"model": "small:1b", "messages": [], "keep_alive": 0})

    def test_ollama_chat_streams_and_reports_timings(self):
        seen, thought, ui, model = [], [], mock.Mock(), harness.Ollama("small:1b", url=self.url)
        ui.timings, ui.thinking = seen.append, thought.append
        messages = [{"role": "user", "content": "hi"}]
        model.thinks = True
        self.assertEqual(model.chat(messages, ui), "Hello")  # the thinking text is not part of the answer
        self.assertEqual(thought, ["let me think"])
        self.assertEqual(seen, [{"cache_n": None, "prompt_n": 7, "prompt_ms": 30.0, "predicted_n": 2,
                                 "predicted_ms": 100.0}])  # Ollama does not say what was cached
        self.assertEqual(FakeServer.ollama_requests[-1],
                         {"model": "small:1b", "messages": messages, "stream": True, "think": False,
                          "options": {**harness.OPTIONS, "num_predict": harness.answer_room(model.size)}})
        with mock.patch.object(harness, "THINK", "high"):
            model.chat(messages, ui)
        self.assertEqual(FakeServer.ollama_requests[-1]["think"], "high")
        model.thinks = False
        with mock.patch.object(harness, "THINK", True):
            model.chat(messages, ui)
        self.assertNotIn("think", FakeServer.ollama_requests[-1])  # Ollama refuses it for a model that cannot think
        with self.assertRaisesRegex(OSError, "model crashed"):
            model.chat([{"role": "user", "content": "fail"}], ui)

    def test_both_servers_say_when_an_answer_was_cut_off(self):
        for model in (harness.Ollama("small:1b", url=self.url), harness.Llama(8192, url=self.url)):
            model.thinks = True
            session = harness.Session(mock.Mock())
            for content, cut in (("long", True), ("hi", False)):
                session.cut = session.thought = False
                model.chat([{"role": "user", "content": content}], mock.Mock(), session)
                # Both answers of the fake server think; only the long one is stopped by the limit.
                self.assertEqual((session.cut, session.thought), (cut, True), (model.label, content))
        harness.Ollama("small:1b", url=self.url).chat([{"role": "user", "content": "hi"}], mock.Mock())  # no session
        harness.Llama(8192, url=self.url).chat([{"role": "user", "content": "long"}], mock.Mock(), think=False)
        self.assertFalse(FakeServer.chat_request["chat_template_kwargs"]["enable_thinking"])
        model = harness.Ollama("small:1b", url=self.url)
        model.thinks = True
        with mock.patch.object(harness, "THINK", True):
            model.chat([{"role": "user", "content": "hi"}], mock.Mock(), think=False)
        self.assertIs(FakeServer.ollama_requests[-1]["think"], False)

    def test_ollama_counts_learn_the_characters_per_token_from_its_calls(self):
        model, ui = harness.Ollama("small:1b", url=self.url), mock.Mock()
        self.assertEqual(model.count([{"content": "x" * 20}, {"content": "y" * 10}]), 10)  # 3 until a call reports
        model.chat([{"role": "user", "content": "z" * 35}], ui)  # the fake server says 7 tokens: 5 a token
        self.assertAlmostEqual(model.chars_per_token, 4.5)  # less a tenth, to stay on the safe side
        self.assertEqual(model.count([{"content": "x" * 90}]), 20)
        model.chat([{"role": "user", "content": "z" * 70}], ui)  # 10 a token: no tokenizer gives that
        model.chat([{"role": "user", "content": "z" * 7, "images": ["aGk="]}], ui)  # the image's tokens are unknown
        self.assertAlmostEqual(model.chars_per_token, 4.5)

    def test_a_learned_ratio_lets_more_code_in(self):
        budgets = []
        with mock.patch.object(harness, "search", lambda question, budget, files: budgets.append(budget) or []):
            for ratio in (3, 4):
                model = FakeModel("m", 4000)
                model.chars_per_token = ratio
                session = harness.Session(mock.Mock())
                session.model = model
                session.send("zzz")
        self.assertGreater(budgets[1], budgets[0])

    def test_context_setting_reaches_ollama_and_caps_llama(self):
        model = harness.Ollama("small:1b", url=self.url)
        with mock.patch.object(harness, "CONTEXT", 16384):
            model.enter()
            model.chat([{"role": "user", "content": "hi"}], mock.Mock())
            self.assertEqual([request["options"]["num_ctx"] for request in FakeServer.ollama_requests], [16384, 16384])
            self.assertEqual(harness.Llama(8192).size, 8192)  # llama-server's window cannot be made larger
        with mock.patch.object(harness, "CONTEXT", 4096):
            self.assertEqual(harness.Llama(8192).size, 4096)

    def test_cancel_ends_a_stalled_ollama_call(self):
        FakeServer.release = threading.Event()
        self.addCleanup(FakeServer.release.set)
        session, result, model = harness.Session(), [], harness.Ollama("small:1b", url=self.url)
        call = threading.Thread(target=lambda: result.append(
            model.chat([{"role": "user", "content": "stall"}], harness.PLAIN, session)))
        call.start()
        deadline = time.time() + 5
        while "Hel" not in self.printed.getvalue() and time.time() < deadline:
            time.sleep(0.01)
        session.cancel()
        call.join(3)
        self.assertFalse(call.is_alive())
        self.assertEqual(result, ["Hel"])

    def test_props_tokenize_and_streaming(self):
        messages = [{"role": "system", "content": "one two"}, {"role": "user", "content": "three"}]
        self.assertEqual(harness.count(messages, self.url), 3)
        self.assertEqual(harness.chat(messages, self.url), "Hello")
        self.assertEqual(self.printed.getvalue(), "thinking\nHello\n")  # the thinking text is not part of the answer
        self.assertEqual(FakeServer.chat_request,
                         {"messages": messages, "stream": True, "max_tokens": harness.MAX_OUT, **harness.OPTIONS,
                          "chat_template_kwargs": {"enable_thinking": False}})
        with mock.patch.object(harness, "THINK", "low"):
            harness.chat(messages, self.url)
        self.assertEqual(FakeServer.chat_request["chat_template_kwargs"],
                         {"enable_thinking": True, "reasoning_effort": "low"})

    def test_the_answer_gets_at_most_a_quarter_of_the_window(self):
        self.assertEqual([harness.answer_room(size) for size in (4096, 8192, 65536)], [1024, 2000, 2000])
        with mock.patch.object(harness, "MAX_OUT", 16000):
            self.assertEqual([harness.answer_room(size) for size in (8192, 65536)], [2048, 16000])
            harness.Llama(8192, url=self.url).chat([{"role": "user", "content": "hi"}])
            self.assertEqual(FakeServer.chat_request["max_tokens"], 2048)
            model = harness.Ollama("small:1b", url=self.url)
            model.size = 65536
            self.assertEqual(model.options()["num_predict"], 16000)

    def test_cancel_ends_a_stalled_call(self):
        FakeServer.release = threading.Event()
        self.addCleanup(FakeServer.release.set)
        session, result = harness.Session(), []
        call = threading.Thread(target=lambda: result.append(
            harness.chat([{"role": "user", "content": "stall"}], self.url, harness.PLAIN, session)))
        call.start()
        deadline = time.time() + 5
        while "Hel" not in self.printed.getvalue() and time.time() < deadline:
            time.sleep(0.01)
        started = time.time()
        session.cancel()
        call.join(3)
        self.assertFalse(call.is_alive())
        self.assertEqual(result, ["Hel"])
        self.assertLess(time.time() - started, 1)

    def test_timings(self):
        seen = []
        ui = mock.Mock(timings=seen.append)
        harness.chat([{"role": "user", "content": "hi"}], self.url, ui)
        self.assertEqual(seen, [{"predicted_n": 2}])


if __name__ == "__main__":
    unittest.main()
