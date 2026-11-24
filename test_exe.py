"""Tests of tolken.exe, the one-file build of tolken.spec, used as it is used: from a folder of its own, in a project.

Run: set TOLKEN_EXE=dist\\tolken.exe, then python -m unittest test_exe (they are skipped without TOLKEN_EXE).
The full-screen interface is driven in a terminal of its own, which on Windows needs pywinpty
(pip install pywinpty). .github/workflows/release.yml runs them on each build, before it is published.

A fake llama-server stands in for the model. What is checked is what the build could break: that
the settings and sessions are kept next to the exe, that the full-screen interface starts (Textual, and
Pygments for the code in an answer), that PDF and Word files are read (PyMuPDF), that git and a
command run and see nothing of the exe's own run-time setup, and that a second tolken.exe started
from a command runs.
"""
import ctypes
import glob
import http.server
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from unittest import mock

EXE = os.environ.get("TOLKEN_EXE")
PYTHON = "python" if os.name == "nt" else "python3"
SECONDS = 120  # for a run of the exe; it unpacks itself at each start, which a virus scanner can make slow
ESCAPES = re.compile(r"\x1b(\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(\x07|\x1b\\)|[PX^_][^\x1b]*\x1b\\|[ -/]*[0-~])")

# A command that prints what a command run by tolken can see of tolken.exe: the _PYI_ variables, and on
# Windows the DLL folder it was given and the DLLs it would load from the folders tolken.exe unpacks to.
PROBE = r'''import ctypes, glob, os, sys, tempfile
print("PYI:", sorted(name for name in os.environ if name.upper().startswith("_PYI_")))
if os.name == "nt":
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
    folder = ctypes.create_unicode_buffer(1024)
    kernel32.GetDllDirectoryW(1024, folder)
    print("DLL folder:", repr(folder.value))
    found = []
    for unpacked in glob.glob(os.path.join(tempfile.gettempdir(), "_MEI*")):
        for path in sorted(glob.glob(os.path.join(unpacked, "*.dll"))):
            name = os.path.basename(path)
            if name.lower().startswith(("python", "vcruntime")):  # loaded already: the probe's own
                continue
            try:
                dll = ctypes.CDLL(name, winmode=0)  # Windows' own search order, as a program finds its DLLs
            except OSError:
                continue
            loaded = ctypes.create_unicode_buffer(1024)
            kernel32.GetModuleFileNameW(dll._handle, loaded, 1024)
            if os.path.basename(os.path.dirname(loaded.value)).upper().startswith("_MEI"):
                found.append(loaded.value)
    print("from tolken.exe:", found)
'''


class Model(http.server.BaseHTTPRequestHandler):
    """A llama-server whose answers are Model.answers, one per call; the calls' messages go to Model.calls."""

    answers, calls = [], []

    def log_message(self, *args):
        pass

    def reply(self, body, kind="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.reply(json.dumps({"default_generation_settings": {"n_ctx": 8192}, "model_path": "/m/demo.gguf"}).encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            self.reply(json.dumps({"tokens": body["content"].split()}).encode())
            return
        Model.calls.append(body["messages"])
        answer = Model.answers.pop(0) if Model.answers else "DONE"
        chunks = [{"choices": [{"delta": {"content": answer}}]}, {"choices": [], "timings": {"predicted_n": 9}}]
        self.reply(("".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n").encode(),
                   "text/event-stream")


def said(call):
    """All that one call to the model was told, as one text."""
    return "\n".join(message["content"] if isinstance(message["content"], str) else json.dumps(message["content"])
                     for message in call)


class Terminal:
    """A program in a terminal of its own: on Windows a pseudo console (pywinpty), elsewhere a pty.

    raw() is all it has written; shown() is that without the escape codes that place and colour the text.
    """

    def __init__(self, argv, cwd):
        self.written, self.lock = [], threading.Lock()
        if os.name == "nt":
            import winpty
            self.process = winpty.PtyProcess.spawn(argv, cwd=cwd, dimensions=(40, 120))
            read, self.write = lambda: self.process.read(65536), self.process.write
        else:
            import fcntl
            import pty
            import struct
            import termios
            master, child = pty.openpty()
            fcntl.ioctl(child, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
            self.process = subprocess.Popen(argv, cwd=cwd, stdin=child, stdout=child, stderr=child,
                                            env={**os.environ, "TERM": "xterm-256color"}, start_new_session=True)
            os.close(child)
            read = lambda: os.read(master, 65536).decode("utf-8", "replace")
            self.write = lambda text: os.write(master, text.encode())
        threading.Thread(target=self.keep, args=(read,), daemon=True).start()

    def keep(self, read):
        try:
            while text := read():
                with self.lock:
                    self.written.append(text)
        except (EOFError, OSError):
            pass

    def raw(self):
        with self.lock:
            return "".join(self.written)

    def shown(self):
        return ESCAPES.sub("", self.raw())

    def wait_for(self, text, seconds=SECONDS):
        end = time.monotonic() + seconds
        while text not in self.shown():
            if time.monotonic() > end:
                raise AssertionError(f"{text!r} was not shown; the terminal shows:\n{self.shown()[-3000:]}")
            time.sleep(0.2)

    def running(self):
        return self.process.isalive() if os.name == "nt" else self.process.poll() is None

    def stop(self):
        if self.running():
            self.process.terminate(force=True) if os.name == "nt" else self.process.kill()

    def exit_code(self, seconds=SECONDS):
        end = time.monotonic() + seconds
        while self.running():
            if time.monotonic() > end:
                raise AssertionError(f"still running; the terminal shows:\n{self.shown()[-3000:]}")
            time.sleep(0.2)
        return self.process.exitstatus if os.name == "nt" else self.process.returncode


@unittest.skipUnless(EXE, "set TOLKEN_EXE to the tolken.exe to test")
class Exe(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Model)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        Model.answers, Model.calls = [], []
        # The exe in a folder of its own, as in %USERPROFILE%\.local\bin, with the settings beside it.
        self.bin = os.path.join(self.root, "bin")
        os.mkdir(self.bin)
        self.exe = shutil.copy(EXE, self.bin)
        self.settings = {"servers": [f"http://127.0.0.1:{server.server_port}"], "model": "", "context": None,
                         "think": False, "approve": "auto", "run": [f"{PYTHON} probe.py", "tolken"], "instructions": "",
                         "suggest": False}
        with open(os.path.join(self.bin, "settings.json"), "w", encoding="utf-8") as f:
            json.dump(self.settings, f)
        # The project it is run in: a git repository, whose .gitignore leaves out ignored.txt.
        self.project = os.path.join(self.root, "project")
        os.mkdir(self.project)
        for name, text in {"main.py": "print('hi')\n", ".gitignore": "ignored.txt\n", "ignored.txt": "secret\n",
                           "probe.py": PROBE}.items():
            with open(os.path.join(self.project, name), "w", encoding="utf-8") as f:
                f.write(text)
        for command in (["init", "-q"], ["add", "."]):
            subprocess.run(["git", *command], cwd=self.project, check=True)

    def plain(self, *messages, exe=None):
        """Run the exe's plain prompt in the project with these messages, then /exit; what it printed."""
        done = subprocess.run([exe or self.exe, "--plain"], cwd=self.project, input="\n".join([*messages, "/exit", ""]),
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=SECONDS)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        return done.stdout + done.stderr

    def test_the_settings_are_kept_next_to_the_exe(self):
        other = os.path.join(self.root, "other")
        os.mkdir(other)
        exe = shutil.copy(EXE, other)
        printed = self.plain(exe=exe)  # with no settings beside it, the defaults are written there
        self.assertIn(f"[wrote the default settings to {os.path.join(other, 'settings.json')}]", printed)
        with open(os.path.join(other, "settings.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["approve"], "ask")
        self.assertFalse(os.path.exists(os.path.join(self.project, "settings.json")))
        self.plain("hello")  # and the ones beside it are read: they name the fake server
        self.assertEqual(len(Model.calls), 1)
        self.assertEqual(len(glob.glob(os.path.join(self.bin, "sessions", "*.jsonl"))), 1)  # the session is beside it too
        self.assertFalse(os.path.exists(os.path.join(self.project, "sessions")))

    def test_pdf_and_word_files_are_read(self):
        import pymupdf
        with pymupdf.open() as pdf:
            pdf.new_page().insert_text((72, 72), "The ring was found in the river.")
            pdf.new_page()  # a page without text, as a scan is: it is made an image
            pdf.save(os.path.join(self.project, "story.pdf"))
        with zipfile.ZipFile(os.path.join(self.project, "notes.docx"), "w") as docx:
            docx.writestr("word/document.xml", '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml'
                          '/2006/main"><w:body><w:p><w:r><w:t>Speak friend and enter.</w:t></w:r></w:p></w:body>'
                          '</w:document>')
        Model.answers = ["They tell a story.\nDONE"]
        self.plain("@story.pdf @notes.docx what do these say?")
        told = said(Model.calls[0])
        self.assertIn("The ring was found in the river.", told)
        self.assertIn("1 page(s) without text, such as scans, are sent as images", told)
        self.assertIn("Speak friend and enter.", told)
        self.assertNotIn("cannot be read", told)

    def test_git_and_commands_run_without_the_exe_s_own_setup(self):
        Model.answers = [f"Let me look.\nRUN: {PYTHON} probe.py", "DONE"]
        self.plain("what can a command see?")
        told = said(Model.calls[0])
        self.assertIn("main.py", told)
        self.assertNotIn("ignored.txt", told)  # listed by git, which .gitignore is read by
        output = said(Model.calls[1][-1:])
        print(f"\n{output[output.find('PYI:'):][:2000]}")
        self.assertIn("PYI: []", output)
        if os.name == "nt":
            self.assertIn("DLL folder: ''", output)
            self.assertIn("from tolken.exe: []", output)

    def test_a_tolken_exe_started_by_a_command_runs(self):
        Model.answers = ["RUN: tolken --plain", "DONE"]
        with mock.patch.dict(os.environ, PATH=self.bin + os.pathsep + os.environ["PATH"]):  # as .local\bin is
            self.plain("start another")
        output = said(Model.calls[1][-1:])
        self.assertIn("8192 tokens | approve: auto", output)  # the second one's plain prompt, with the same settings
        self.assertNotIn("Traceback", output)

    def test_the_full_screen_interface(self):
        Model.answers = ["The answer, with code:\n\n```python\nprint('the one ring')\n```\n\nDONE"]
        terminal = Terminal([self.exe], self.project)
        self.addCleanup(terminal.stop)
        terminal.wait_for("0/8,192")  # the status line, with the window of the model it found
        self.assertIn("enter send", terminal.shown())  # not the plain prompt, to which it falls back without Textual
        terminal.write("hello")
        terminal.wait_for("hello")
        terminal.write("\r")
        terminal.wait_for("print('the one ring')")
        self.assertNotIn("print('the one ring')", terminal.raw())  # coloured by Pygments, which splits it with codes
        self.assertEqual(said(Model.calls[0][-1:]).strip()[-5:], "hello")
        terminal.write("/exit")
        terminal.wait_for("/exit")
        terminal.write("\r")
        self.assertEqual(terminal.exit_code(), 0)


@unittest.skipUnless(os.name == "nt", "Windows only")
class Probe(unittest.TestCase):
    """The probe sees the DLL folder a parent gave its programs (else its "from tolken.exe: []" would prove nothing)."""

    def test_it_sees_a_dll_folder_given_to_it(self):
        folder = tempfile.mkdtemp(prefix="_MEI")
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        shutil.copy(glob.glob(os.path.join(sys.base_prefix, "DLLs", "libcrypto*.dll"))[0], folder)
        probe = os.path.join(folder, "probe.py")
        with open(probe, "w", encoding="utf-8") as f:
            f.write(PROBE)
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.SetDllDirectoryW(folder)
        try:
            done = subprocess.run([sys.executable, probe], capture_output=True, text=True, timeout=SECONDS)
        finally:
            kernel32.SetDllDirectoryW(None)
        print(f"\n{done.stdout}{done.stderr}")
        self.assertIn(os.path.basename(folder), done.stdout.split("from tolken.exe:")[1])


if __name__ == "__main__":
    unittest.main()
