#!/usr/bin/env python3
"""tolken: a minimal coding harness for a local llama-server with a small context window.

Run it from the project you want to ask about. A message is one model call, plus
one for each time the model asks for more files or, after its edits, goes on.
In a terminal with Textual installed it opens the full-screen interface (tui.py);
otherwise, or with --plain, it is a plain prompt.

On Windows it is also tolken.exe, from the releases on GitHub: all of it in one
file that needs nothing installed (tolken.spec builds it). Put it in a folder on
PATH, such as %USERPROFILE%\\.local\\bin, and run tolken from the project.

  @path         pin a whole file
  @path:10-50    pin a line range
  /model         list the models (llama-server's and every Ollama server's); /model NAME switches
  /settings      list the settings and change one (full-screen interface)
  /undo          put back the files that the last message to change files changed
  /clear         forget the pinned code and the conversation
  /exit          exit
  \"\"\"            on a line by itself starts and ends a multi-line message (plain prompt)

Settings are read at startup from settings.json next to this file (to
tolken.exe), which is written with the defaults when it is missing; /settings
changes one, there and for the session that is running. Keys left out keep
their defaults:
  servers         llama-server and Ollama URLs, on this computer or another one on the
                  network (there, set OLLAMA_HOST=0.0.0.0:11434 and open port 11434);
                  each server is asked which of the two it is
  model           the model to start with, named as for /model; "" for the first one
                  listed (the models are listed in the order of their servers)
  context         context window in tokens; null for what the server gives. Ollama
                  loads the model at this size; llama-server's size is fixed by its
                  -c option, so there it can only make the window smaller
  think           false, true, or a level: "low", "medium" or "high"
  think_max       tokens of thinking after which the model is asked again without
                  thinking; null for no cap (then only max_out ends it)
  max_out        tokens kept for the answer, at most a quarter of the window; thinking
                  counts against it too
  options         sent to the server as they are: temperature, top_p, top_k, min_p,
                  repeat_penalty, seed...
  approve         "ask" shows each change (edit, new file, deletion) and asks before
                  making it; "auto" makes it and shows it. Shift+Tab switches between
                  them in the full-screen interface
  run             the commands the model may run: a command is allowed when its words
                  begin with those of an entry. In "auto" they run at once; any other
                  command, and every command in "ask", waits for the user
  instructions    text added to the end of the system prompt, for every project:
                  "Before ending with DONE, run RUN: build.cmd and fix what fails"
  suggest         what the full-screen interface offers as the next message, in the
                  empty input, for Tab to take: "rules" for what follows plainly from
                  how the last message ended ("continue", "commit the changes"),
                  "model" to ask the model as well when nothing does, false for nothing

Every prompt carries the project's file list. A pinned file also brings the
project files it #includes, while they fit. Without a pin, the words of the
question are searched in the current directory and the best-matching snippets
are sent. If the model needs other files it replies "NEED: path ...", and
tolken loads them and asks again, for as long as the files fit beside the code
already loaded. Loaded code stays until /clear.

Each call continues the conversation of the call before: the new message holds
only the code it does not have yet (of a file it has an older copy of, only the
changed parts, with lines around them), files added to or removed from the
project, and the question. The server has the rest cached,
so it reads only what is new. When that no longer fits the window, the prompt
is built afresh: the file list, the loaded code, the newest exchanges and the
question. The file list and those exchanges grow with the window.

Edits are applied as each answer arrives, once the user confirms them. The
model is then called again, and told how its edits went, unless it ended the
answer with a DONE line; it is also called again when it ends with a CONTINUE
line, asks for files, an edit could not be applied, or its answer was cut off
at max_out. The user answers a change or a command with yes, no, or what to do
instead; after a no, the rest of that answer is not asked about or made. With
what to do instead, the model is called again at once and told it. It stops
when the user just says no, when two answers in a row change nothing, when the
window is full and an answer changes nothing, or after MAX_CALLS calls. What
the model was not told then (an edit declined or undone, one that failed, a
command's output) comes before the next message.

The model runs a command (to build or test) with a "RUN: command" line, after
its edits are made. It runs without a shell (shell syntax is refused, since a
.cmd or .bat file runs in cmd.exe) in the project folder for at most
RUN_TIMEOUT seconds, and the start and end of its output go back to the model,
which is then called again.

Every call of the model is logged, to look into later what it was sent and what
it answered: a file for each session in the logs folder next to this file (to
tolken.exe), a JSON line for each call with the messages the call added to the
prompt, the model's thinking and its answer as it came, and a line with what
the answer led to (the edits made, what could not be done, the commands'
output). The newest KEEP_LOGS files are kept.
"""
import concurrent.futures
import difflib
import http.client
import itertools
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# The folder of this file, where the settings and the logs are kept. tolken.exe (see tolken.spec) is unpacked
# to a new temporary folder at each start, so it keeps them next to itself.
HOME = os.path.dirname(sys.executable if getattr(sys, "frozen", False) else os.path.abspath(__file__))
SETTINGS = os.path.join(HOME, "settings.json")
# The folder the model's calls are logged in, a file for each session (see Session.keep); None for no log.
LOGS = os.path.join(HOME, "logs")
KEEP_LOGS = 30          # the newest log files kept; older ones are deleted when a new one is begun
# The defaults of the settings; load_settings() replaces them with those of the file, and writes them
# to the file when there is none.
SERVERS = ["http://169.254.51.51:11434", "http://127.0.0.1:8080"]  # an Ollama server's and llama-server's
MODEL = "bonsai-2:27b"  # the model to start with, named as for /model; "" for the first one listed
CONTEXT = 131072        # context window in tokens; None for what the server gives
THINK = True            # False, True, or a level: "low", "medium" or "high"
THINK_MAX = None        # tokens of thinking after which the model is asked again without thinking; None for no cap
OPTIONS = {"temperature": 0.6}  # sent to the server as they are
MAX_OUT = 16000         # tokens reserved for the answer, thinking included
APPROVE = "ask"         # "ask": each change waits for the user; "auto": changes are made at once (anything else asks)
# The commands allowed to run; arguments may follow.
RUN = ["build.cmd", "git status", "git diff", "git log", "git show", "git add", "git commit"]
# The user's own text, added to the end of the system prompt.
INSTRUCTIONS = ("After your changes, before ending with DONE, run RUN: build.cmd and fix any errors it reports.\n"
                "Git: you may read the repository with RUN: git status, RUN: git diff, RUN: git log and RUN: git show."
                " Commit only when the user asks: write RUN: git add followed by the files, then RUN: git commit -m \"message\", with the message on one line, in double quotes, and without % & ; or backquotes.")
SUGGEST = "model"       # the next message offered in the empty input: False, "rules", or "model" (which asks the model too)
RUN_TIMEOUT = 120       # seconds a command may run before it is stopped
RUN_CHARS = 3000        # characters of a command's output the model gets, per 8,192 tokens of window
SEARCH_TOKENS = 4000    # at most this much automatically found code per question, however large the window
MARGIN = 200            # allowance for chat-template tokens
IMAGE_TOKENS = 1600     # what an image of 1,280 x 1,280 pixels costs, about; Ollama cannot count it
DOCUMENTS = {".pdf", ".docx"}  # shown as their text
IMAGES = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}  # sent as pictures, to a model that can see
BEFORE, AFTER = 10, 20  # lines kept around a search hit
MAX_FILE = 1_000_000    # bytes; larger files are not searched
THREADS = 32            # files read at a time: opening one takes long on Windows, and a large project has many
GIT_SECONDS = 10        # how long git may take to list the files; after that it is asked for the tracked ones only
COMMON = 200            # a word found on more lines than this is ignored
MAX_ANSWER = 3000       # characters of earlier exchanges carried into a prompt built afresh, per 8,192 tokens of window
MAP_CHARS = 1500        # characters of the file list, per 8,192 tokens of window
MAX_CALLS = 20          # model calls per message at most, so a model that keeps editing cannot go on forever
SKIP_DIRS = {"node_modules", "build", "dist", "out", "target", "bin", "obj",
             "__pycache__", "venv"}  # hidden directories are skipped as well
STOPWORDS = set("""
the and for with that this what why how does did can could should would not are
was were has have had from into when where which who its there their then than
them they you your our but any all out use used using about after before file
files code function functions class method line lines error bug fix make made
change work works working please explain show tell find look need want here
like just also
""".split())

EXAMPLE_PATH = "<path>"  # stands for a real path in RULES; an edit that still has angle brackets is refused
# How a change is written: part of the rules, and shown again when a block could not be read.
FORM = f"""FILE: {EXAMPLE_PATH}
<<<<<<< SEARCH
existing lines to replace, copied exactly, as few as possible
=======
new lines
>>>>>>> REPLACE"""
WRITING = f"""You are a coding assistant working inside the user's project folder; its files are listed below. You can read, create and change files in it. Answer briefly.

Reading: you only see the contents of files shown in the conversation; do not guess what a file contains from its name. A file shown again replaces its earlier copy; when only its changed parts are shown, the rest of it is as before. When a question is about the project's existing code and you need files that are not shown, reply with only this line:
NEED: path/one path/two
The files will then be shown to you. Ask for whole files; add :START-END to a path only to read another part of a file that was shown cut short. Do not ask for files the question does not depend on, and never ask the user to paste files.

Writing: when asked to create or change files, do it yourself with edit blocks; they are applied for you after the user confirms. Never say you cannot create or edit files. Each change is a FILE: line and a block, where {EXAMPLE_PATH} is the file's path relative to the project folder:

{FORM}

To create a new file, leave SEARCH empty; put it directly in the project folder unless the user names a place.
To delete or remove a file, do not use a block; write this line for it:
DELETE: {EXAMPLE_PATH}
Your changes are applied as you give them, and you are then told how they went, so you can make them a file or two at a time. When the task is finished, end the answer with a line saying only DONE. To go on without changing files now (after making a plan, say), end it with a line saying only CONTINUE."""


def running(allowed):
    """The part of the rules about running commands, naming the allowed ones."""
    return ("Running: to build the project or run its tests, write a line\nRUN: <command>\nIt runs after your "
            "changes are applied, and its output comes back to you. These commands are allowed (arguments may "
            f"follow them): {', '.join(allowed) or 'none'}. Any other command needs the user's approval, and the "
            "user may refuse it. Do not use shell syntax: no pipes, &, &&, ;, ^, %, or redirection.")


def rules():
    """The system prompt: the writing rules, the running rules naming the allowed commands, the user's instructions."""
    return "\n\n".join(part for part in (WRITING, running(RUN), INSTRUCTIONS.strip()) if part)


RULES = rules()  # the system prompt; load_settings() builds it again with the settings' commands and instructions

NO_CODE = "(No file contents have been shown yet.)"  # stands where the code would be
GO_ON = ("Continue the task. The latest copy of each file above, with the changed parts shown after it, "
         "is as it is now. Give each change as a FILE: line and a block of <<<<<<< SEARCH, ======= and "
         ">>>>>>> REPLACE lines. If the task is finished, reply with only DONE.")  # follows the results of the model's edits

# The path line is left out when a block follows another one for the same file. A block's text never runs
# past the next <<<<<<< SEARCH line, so a block left open is dropped, not run into the next one. Small
# models often end a block with a second ======= line; it is taken for the end when no >>>>>>> REPLACE
# comes before the next block or the end of the answer.
BLOCK = re.compile(
    r"^(?:([^\n]+)\n+)?(?:```[^\n]*\n)?<{5,9} SEARCH[ \t]*\n((?:(?!^<{5,9} SEARCH).)*?)^={5,9}[ \t]*\n"
    r"((?:(?!^<{5,9} SEARCH).)*?)"
    r"(?:^>{5,9} REPLACE|^={5,9}[ \t]*$(?=(?:(?!^<{5,9} SEARCH|^>{5,9} REPLACE).)*(?:\Z|^<{5,9} SEARCH)))",
    re.S | re.M)
LABEL = re.compile(r"^(?:file(?:[ _]?(?:path|name))?|path)\s*[:=]\s*", re.I)  # "FILE:" before a block's path
INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*"([^"]+)"', re.M)
# The model has more to do; markdown and case are allowed, since a missed one ends the task too early.
# Both are looked for outside the edit blocks, and not indented, as a line of code would be. CONTINUE
# stands alone on its line, so that a sentence such as "Continue using the old API" is not one.
CONTINUE = re.compile(r"^[>*_`#-]*CONTINUE[*_`.!]*[ \t\r]*$", re.M | re.I)
# The task is finished; in capitals only, so that "Done with a.py, now b.py" does not end it.
DONE = re.compile(r"^[>*_`#-]*DONE\b", re.M)
RUN_LINE = re.compile(r"^RUN:(.*)$", re.M)  # a command the model wants to run
SHELL = re.compile(r"\|\||&&|[|;<>`&^%]")  # what only a shell understands; a .cmd or .bat file runs in cmd.exe


# ---------- context: pins, includes, search, overview ----------

def norm(path):
    return os.path.normpath(path).replace("\\", "/")


def read_text(path, errors="strict"):
    with open(path, encoding="utf-8", errors=errors, newline="") as f:
        return f.read()


def split_range(token):
    """(path, start, end) for "path" or "path:START-END"; end is None for a whole file."""
    ranged = re.match(r"^(.*):(\d+)-(\d+)$", token)
    return (ranged[1], max(1, int(ranged[2])), int(ranged[3])) if ranged else (token, 1, None)


def parse_pins(text):
    """Context entries [path, start, end] for the @mentions that name existing files.

    A path with spaces is written in quotes: @"My Documents/spec.pdf".
    """
    pins = []
    for quoted, bare in re.findall(r'@"([^"]+)"|@(\S+)', text):
        path, start, end = split_range(quoted or bare.rstrip(".,;:?!)"))
        if os.path.isfile(path):
            pins.append([norm(path), start, end])
    return pins


def label(entry):
    path, start, end = entry
    return path if (start, end) == (1, None) else f"{path}:{start}-{end}"


def kind(path):
    """"document" for a PDF or Word file, "image" for a picture, None for any other (text) file."""
    extension = os.path.splitext(path)[1].lower()
    return "document" if extension in DOCUMENTS else "image" if extension in IMAGES else None


READ = {}  # absolute path -> ((mtime, size), (text, images)): what documents.py made of a document or image


def contents(path):
    """(text, images) of a PDF, Word or image file, remembered until the file changes.

    images are PNGs in base64: the picture itself, or a PDF's pages without text (scans).
    A file that cannot be read gives a text saying so.
    """
    try:
        import documents  # PyMuPDF is loaded only once such a file is shown
    except ImportError:
        return "[this file cannot be read: PyMuPDF is not installed (pip install pymupdf)]", []
    status = os.stat(path)
    key, known = (status.st_mtime_ns, status.st_size), READ.get(os.path.abspath(path))
    if known and known[0] == key:
        return known[1]
    with open(path, "rb") as f:
        data = f.read()
    try:
        if kind(path) == "image":
            found = ("", [documents.read_image(data)])
        elif path.lower().endswith(".docx"):
            found = (documents.read_docx(data)["text"], [])
        else:
            pdf = documents.read_pdf(data)
            notes = [f"[{len(pdf['images'])} page(s) without text, such as scans, are sent as images]"] * bool(
                pdf["images"]) + [f"[{pdf['note']}]"] * bool(pdf["note"])
            found = ("\n\n".join([pdf["text"], *notes]).strip(), [image["data"] for image in pdf["images"]])
    except Exception as error:  # a broken file; PyMuPDF raises its own kinds of error
        found = (f"[this file could not be read: {error}]", [])
    READ[os.path.abspath(path)] = (key, found)
    return found


def text_lines(path):
    """The lines a file is shown as: a text file's own, a document's text, none for an image."""
    return contents(path)[0].splitlines() if kind(path) else read_text(path, "replace").splitlines()


def images_of(entry):
    """The images that go with an entry: an image file's, a PDF's scanned pages; none for text."""
    return contents(entry[0])[1] if kind(entry[0]) else []


def snippet(entry):
    """The header and code of one context entry, as sent to the model."""
    path, start, end = entry
    if kind(path) == "image":
        return f"--- {path} (an image of {os.path.getsize(path):,} bytes; it is attached) ---"
    lines = text_lines(path)
    end = min(end or len(lines), len(lines))
    return f"--- {path} (lines {start}-{end} of {len(lines)}) ---\n" + "\n".join(lines[start - 1:end])


def changed_parts(entry, old, new):
    """new, the snippet of a whole text file whose older snippet old the model has, as only the parts that changed.

    Each part comes with 10 lines around it and the line numbers it had. new itself when
    there is no older copy, the entry is a range or not text, or the parts are not much shorter.
    """
    path, start, end = entry
    if old is None or kind(path) or (start, end) != (1, None):
        return new
    before, after = old.split("\n")[1:], new.split("\n")[1:]  # without the "--- path" line
    parts = []
    for group in difflib.SequenceMatcher(None, before, after, autojunk=False).get_grouped_opcodes(10):
        i1, i2, j1, j2 = group[0][1], group[-1][2], group[0][3], group[-1][4]
        if j2 <= j1:
            return new
        parts.append(f"lines {j1 + 1}-{j2} (were {i1 + 1}-{i2}):\n" + "\n".join(after[j1:j2]))
    text = f"--- {path} (changed parts of its {len(after)} lines; the rest is as shown before) ---\n"
    text += "\n⋮\n".join(parts)
    return text if parts and len(text) <= len(new) // 2 else new


def chars(entries):
    """Size of the entries' snippets in characters."""
    return sum(len(snippet(entry)) for entry in entries)


def within(entries, budget):
    """entries without its first (least important) ones, so the rest fits budget characters."""
    sizes = [len(snippet(entry)) for entry in entries]
    total, first = sum(sizes), 0
    while first < len(sizes) and total > budget:
        total -= sizes[first]
        first += 1
    return entries[first:]


UNTRACKED = True  # whether git is asked for the untracked files too; False once that took too long


def git_files():
    """The files git lists under the current directory; None outside a git work tree, or without git.

    They are the tracked files and the untracked ones that .gitignore does not exclude. To find the
    untracked ones git reads every folder; once that has taken over GIT_SECONDS, only the tracked files
    are listed from then on, which git knows without looking (Session.send() tells the user).
    """
    global UNTRACKED
    command = ["git", "ls-files", "-z", "--cached"]
    try:
        if UNTRACKED:
            try:
                done = subprocess.run(command + ["--others", "--exclude-standard"], capture_output=True,
                                      timeout=GIT_SECONDS)
            except subprocess.TimeoutExpired:
                UNTRACKED = False
        if not UNTRACKED:
            done = subprocess.run(command, capture_output=True, timeout=GIT_SECONDS)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode:
        return None
    return [norm(path) for path in done.stdout.decode("utf-8", "replace").split("\0") if path]


def project_files():
    """Paths of the files under the current directory, without the skipped and hidden directories.

    In a git work tree, only the files git lists (so not what .gitignore excludes), in the same order.
    """
    listed = git_files()
    if listed is None:
        for folder, dirs, files in os.walk("."):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
            for name in sorted(files):
                yield norm(os.path.join(folder, name))
        return
    kept = [path.split("/") for path in listed]
    kept = [parts for parts in kept if not any(d in SKIP_DIRS or d.startswith(".") for d in parts[:-1])]
    # The order os.walk gives: a folder's own files first, then its folders, each sorted by name.
    for parts in sorted(kept, key=lambda parts: [(1, d) for d in parts[:-1]] + [(0, parts[-1])]):
        yield "/".join(parts)


def lines_of(path):
    """The lines of a text file, or None when it is too large or not UTF-8 text."""
    try:
        if os.path.getsize(path) <= MAX_FILE:
            text = read_text(path)
            return None if "\0" in text else text.splitlines()
    except (OSError, UnicodeDecodeError):
        pass
    return None


def includes(pins):
    """Whole-file entries for the project files that pinned files #include, least important first.

    A header named like the pinned file (foo.h for foo.cpp) comes last.
    """
    found, project = [], None
    for path, _, _ in pins:
        if kind(path):  # a document or an image includes nothing
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        targets = []
        for name in INCLUDE.findall(read_text(path, "replace")):
            target = norm(os.path.join(os.path.dirname(path), name))
            if not os.path.isfile(target):
                if project is None:
                    project = list(project_files())
                target = next((p for p in project if ("/" + p).endswith("/" + norm(name))), None)
            if target:
                targets.append(target)
        found += sorted(targets, key=lambda t: os.path.splitext(os.path.basename(t))[0] == stem)
    pinned = {entry[0] for entry in pins}
    return [[target, 1, None] for target in dict.fromkeys(found) if target not in pinned]


TEXT = {}  # absolute path -> ((mtime, size), whether it is text), so later messages need not open every file


def is_text(path):
    """False for files that are too large or look binary. Remembered until the file changes."""
    try:
        status = os.stat(path)
        key, known = (status.st_mtime_ns, status.st_size), TEXT.get(os.path.abspath(path))
        if known and known[0] == key:
            return known[1]
        with open(path, "rb") as f:
            text = status.st_size <= MAX_FILE and b"\0" not in f.read(1024)
    except OSError:
        return False
    TEXT[os.path.abspath(path)] = (key, text)
    return text


def in_threads(work, paths, session=None, doing=""):
    """work(path) for each of paths, in their order, worked out THREADS at a time.

    With a session, doing and how many are done are shown as its activity, and the results end early
    once it is cancelled.
    """
    pool, shown = concurrent.futures.ThreadPoolExecutor(THREADS), time.monotonic()
    try:
        for number, result in enumerate(pool.map(work, paths), 1):
            if session and session.stopped:
                return
            if session and time.monotonic() - shown >= 0.2:
                session.ui.activity(f"{doing} · {number:,} of {len(paths):,} files")
                shown = time.monotonic()
            yield result
    finally:
        pool.shutdown(wait=False, cancel_futures=True)  # the files not begun yet are left alone


def readable_files(session=None):
    """The project's files that can be shown: text (not too large), PDF and Word documents, and images.

    With a session, its activity shows how far the listing is, and a cancelled one gets a list cut short.
    """
    paths = list(project_files())
    shown = in_threads(lambda path: bool(kind(path)) or is_text(path), paths, session, "listing project")
    return [path for path, show in zip(paths, shown) if show]


def project_map(budget, files=None):
    """The folder name and file list, shortened to fit budget characters.

    Folders are shown as one row each, "folder/ (N files)", one at a time until the list fits:
    the deepest first and, of those, the one with the most rows. files are the project's text
    files, when they have been listed already.
    """
    files = readable_files() if files is None else files
    folders = {}  # folder -> [the files right in it, the folders right in it, how many files it holds]
    for path in files:
        parts = path.split("/")
        for depth in range(1, len(parts)):
            folder = folders.setdefault("/".join(parts[:depth]) + "/", [[], set(), 0])
            folder[2] += 1
            if depth == len(parts) - 1:
                folder[0].append(path)
            else:
                folder[1].add("/".join(parts[:depth + 1]) + "/")

    def row(folder):
        return f"{folder} ({folders[folder][2]} files)"

    size, collapsed = sum(len(path) + 1 for path in files), set()  # size: of the list, with a newline per row
    for folder in sorted(folders, key=lambda f: (-f.count("/"), -len(folders[f][0]) - len(folders[f][1]))):
        if size - 1 <= budget:
            break
        direct, inner, _ = folders[folder]  # the folders in it are collapsed already, being deeper
        size += len(row(folder)) + 1 - sum(len(path) + 1 for path in direct) - sum(len(row(f)) + 1 for f in inner)
        collapsed.add(folder)

    def shown(path):
        """The row that path is listed under: its outermost collapsed folder, or itself."""
        parts = path.split("/")
        return next((f for f in ("/".join(parts[:depth]) + "/" for depth in range(1, len(parts))) if f in collapsed),
                    path)

    rows = dict.fromkeys(shown(path) for path in files)
    listing = "\n".join(row(r) if r.endswith("/") else r for r in rows)
    return f"Project folder: {os.path.basename(os.getcwd())}\nFiles:\n{listing[:budget]}"


def heads(paths, budget):
    """Entries for the first lines of each file: as many lines each as fit budget characters."""
    files = {path: lines for path in paths if (lines := text_lines(path) if kind(path) else lines_of(path))}
    for keep in (100_000, 800, 600, 400, 300, 200, 150, 100, 60, 40, 25, 15, 5, 1):
        if sum(len("\n".join(lines[:keep])) + len(path) + 30 for path, lines in files.items()) <= budget:
            break
    return within([[path, 1, None if keep >= len(lines) else keep] for path, lines in files.items()], budget)


def keywords(question):
    """Search words of a question (lowercase) with their weights."""
    words = {}
    for word in re.findall(r"[A-Za-z_]\w{2,}", question):
        if word.lower() not in STOPWORDS:
            identifier = "_" in word or any(c.isdigit() or c.isupper() for c in word[1:])
            words[word.lower()] = max(words.get(word.lower(), 0), 3 if identifier else 1)
    return words


def search(question, budget, files=None, session=None):
    """Pick snippets for a question by plain text search under the current directory.

    budget is in characters; files are the project's readable files, when they have been listed
    already (only text files are searched). Returns context entries, least relevant first.
    With a session, its activity shows how far the search is, and cancelling it cuts the search short.
    """
    words = keywords(question)
    if not words:
        return []

    def whole(words):
        return re.compile(r"\b(?:" + "|".join(words) + r")\b", re.I)

    pattern, sought = whole(words), set(words)  # sought: the words still looked for

    def holding(path):
        """The lines of a text file that holds one of the words somewhere, else None: most files hold none,
        and one look at the whole text spares going through their lines."""
        try:
            if not kind(path) and os.path.getsize(path) <= MAX_FILE:
                text = read_text(path)
                if "\0" not in text and pattern.search(text):
                    return text.splitlines()
        except (OSError, UnicodeDecodeError):
            pass
        return None

    paths = list(project_files()) if files is None else files
    hits, sizes, counts = {}, {}, dict.fromkeys(words, 0)  # hits: path -> {line number: words}
    for path, lines in zip(paths, in_threads(holding, paths, session, "searching project")):
        for number, line in enumerate(lines or (), 1):
            found = {w.lower() for w in pattern.findall(line)} & words.keys()
            if found:
                hits.setdefault(path, {})[number] = found
                sizes[path] = len(lines)
                for word in found:
                    counts[word] += 1
        if lines and any(counts[word] > COMMON for word in sought):
            # A word found on that many lines is ignored below, so it need not be looked for any further.
            sought = {word for word in sought if counts[word] <= COMMON}
            if not sought:
                break
            pattern = whole(sought)
    words = {w: weight for w, weight in words.items() if counts[w] <= COMMON}
    windows = []  # [score, path, start, end]
    for path, lines in hits.items():
        bonus = 5 * sum(w in os.path.basename(path).lower() for w in words)
        for number in sorted(lines):
            score = sum(words.get(w, 0) for w in lines[number])
            if not score:
                continue
            start, end = max(1, number - BEFORE), min(sizes[path], number + AFTER)
            if windows and windows[-1][1] == path and start <= windows[-1][3] + 1:
                windows[-1][0] += score + bonus
                windows[-1][3] = end
            else:
                windows.append([score + bonus, path, start, end])
    windows.sort(key=lambda w: -w[0])
    chosen = []
    for _, path, start, end in windows:
        budget -= len(snippet([path, start, end]))
        if chosen and budget < 0:
            break
        chosen.append([path, start, end])
    return chosen[::-1]


# ---------- display ----------

class Plain:
    """Shows tolken's output with print() and asks with input().

    The full-screen interface (tui.py) provides the same methods, apart from answer().
    """

    thought = False  # thinking text was printed last, so the answer starts on a new line

    def note(self, text, kind="info"):
        """A line about what happened. kind is "info", "good" (it worked), "bad" (it failed) or "warn"
        (something to know about, though the work went on)."""
        print(f"[{text}]")

    def activity(self, text, call=None, used=None):
        """What is happening now, with the call number and prompt tokens once known."""

    def thinking(self, piece):
        """A streamed piece of the model's thinking, which is not part of the answer. Dim in a terminal."""
        print(f"\033[2m{piece}\033[0m" if sys.stdout.isatty() else piece, end="", flush=True)
        self.thought = True

    def text(self, piece):
        """A streamed piece of the answer."""
        if self.thought:
            print()
            self.thought = False
        print(piece, end="", flush=True)

    def timings(self, data):
        """The server's timing figures for the call that just ended."""

    def show(self, path, old, new):
        """Show how a file's text changes, without asking (the "auto" mode). new is None for deleting the file."""
        if new is None:
            print(f"{path}: {len(old.splitlines())} lines")
        else:
            print("\n".join(difflib.unified_diff(old.splitlines(), new.splitlines(), path, path, lineterm="")))

    def confirm(self, path, old, new):
        """Show how a file's text would change and ask whether to apply it. new is None for deleting the file.

        Returns True (yes), False (no) or the user's comment (a no, with what to do instead).
        """
        self.show(path, old, new)
        return self.answer(f"Delete {path}? [y/N, or what to do instead] " if new is None
                           else f"Apply to {path}? [y/N, or what to do instead] ")

    def allow(self, command):
        """Ask whether to run a command the model wants to run. Returns what confirm() does."""
        return self.answer(f"Run {command}? [y/N, or what to do instead] ")

    def ran(self, command, code, seconds, output):
        """Show the output of a command that ran; code is None when it was stopped for taking too long."""
        print(clip(output, RUN_CHARS))

    def answer(self, question):
        """y or yes: True; n, no or nothing: False; any other text: the user's comment (a no, with what to do instead)."""
        try:
            answer = input(question).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        return True if answer.lower() in ("y", "yes") else False if answer.lower() in ("", "n", "no") else answer


PLAIN = Plain()


# ---------- prompt and server ----------

def build(ctx, question, history, project="", vision=False):
    """Chat messages: rules and file list, then the code, then the earlier exchanges and the question.

    With vision the images of the code (pictures, a PDF's scanned pages) go with the message holding it.
    """
    code = "\n\n".join(snippet(entry) for entry in ctx) or NO_CODE
    messages = [{"role": "system", "content": f"{RULES}\n\n{project}" if project else RULES}]
    for asked, answered in history:
        messages += [{"role": "user", "content": asked}, {"role": "assistant", "content": answered}]
    messages.append({"role": "user", "content": question})
    messages[1]["content"] = f"{code}\n\n{messages[1]['content']}"
    pictures = [picture for entry in ctx for picture in images_of(entry)] if vision else []
    if pictures:
        messages[1]["images"] = pictures
    return messages


def recent(history, budget):
    """The newest exchanges whose text fits budget characters: at least the newest, its answer cut to fit."""
    kept = []
    for asked, answered in reversed(history):
        if kept and len(asked) + len(answered) > budget:
            break
        kept.insert(0, (asked, answered if kept else answered[:budget]))
        budget -= len(asked) + len(kept[0][1])
    return kept


def changes(listed, files):
    """A line about the files added to and removed from the project since the file list was sent; "" for none."""
    added, removed = [path for path in files if path not in listed], sorted(listed - set(files))
    return " ".join([f"Files added since the file list above: {', '.join(added)}."] * bool(added)
                    + [f"Files removed since the file list above: {', '.join(removed)}."] * bool(removed))


def post(path, body, base, session=None):
    """The response to body, posted as JSON to a server.

    With a session, the connection's socket is put on it as soon as the connection is made, so that
    Session.cancel() can end a call that the server has not begun to answer: it may read the prompt for long.
    """
    request = urllib.request.Request(base + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    if not session:
        return urllib.request.urlopen(request)

    class Connection(http.client.HTTPConnection):
        def connect(self):
            super().connect()
            session.sock = self.sock
            if session.stopped:  # cancelled while it was being made
                raise OSError("cancelled")

    class Handler(urllib.request.HTTPHandler):
        def http_open(self, request):
            return self.do_open(Connection, request)

    try:
        return urllib.request.build_opener(Handler).open(request)
    except BaseException:
        session.sock = None
        raise


def get(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def explain(error):
    """A server or connection error as one line of text."""
    if not isinstance(error, urllib.error.HTTPError):
        return str(error)
    body = error.read().decode(errors="replace")
    try:
        body = json.loads(body).get("error", body)
    except (ValueError, AttributeError):
        pass
    return f"{error.code} {str(body)[:300]}"


def stream(response, read, ui=PLAIN, session=None):
    """Pass a streamed answer on through ui.text, and its thinking through ui.thinking; return the answer's text.

    read(line) turns one line of the stream into (piece of text, piece of thinking, timings,
    done, why the server stopped). The stream's socket is recorded on the session, so that
    Session.cancel() can end the call from another thread; what arrived until then is returned.
    The session also learns whether the model thought (session.thought) and whether the answer
    was cut off at its length limit (session.cut). When the session has a think_max, the stream
    ends after that many pieces of thinking (a piece is about a token) and session.overthought is set.
    The thinking itself is left in session.thinking, for the log.
    """
    pieces, thoughts, cap = [], [], getattr(session, "think_max", None)
    if session:
        session.sock = getattr(getattr(response.fp, "raw", None), "_sock", None)
    try:
        for raw in response:
            if session and session.stopped:
                break
            piece, thought, timings, done, reason = read(raw.decode("utf-8").strip())
            if timings:
                ui.timings(timings)
            if reason and session:
                session.cut = reason == "length"
            if thought:
                ui.thinking(thought)
                if session:
                    session.thought = True
                thoughts.append(thought)
                if cap and len(thoughts) >= cap:
                    session.overthought = True
                    break
            if piece:
                pieces.append(piece)
                ui.text(piece)
            if done:
                break
    except KeyboardInterrupt:
        ui.text("\n[stopped]")
        if session:
            session.stopped = True  # send() then ends the message, without this answer's edits and commands
    except (OSError, http.client.HTTPException) as error:
        if not (session and session.stopped):  # a cancelled call ends with a read error; that is expected
            raise OSError(f"the answer was cut off ({error})") from error
    finally:
        if session:
            session.sock, session.thinking = None, "".join(thoughts)
        response.close()
    ui.text("\n")
    return "".join(pieces)


# ---------- llama-server ----------

def at(url):
    """"@host" for a server on another computer, "" for one on this computer: tells apart a model on two servers."""
    host = urllib.parse.urlsplit(url).hostname
    return "" if host in ("localhost", "127.0.0.1", "::1") else f"@{host}"


def props(url):
    """What the llama-server at url reports about itself. Raises OSError when it cannot be reached."""
    return get(url + "/props")


def count(messages, url):
    """Tokens in the messages, counted by the tokenizer of the llama-server at url (no generation)."""
    with post("/tokenize", {"content": "\n".join(m["content"] for m in messages)}, url) as response:
        return len(json.load(response)["tokens"])


def llama_line(line):
    if line == "data: [DONE]":
        return None, None, None, True, None
    if not line.startswith("data: "):
        return None, None, None, False, None
    chunk = json.loads(line[6:])
    choice = (chunk.get("choices") or [{}])[0]
    delta = choice.get("delta", {})
    return (delta.get("content"), delta.get("reasoning_content"), chunk.get("timings"), False,
            choice.get("finish_reason"))


def answer_room(size):
    """Tokens kept for the answer (thinking included) in a window of size: max_out, but at most a quarter.

    The settings serve every model, and a small window must keep room for the prompt.
    """
    return min(MAX_OUT, size // 4)


def chat(messages, url, ui=PLAIN, session=None, size=None, think=None):
    """Stream one completion from the llama-server at url through ui.text and return its text.

    size is the model's window, which caps the answer's length (see answer_room); think
    is the think setting to use in place of the settings' one.
    """
    think = THINK if think is None else think
    thinking = {"enable_thinking": bool(think)}  # read by the chat templates of thinking models
    if isinstance(think, str):
        thinking["reasoning_effort"] = think
    response = post("/v1/chat/completions", {"messages": messages, "stream": True,
                                             "max_tokens": answer_room(size) if size else MAX_OUT,
                                             **OPTIONS, "chat_template_kwargs": thinking}, url, session)
    return stream(response, llama_line, ui, session)


class Llama:
    """The model that llama-server runs."""

    vision = False  # images are not sent to llama-server (it needs a projector model and another request form)

    def __init__(self, size, name="", url="http://localhost:8080"):
        self.size, self.name, self.url = min(size, CONTEXT or size), name, url  # the settings can only make it smaller
        self.label = f"llama.cpp{at(url)} · {name.removesuffix('.gguf') or url}"
        self.detail = f"{size:,} tokens"  # shown beside the label in the model list

    def enter(self):
        """Called when this model is switched to: read its size again, in case the server was restarted."""
        info = props(self.url)
        self.__init__(info["default_generation_settings"]["n_ctx"], os.path.basename(info.get("model_path") or ""),
                      self.url)

    def leave(self):
        """Called when another model is switched to. tolken cannot unload llama-server's model."""

    def count(self, messages):
        return count(messages, self.url)

    def chat(self, messages, ui=PLAIN, session=None, think=None):
        return chat(messages, self.url, ui, session, self.size, think)


# ---------- Ollama ----------

def ollama_line(line):
    if not line:
        return None, None, None, False, None
    chunk = json.loads(line)
    if chunk.get("error"):
        raise OSError(chunk["error"])
    timings = None
    if chunk.get("done"):  # the last line carries the figures; durations are in nanoseconds
        # Ollama does not say how much of the prompt was cached; its prompt count includes the cached part.
        timings = {"cache_n": None, "prompt_n": chunk.get("prompt_eval_count", 0),
                   "prompt_ms": chunk.get("prompt_eval_duration", 0) / 1e6,
                   "predicted_n": chunk.get("eval_count", 0), "predicted_ms": chunk.get("eval_duration", 0) / 1e6}
    message = chunk.get("message") or {}
    return message.get("content"), message.get("thinking"), timings, bool(chunk.get("done")), chunk.get("done_reason")


class Ollama:
    """One of the models that an Ollama server serves."""

    def __init__(self, name, detail="", url="http://localhost:11434"):
        self.name, self.detail, self.url = name, detail, url
        self.label = f"ollama{at(url)} · {name}"
        self.size = 4096  # replaced by enter() with what Ollama really gives the model
        self.thinks = False  # whether the model can think; only then is the think setting sent
        self.vision = False  # whether the model can see images; set by enter()
        self.chars_per_token = 3  # what count() divides by; learned from the prompt sizes Ollama reports

    def options(self):
        """What Ollama runs the model with: the settings' options, the answer's length and the context size."""
        return {**OPTIONS, "num_predict": answer_room(self.size), **({"num_ctx": CONTEXT} if CONTEXT else {})}

    def enter(self):
        """Load the model and find out the context size Ollama runs it with."""
        with post("/api/show", {"model": self.name}, self.url) as response:
            show = json.load(response)
        self.thinks = "thinking" in (show.get("capabilities") or [])
        self.vision = "vision" in (show.get("capabilities") or [])
        # No messages: just load, with the options of the chat calls, so that the first of them does not reload it.
        post("/api/chat", {"model": self.name, "messages": [], "options": self.options()}, self.url).close()
        loaded = [model for model in get(self.url + "/api/ps").get("models", []) if model.get("name") == self.name]
        declared = re.search(r"num_ctx\s+(\d+)", show.get("parameters") or "")
        self.size = (loaded and loaded[0].get("context_length")) or (declared and int(declared[1])) or 4096

    def leave(self):
        """Unload the model at once, so its GPU memory is free for the next one."""
        try:
            post("/api/chat", {"model": self.name, "messages": [], "keep_alive": 0}, self.url).close()
        except OSError:
            pass

    def count(self, messages):
        """An estimate on the safe side: Ollama has no tokenizer to ask, and cuts a too-long prompt silently."""
        pictures = sum(len(message.get("images") or ()) for message in messages)
        return int(sum(len(message["content"]) for message in messages) / self.chars_per_token) + IMAGE_TOKENS * pictures

    def learn(self, messages, tokens):
        """Set chars_per_token from a prompt that Ollama says was tokens long, less a tenth to stay on the safe side.

        A prompt with images teaches nothing, nor a figure no tokenizer gives (over 6 characters a token).
        """
        chars = sum(len(message["content"]) for message in messages)
        if tokens and not any(message.get("images") for message in messages) and chars / tokens <= 6:
            self.chars_per_token = 0.9 * chars / tokens

    def chat(self, messages, ui=PLAIN, session=None, think=None):
        """Stream one answer through ui; think is the think setting to use in place of the settings' one."""
        body = {"model": self.name, "messages": messages, "stream": True, "options": self.options()}
        if self.thinks:  # Ollama refuses the think setting for a model that cannot think
            body["think"] = THINK if think is None else think

        def read(line):
            piece, thought, timings, done, reason = ollama_line(line)
            if timings:
                self.learn(messages, timings["prompt_n"])
                limit = self.size - answer_room(self.size)
                if timings["prompt_n"] > limit:  # the estimate was low, and Ollama cuts a prompt that is too long
                    ui.note(f"Ollama read {timings['prompt_n']:,} prompt tokens, more than the {limit:,} meant "
                            "for it; it may have cut the start of the prompt", "warn")
            return piece, thought, timings, done, reason

        return stream(post("/api/chat", body, self.url, session), read, ui, session)


def models():
    """The models that can be chosen, server by server; a server that does not answer is left out.

    A server that answers /props is llama-server; one that answers it with an error is asked as Ollama.
    """
    found = []
    for url in SERVERS:
        try:
            try:
                info = props(url)
            except urllib.error.HTTPError:  # a server, but not llama-server
                for model in get(url + "/api/tags").get("models", []):
                    found.append(Ollama(model["name"], f"{model.get('size', 0) / 1e9:.1f} GB", url))
            else:
                found.append(Llama(info["default_generation_settings"]["n_ctx"],
                                   os.path.basename(info.get("model_path") or ""), url))
        except (OSError, KeyError, ValueError):
            pass
    return found


def pick(found, wanted):
    """The model in the list that a number or (part of) a name refers to; None if none or several do."""
    if wanted.isdigit():
        return found[int(wanted) - 1] if 0 < int(wanted) <= len(found) else None
    exact = [model for model in found if model.name.lower() == wanted.lower()]
    partial = [model for model in found if wanted.lower() in model.label.lower()]
    return exact[0] if exact else partial[0] if len(partial) == 1 else None


# ---------- fitting the prompt ----------

def fit(ctx, question, history, limit, tokens, project="", ui=PLAIN, vision=False):
    """Trim ctx (and, as a last resort, the earlier exchanges, oldest first) until the prompt fits.

    tokens counts a list of messages; vision: whether the model is sent the images of the code.
    Returns (messages, tokens, history), or None when the question alone is too long.
    """
    while True:
        messages = build(ctx, question, history, project, vision)
        used = tokens(messages) + MARGIN
        if used <= limit:
            return messages, used, history
        if len(ctx) > 1:
            ui.note(f"dropped {label(ctx.pop(0))}")
        elif ctx:
            path, start, end = ctx[0]
            total = len(text_lines(path))  # none for an image, which is dropped rather than cut
            end = min(end or total, total)
            keep = int((end - start + 1) * limit / used * 0.9)
            if keep < 1:
                ui.note(f"dropped {label(ctx.pop())}")
            else:
                ctx[0][2] = start + keep - 1
                ui.note(f"{path} does not fit; sending lines {start}-{ctx[0][2]}")
        elif history:
            history = history[1:]
            ui.note("dropped an earlier exchange")
        else:
            return None


# ---------- reading the answer: file requests and edits ----------

def requested(answer, files=None):
    """Context entries for the project files named on NEED: lines of an answer, in the order named.

    A name may end in :START-END. A directory stands for the files in it; a bare
    file name matches anywhere in the project. files are the project's readable files,
    when they have been listed already.
    """
    names = " ".join(re.findall(r"^NEED:(.*)$", answer, re.M)).split()
    project, entries = ((readable_files() if files is None else files) if names else []), []
    for name in names:
        name, start, end = split_range(name.strip("`'\","))
        name = local(norm(name)).lower()
        for path in project:
            if ("/" + path.lower()).endswith("/" + name) or path.lower().startswith(name + "/"):
                if [path, start, end] not in entries:
                    entries.append([path, start, end])
    return entries


def local(path):
    """path without a first folder named like the project, when the project has no such folder.

    Models take the prompt's "Project folder: NAME" line for the start of the paths.
    """
    first, _, rest = path.partition("/")
    return rest if rest and first == os.path.basename(os.getcwd()) and not os.path.isdir(first) else path


def inside(path):
    """Whether path is in the project folder."""
    return os.path.normcase(os.path.realpath(path)).startswith(os.path.normcase(os.path.realpath(".")) + os.sep)


def block_path(line):
    """The file path on the line above a SEARCH/REPLACE block, without its FILE: label or markdown marks.

    "" when the line names no real path (it is empty or still holds the <path> stand-in).
    """
    path = LABEL.sub("", line.strip(" `*#")).strip(" `*:")
    return "" if not path or "<" in path or ">" in path else local(norm(path))


def edit_blocks(answer):
    """[(match, path)] for each SEARCH/REPLACE block in an answer.

    A block with no path line above it, or only a code fence, is for the file of the block before.
    """
    found, path = [], ""
    for block in BLOCK.finditer(answer):
        line = block[1]
        if line is not None and line.strip(" `~"):
            path = block_path(line)
        found.append((block, path))
    return found


def prose(answer):
    """An answer without its edit blocks. tolken's own lines (NEED:, RUN:, DELETE:, CONTINUE, DONE) are looked
    for in it, so that a line of a file's new text, such as a continue statement, is not taken for one."""
    return BLOCK.sub("", answer)


def parse_edits(answer):
    """[(path, search, replace)] for each SEARCH/REPLACE block in an answer."""
    return [(path, block[2].removesuffix("\n"), block[3].removesuffix("\n")) for block, path in edit_blocks(answer)]


def unread_blocks(answer):
    """What each <<<<<<< SEARCH line of an answer that began no block lacks: the blocks edit_blocks() does not find."""
    read = [block.span() for block, _ in edit_blocks(answer)]
    starts = [mark.start() for mark in re.finditer(r"^<{5,9} SEARCH", answer, re.M)]
    lacks = []
    for start, end in zip(starts, starts[1:] + [len(answer)]):
        if not any(first <= start < last for first, last in read):
            text = answer[start:end]  # up to the next block
            lacks.append("text after <<<<<<< SEARCH on its line" if not re.match(r"<{5,9} SEARCH[ \t]*\n", text)
                         else "no >>>>>>> REPLACE line at its end" if re.search(r"^={5,9}[ \t]*$", text, re.M)
                         else "no ======= line between the old and the new lines")
    return lacks


def places(lines, old_lines, key):
    """Where old_lines start in lines, compared after key: str, or str.strip to ignore indentation."""
    want = [key(line) for line in old_lines]
    return [i for i in range(len(lines) - len(want) + 1) if [key(line) for line in lines[i:i + len(want)]] == want]


def indent(line):
    return line[:len(line) - len(line.lstrip())]


def reindent(old_lines, found, new_lines):
    """new_lines moved by as much as the lines found are indented more (or less) than old_lines.

    Only when new_lines begin at old_lines' indentation, so a REPLACE already indented right is kept.
    """
    first = next((i for i, line in enumerate(old_lines) if line.strip()), None)
    start = next((line for line in new_lines if line.strip()), None)
    if first is None or start is None or indent(start) != indent(old_lines[first]):
        return new_lines
    have, want = indent(old_lines[first]), indent(found[first])
    if want != have and want.endswith(have):
        return [want[:len(want) - len(have)] + line if line.strip() else line for line in new_lines]
    if want != have and have.endswith(want):
        extra = have[:len(have) - len(want)]
        return [line.removeprefix(extra) for line in new_lines]
    return new_lines


def apply_edit(text, old, new):
    """text with old replaced by new, or None unless old matches exactly one place.

    Tries whole lines exactly, then whole lines ignoring indentation (new is then
    indented as the lines found are), then a unique substring.
    """
    old_lines, lines = old.split("\n"), text.split("\n")
    new_lines = new.split("\n") if new else []
    for key in (str, str.strip):
        spots = places(lines, old_lines, key)
        if len(spots) == 1:
            found = lines[spots[0]:spots[0] + len(old_lines)]
            if key is str.strip:
                new_lines = reindent(old_lines, found, new_lines)
            return "\n".join(lines[:spots[0]] + new_lines + lines[spots[0] + len(old_lines):])
        if spots:
            return None
    return text.replace(old, new) if text.count(old) == 1 else None


def edit_problem(text, old):
    """Why apply_edit cannot replace old in text: it matches several places, or none (then the nearest lines)."""
    lines, old_lines = text.split("\n"), old.split("\n")
    count = max(len(places(lines, old_lines, str)), len(places(lines, old_lines, str.strip)), text.count(old))
    if count > 1:
        return f"its SEARCH text matches {count} places; add lines around it so that it matches one"
    first = next((i for i, line in enumerate(old_lines) if line.strip()), None)
    stripped = [line.strip() for line in lines]
    close = difflib.get_close_matches(old_lines[first].strip(), stripped, 1, 0.6) if first is not None else []
    if not close:
        return "its SEARCH text is not in the file"
    start = max(0, stripped.index(close[0]) - first - 2)
    nearest = lines[start:start + min(len(old_lines) + 4, 30)]
    return (f"its SEARCH text is not in the file; the nearest lines ({start + 1}-{start + len(nearest)}) are:\n"
            + "\n".join(nearest))


def verdict(answer):
    """A ui's answer to a question: True (yes), False (no), or the user's comment (a no, with what to do instead)."""
    if isinstance(answer, str):
        return answer.strip() or False
    return bool(answer)


def refusal(note, comment):
    """The note about a change or command the user said no to, with what they said instead, if anything."""
    return f"{note}. Instead, the user said: {comment}" if comment else note


def approved(ui, path, old, new, auto):
    """Whether to make a change: the user's answer (see verdict), or with auto (the "auto" mode) True, once shown."""
    if auto:
        ui.show(path, old, new)
        return True
    return verdict(ui.confirm(path, old, new))


def edit_file(path, blocks, ui=PLAIN, auto=False):
    """Apply (search, replace) blocks to one file after showing the diff and asking (with auto, not asking).

    Returns (done, problems): done is True when applied, False when the user
    declined, the user's comment when they declined with one, None when there
    was nothing to ask about; problems are the notes about blocks that could
    not be applied.
    """
    problems = []

    def problem(text):
        ui.note(text, "bad")
        problems.append(text)

    exists = os.path.isfile(path)
    try:
        raw = read_text(path) if exists else ""
    except UnicodeDecodeError:
        problem(f"could not edit {path}: it is not UTF-8")
        return None, problems
    old = new = raw.replace("\r\n", "\n")
    for search_text, replace in blocks:
        if exists and search_text:
            result = apply_edit(new, search_text, replace)
            if result is None:
                problem(f"could not apply an edit to {path}: {edit_problem(new, search_text)}")
        elif not exists and not search_text:
            result = replace + "\n"
        else:
            result = None
            reason = "the file already exists, and an empty SEARCH only creates new files" if exists \
                else "there is no such file"
            problem(f"could not apply an edit to {path}: {reason}")
        if result is not None:
            new = result
    if new == old:
        return None, problems
    answer = approved(ui, path, old, new, auto)
    if answer is True:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(new.replace("\n", "\r\n") if "\r\n" in raw else new)
        ui.note(f"applied {path}", "good")
        return True, problems
    ui.note(f"skipped {path}")
    return answer, problems


def raw_bytes(path):
    """The bytes of a file, or None when there is no such file."""
    try:
        with open(path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        return None


def missing_folders(path):
    """The folders above path that do not exist yet, deepest first: those that writing it creates."""
    folders, folder = [], os.path.dirname(path)
    while folder and not os.path.isdir(folder):
        folders.append(folder)
        folder = os.path.dirname(folder)
    return folders


def deletions(answer):
    """The paths named on DELETE: lines of an answer, in the order named."""
    paths = []
    for line in re.findall(r"^DELETE:(.*)$", answer, re.M):
        for name in line.split():
            path = block_path(name.strip(","))
            if path and path not in paths:
                paths.append(path)
    return paths


def delete_file(path, ui=PLAIN, auto=False):
    """Delete one file of the project after showing it and asking (with auto, not asking).

    Returns (done, problems) as edit_file does.
    """
    if not inside(path):
        problem = f"could not delete {path}: it is outside the project folder"
    elif not os.path.isfile(path):
        problem = f"could not delete {path}: there is no such file"
    else:
        answer = approved(ui, path, read_text(path, "replace").replace("\r\n", "\n"), None, auto)
        if answer is True:
            os.remove(path)
            ui.note(f"deleted {path}", "good")
            return True, []
        ui.note(f"kept {path}")
        return answer, []
    ui.note(problem, "bad")
    return None, [problem]


# ---------- running commands ----------

def commands(answer):
    """The commands on RUN: lines of an answer, in the order written."""
    found = (line.strip().strip("`").strip() for line in RUN_LINE.findall(answer))
    return [command for command in found if command]


def listed(command):
    """Whether the settings allow command: its words begin with the words of an entry of RUN."""
    words = command.split()
    return any(entry.split() and words[:len(entry.split())] == entry.split() for entry in RUN)


def isolate_commands():
    """In tolken.exe, keep what PyInstaller's start set up for tolken itself from reaching the commands it runs.

    The exe is unpacked to a temporary folder, from which Windows is told to load DLLs first
    (SetDllDirectory); a command, and git, would inherit that, and could load tolken's copy of a DLL
    in place of its own. tolken's own modules, which load theirs as they are imported (PyMuPDF once
    a PDF is shown), still find them in the folder through AddDllDirectory, which is not inherited.
    The _PYI_ variables go too, so that a tolken.exe a command starts unpacks itself rather than
    using this one's folder, which is deleted when this one ends.
    """
    global DLL_FOLDER
    if not getattr(sys, "frozen", False):
        return
    for name in [name for name in os.environ if name.startswith("_PYI_")]:
        del os.environ[name]
    if os.name == "nt":
        import ctypes
        DLL_FOLDER = os.add_dll_directory(sys._MEIPASS)  # kept: closing it would take the folder away
        ctypes.windll.kernel32.SetDllDirectoryW(None)


def kill_tree(process):
    """End a command and the processes it started, which would otherwise hold its output open."""
    try:
        if os.name == "nt":  # while the command still runs: the processes it started are found through it
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], stdin=subprocess.DEVNULL,
                           capture_output=True)
        else:
            os.killpg(process.pid, signal.SIGKILL)  # the group run_command started it as
    except OSError:
        pass
    process.kill()


def run_command(command, session=None):
    """Run command in the project folder, without a shell; return (exit code, seconds, output).

    The exit code is None when it was stopped after RUN_TIMEOUT seconds. The process is
    kept on the session while it runs, so that Session.cancel() can end it. Raises OSError
    when the command cannot start (no such program, or broken quotes).
    """
    started = time.monotonic()
    try:  # on Windows the program itself splits its command line, as it would in a terminal
        process = subprocess.Popen(command if os.name == "nt" else shlex.split(command), stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   start_new_session=os.name != "nt")  # a group of its own, for kill_tree()
    except ValueError as error:
        raise OSError(error) from error
    if session:
        session.process = process
    try:
        out, _ = process.communicate(timeout=RUN_TIMEOUT)
        code = process.returncode
    except subprocess.TimeoutExpired:
        kill_tree(process)
        out, _ = process.communicate()
        code = None
    except KeyboardInterrupt:  # Ctrl+C at the plain prompt
        kill_tree(process)
        raise
    finally:
        if session:
            session.process = None
    return code, time.monotonic() - started, out.decode("utf-8", "replace")


def clip(text, budget):
    """text cut to about budget characters: the start (a fifth of them) and the end, where tests sum up."""
    if len(text) <= budget:
        return text
    start, end = text[:budget // 5], text[-(budget - budget // 5):]
    start, end = start.rpartition("\n")[0] or start, end.partition("\n")[2] or end  # whole lines
    return f"{start}\n⋮ {text.count(chr(10)) - start.count(chr(10)) - end.count(chr(10))} lines left out ⋮\n{end}"


# ---------- the conversation ----------

class Session:
    """The conversation state and the per-message loop, independent of how it is shown."""

    def __init__(self, ui=PLAIN, size=8192, name=""):
        self.ui = ui
        self.model = Llama(size, name)  # every session starts on llama-server's model; use() switches
        self.sock, self.stopped = None, False  # the running call's socket; set by cancel()
        self.process = None  # the running command; cancel() ends it
        self.approve = APPROVE  # "ask" or "auto"; Shift+Tab in the full-screen interface switches it
        self.cut = self.thought = False  # whether the last answer hit max_out, and whether the model thought
        self.overthought = False  # whether the last answer's thinking reached think_max
        self.thinking = ""  # the last answer's thinking, which stream() leaves here
        self.journal = None  # this session's file in LOGS, named at its first call; False once it could not be written
        self.untracked = True  # whether git listed the untracked files at the last message; see git_files()
        self.clear()

    @property
    def size(self):
        """The context window of the model in use, in tokens."""
        return self.model.size

    @property
    def think_max(self):
        """The settings' cap on thinking, read by stream()."""
        return THINK_MAX

    def use(self, model):
        """Switch to another model, keeping the conversation. False (and no switch) if it cannot be loaded."""
        if model.label != self.model.label:
            self.model.leave()  # first, so that the new model finds the GPU memory free
        try:
            model.enter()
        except (OSError, KeyError, ValueError) as error:
            self.ui.note(f"could not load {model.label}: {explain(error)}", "bad")
            return False
        if model.label != self.model.label:
            self.log = []  # the new model has none of it cached: its first call builds the prompt afresh
        self.model = model
        self.ui.note(f"model: {model.label} · {model.size:,} tokens")
        return True

    def configure(self, name, value):
        """Change one setting, in settings.json and for this session (see save_setting, which may raise ValueError)."""
        save_setting(name, value)
        if name == "approve":
            self.approve = value
        elif name in ("run", "instructions"):
            self.log = []  # the system prompt has changed: the next call builds the prompt afresh

    def keep(self, **record):
        """Add one line to this session's log file. The first line begins the file, and the oldest files beyond
        KEEP_LOGS are deleted then. A log that cannot be written is given up, so that it never ends a message."""
        if not LOGS or self.journal is False:
            return
        try:
            if not self.journal:
                os.makedirs(LOGS, exist_ok=True)
                for old in sorted(name for name in os.listdir(LOGS) if name.endswith(".jsonl"))[:1 - KEEP_LOGS]:
                    os.remove(os.path.join(LOGS, old))
                self.journal = os.path.join(LOGS, f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}.jsonl")
            # Letters as they are; half a surrogate pair, which UTF-8 cannot hold, as the escape JSON has for it.
            with open(self.journal, "a", encoding="utf-8", errors="backslashreplace", newline="") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as error:
            self.journal = False
            self.ui.note(f"could not write the log in {LOGS}: {error.strerror or error}", "warn")

    def clear(self):
        """Forget the loaded code and the conversation."""
        self.ctx, self.pins = [], []  # ctx: code kept between messages; pins: the part asked for by name
        self.history = []  # (message, answer) of each message so far; during a message, the last is its own
        # The prompt of the last call and its answer, which the next call continues while it fits, so that
        # the server reads only what is new; logged: the code in it, as shown; listed: its file list.
        self.log, self.logged, self.listed = [], {}, set()
        self.untold = []  # what happened after an answer that the model has not been told yet
        self.finished = []  # the files whose last edits all applied, in the order they were made; see apply()
        # For /undo: path -> [bytes before (None: it was created), bytes after (None: deleted), folders created],
        # for the last message that changed files; fresh: the running message has changed none yet.
        self.saved, self.fresh = {}, True
        self.suggestion = ""  # the user's likely next message, where how the last one ended makes it plain; see send()

    def remember(self, path, before, folders):
        """Keep what path held before this message first changed it, and what it holds now, for /undo."""
        if self.fresh:
            self.saved, self.fresh = {}, False
        self.saved.setdefault(path, [before, None, folders])[1] = raw_bytes(path)

    def undo(self):
        """Put back the files that the last message to change files changed, unless they changed since.

        Returns notes about what was done; the model is told with the next message.
        """
        if not self.saved:
            return ["nothing to undo"]
        notes, undone = [], []
        for path, (before, after, folders) in self.saved.items():
            if before == after:  # created and deleted again: nothing to put back
                continue
            if raw_bytes(path) != after:
                notes.append(f"kept {path}: it has changed since")
                continue
            if before is None:
                os.remove(path)
                for folder in folders:
                    if os.path.isdir(folder) and not os.listdir(folder):
                        os.rmdir(folder)
                notes.append(f"removed {path}, which was new")
            else:
                with open(path, "wb") as f:
                    f.write(before)
                notes.append(f"put back {path}" if after is not None else f"restored {path}, which was deleted")
            undone.append(path)
        self.saved, self.suggestion = {}, ""
        if undone:
            self.untold.append(f"the user undid your changes to: {', '.join(undone)}")
        return notes

    def compose(self, shown, prompt, history, project, files, news):
        """(messages, tokens, what code they bring) for the next call; None if the message cannot fit.

        The conversation so far is continued with a message holding news (a line about added and
        removed files), the code in shown that it does not have as it is now (of a file it has an
        older copy of, only the changed parts), and the prompt, when that fits. Otherwise the prompt is built afresh from the file list, shown (which fit() may
        trim), the history and the prompt.
        """
        vision = getattr(self.model, "vision", False)
        texts = [snippet(entry) for entry in shown]
        new = [(entry, text) for entry, text in zip(shown, texts) if self.logged.get(tuple(entry)) != text]
        if self.log:
            parts = [changed_parts(entry, self.logged.get(tuple(entry)), text) for entry, text in new]
            content = "\n\n".join([news] * bool(news) + parts + [prompt])
            message = {"role": "user", "content": content}
            pictures = [picture for entry, _ in new for picture in images_of(entry)] if vision else []
            if pictures:
                message["images"] = pictures
            messages = self.log + [message]
            used = self.model.count(messages) + MARGIN
            if used <= self.size - answer_room(self.size):
                self.logged.update((tuple(entry), text) for entry, text in new)
                return messages, used, "+ " + ", ".join(label(entry) for entry, _ in new) if new else "no new code"
        limit = self.size - answer_room(self.size)
        fitted = fit(shown, prompt, history, limit, self.model.count, project, self.ui, vision)
        if not fitted:
            return None
        messages, used, _ = fitted
        self.logged, self.listed = {tuple(entry): snippet(entry) for entry in shown}, set(files)
        return messages, used, ("rebuilt: " if self.log else "") + (", ".join(map(label, shown)) or "no code")

    def suggest(self, call=None):
        """The user's likely next message, asked of the model in a short call outside the conversation; "" for none.

        The call continues the last one's prompt, which the server has cached, with ASK_NEXT, and
        is ended after the first line of the answer. Nothing of it is kept in the conversation.
        call is a Call through which another thread can end it (the user has begun to type).
        """
        call = call or Call()
        messages = self.log + [{"role": "user", "content": ASK_NEXT}]
        try:
            if not self.log or self.model.count(messages) + MARGIN > self.size - answer_room(self.size):
                return ""
            answer = self.model.chat(messages, Brief(call), call, think=False)
        except (OSError, ValueError):  # the server cannot be reached, or sent what could not be read
            return ""
        line = (answer.strip().splitlines() or [""])[0].strip(" \t\"'`*")
        return "" if not line or len(line) > 80 or NOT_A_MESSAGE.match(line) else line

    def cancel(self):
        """End the running call or command. Meant to be called from another thread; send() then stops."""
        self.stopped = True
        process = self.process
        if process:
            try:
                kill_tree(process)
            except OSError:
                pass
        sock = self.sock
        if sock:
            # shutdown() wakes a blocked read on Linux; on Windows only closing the handle does.
            try:
                sock.shutdown(socket.SHUT_RDWR)
                socket.close(sock.detach())
            except OSError:
                pass

    def send(self, question):
        """Answer one message: build the prompt, call the model, load the files it asks for, apply its edits.

        The model is called again while it asks for files that are not shown, made
        changes without ending with DONE, ends with CONTINUE, an edit could not be
        applied, its answer was cut off, or the user said what to do instead of a
        change or command. It stops when the user just says no to one, the window
        is full and an answer changed nothing, two answers in a row neither changed
        a file nor brought in a new one, or after MAX_CALLS calls.

        Each call continues the conversation of the call before while it fits the
        window (see compose()); the budgets below are for a prompt built afresh.
        The file list and the earlier exchanges kept then grow with the window.
        """
        ui, size = self.ui, self.size
        self.stopped, self.fresh, self.suggestion = False, True, ""
        ui.activity("searching project")
        self.ctx = self.visible([entry for entry in self.ctx if os.path.isfile(entry[0])])
        for pin in self.visible(parse_pins(question)):
            if pin in self.ctx:
                self.ctx.remove(pin)
            self.ctx.append(pin)
            self.pins.append(pin)
        files, listing = readable_files(self), MAP_CHARS * size // 8192
        if self.stopped:  # cancelled while the project's files were listed
            ui.note("cancelled")
            return
        if self.untracked and not UNTRACKED:
            ui.note(f"git took over {GIT_SECONDS} s to list this project's files; from now on only the files it "
                    "tracks are listed, so a new file is seen once it is added to git", "warn")
        self.untracked = UNTRACKED
        project = project_map(listing, files)
        news = changes(self.listed, files) if self.log else ""
        if len(news) > listing:  # so much has changed that the list is sent again, in a prompt built afresh
            self.log, news = [], ""
        self.listed = set(files) if news else self.listed
        prompt = question  # what the next call asks: the question, or how the model's edits went
        if self.untold:  # an edit declined, a failed edit or a command's output the model has not seen
            prompt = "What happened after your last answer:\n" + "\n".join(self.untold) + "\n\n" + question
        mine = False  # whether the last exchange of the history is this message's

        def kept():
            """The earlier exchanges that a prompt built afresh carries."""
            return recent(self.history, MAX_ANSWER * size // 8192)

        def room():
            """Characters of code that fit beside the rules, the file list, the prompt and the kept exchanges."""
            ratio = getattr(self.model, "chars_per_token", 3)  # llama-server's models count exactly; 3 is safe
            return int((size - answer_room(size) - MARGIN) * ratio - len(RULES) - len(project) - len(prompt)
                    - sum(len(asked) + len(answered) for asked, answered in kept()))

        def record(answer):
            nonlocal mine
            if mine:
                self.history[-1] = (question, answer)
            else:
                self.history.append((question, answer))
                mine = True

        budget = room()
        if not self.ctx:
            found = search(question, min(budget, SEARCH_TOKENS * 3), files, self)
            if self.stopped:  # cancelled while the files were searched; what was found until then is not kept
                ui.note("cancelled")
                return
            self.ctx = found
        answer, full, idle = "", False, 0  # full: no room is left for more files; idle: answers in a row that did nothing
        more, progress = None, {}  # whether the model has more to do; the files this message changed so far
        problems, declined = [], []  # of the last answer: what could not be done, and what the user said no to
        for call in itertools.count(1):
            try:
                pinned = [entry for entry in self.ctx if entry in self.pins]
                shown = within(includes(pinned), budget - chars(self.ctx)) + self.ctx
                before = [entry[:] for entry in shown]
                composed = self.compose(shown, prompt, kept(), project, files, news if call == 1 else "")
                if not composed:
                    ui.note("the message is too long for the context window", "bad")
                    break
                messages, used, sent = composed
                self.ctx = [entry for entry in self.ctx if entry in shown]
                ui.note(f"{sent} | {used}/{size} tokens")
                ui.activity("waiting for model", call, used)
                self.cut = self.thought = self.overthought = False  # set by stream() while the answer arrives
                self.thinking = ""
                answer = self.model.chat(messages, ui, self)
                if (self.overthought or self.cut and self.thought) and not self.stopped:
                    # Thinking took the room the answer needed: the same prompt (cached) again, without it.
                    why = (f"the model thought for {THINK_MAX:,} tokens (think_max)" if self.overthought else
                           f"the answer reached max_out ({answer_room(size):,} tokens) while the model was thinking")
                    ui.note(f"{why}; asking again without thinking", "warn")
                    ui.activity("answering without thinking", call, used)
                    self.cut = self.overthought = False
                    answer = self.model.chat(messages, ui, self, think=False)
                    if self.overthought and not self.stopped:  # a model that thinks even when told not to
                        ui.note(f"the model thought for {THINK_MAX:,} tokens again and gave no answer; "
                                "raise think_max in settings.json", "bad")
                if self.cut and not self.stopped:
                    ui.note(f"the answer reached max_out ({answer_room(size):,} tokens) and was cut off", "warn")
                # Logged before anything is done with the answer: the messages this call added to the last one's.
                added = messages[len(self.log):] if self.log and messages[:len(self.log)] == self.log else messages
                self.keep(time=time.strftime("%Y-%m-%dT%H:%M:%S"), project=os.getcwd(), model=self.model.label,
                          call=call, whole=added is messages,
                          messages=[{**message, "images": len(message["images"])} if "images" in message else message
                                    for message in added],
                          thinking=self.thinking, answer=answer, cut=self.cut)
                self.log = messages + [{"role": "assistant", "content": answer}]
                self.untold = []  # the prompt told it
            except urllib.error.HTTPError as error:
                self.log = []  # the next call builds the prompt afresh
                ui.note(f"server error {explain(error)}", "bad")
                return
            except OSError as error:
                self.log = []
                if self.stopped:  # cancelled before the server began to answer
                    break
                if error.filename:  # a loaded file, not the server
                    ui.note(f"could not read {error.filename}: {error.strerror}", "bad")
                else:
                    ui.note(f"cannot reach {self.model.url} ({self.model.label}): {error}", "bad")
                return
            except ValueError as error:  # a line of the stream that is not JSON
                self.log = []
                ui.note(f"the server sent an answer that could not be read: {error}", "bad")
                return
            if self.stopped:
                break
            changed, problems, declined, comment = self.apply(answer)
            said = prose(answer)  # where its RUN:, NEED:, CONTINUE and DONE lines are looked for
            if self.cut:  # the model is called again for the rest
                problems.append(f"your answer was cut off at the length limit ({answer_room(size):,} tokens); the "
                                "complete edit blocks in it were handled. Give the rest in the next answer, a file "
                                "or two at a time.")
            # After the edits, which it may test; none after a change the user said no to.
            ran, refused, comment = ([], [], comment) if declined else self.run(said)
            declined += refused
            report = [f"{done} {path}" for path, done in changed.items()] + problems + ran + declined
            if report:
                self.keep(report=report)
            # Kept for the next message unless a call of this one tells it; only what the model cannot assume.
            self.untold = report if problems or ran or declined else []
            if self.stopped:
                break
            for path, done in changed.items():  # the results the model is told of keep the file list up to date
                (self.listed.discard if done == "deleted" else self.listed.add)(path)
            self.widen([path for path, done in changed.items() if done == "applied"])
            if changed:
                files = readable_files()  # with the files it created or deleted
            wanted = self.visible([entry for entry in requested(said, files) if entry not in self.ctx])
            more = CONTINUE.search(said) or bool(changed) and not DONE.search(said)
            idle = 0 if changed or wanted or comment else idle + 1
            crowded = full or shown != before  # the window is full: going on is worth it only while files change
            if call == MAX_CALLS or not comment and (declined or crowded and not changed or idle == 2
                                                     or not (wanted or more or problems or ran)):
                break  # the user stepped in, nothing is happening, it has gone on long enough, or the model is done
            if changed or problems or more or ran or comment:
                # Tell the model how its edits and commands went, and show it the files it changed or tried to.
                record(answer)
                earlier = [path for path in progress if path not in changed]
                prompt = "\n".join(report + [f"Changed earlier in this task: {', '.join(earlier)}."] * bool(earlier)
                                   + [GO_ON])
                progress.update(changed)
                budget = room()
                named = dict.fromkeys([*changed, *(path for path, _, _ in parse_edits(answer) if path)])
                wanted += [[path, 1, None] for path in named if os.path.isfile(path)
                           and not any(entry[0] == path for entry in self.ctx + wanted)]
            if wanted and chars(self.ctx) + chars(wanted) > budget:
                # No room for all of it: cut the requested files to fit and drop older code. From now on
                # an answer that changes nothing ends the message.
                ranges = [entry for entry in wanted if entry[2]]
                wanted = heads([entry[0] for entry in wanted if not entry[2]][::-1],
                               budget - chars(ranges)) + ranges
                if all(entry in self.ctx for entry in wanted):
                    break
                self.ctx = within([entry for entry in self.ctx if entry not in wanted], budget - chars(wanted))
                full = True
            self.ctx += wanted
            self.pins += wanted
        if answer:
            record(answer)
        if self.stopped:
            ui.note("cancelled")
            return
        wants = re.search(r"^NEED:", prose(answer), re.M)
        if wants:
            ui.note("the model still wants files; pin the part it needs with @file:START-END, "
                    "or reply to let it try again")
        elif more:
            ui.note(f"stopped after {MAX_CALLS} calls; reply to let the model go on" if call == MAX_CALLS
                    else "the model wanted to go on; reply to let it continue")
        # What the user will most likely say next, where that is plain from how the message ended.
        self.suggestion = ("continue" if more else "try again" if wants or problems and not declined
                           else "commit the changes" if not self.fresh and listed("git commit") else "")

    def widen(self, paths):
        """Show the files at paths, just edited, whole from now on: a line range of one was for its old lines.

        The whole file takes the place of its first range, keeping the order of the entries.
        """
        for path in paths:
            for entries in (self.ctx, self.pins):
                ranged = [entry for entry in entries if entry[0] == path and entry[1:] != [1, None]]
                if ranged:
                    at = entries.index(ranged[0])  # the entries before it are not ranges of the file
                    entries[:] = [entry for entry in entries if entry not in ranged]
                    if [path, 1, None] not in entries:
                        entries.insert(at, [path, 1, None])

    def visible(self, entries):
        """entries without the images, with a warning for each, when the model in use cannot see."""
        if getattr(self.model, "vision", False):
            return entries
        for entry in entries:
            if kind(entry[0]) == "image":
                self.ui.note(f"left out {entry[0]}: {self.model.label} cannot see images", "warn")
        return [entry for entry in entries if kind(entry[0]) != "image"]

    def run(self, answer):
        """Run the commands an answer asks for. In "auto" an allowed one runs at once; any other waits for the user.

        Returns (results, refused, comment): what to tell the model about each command, a note
        about the command the user refused, if any, and the user's answer to that one (see apply()).
        """
        ui, results = self.ui, []

        def tell(text, kind="warn"):
            ui.note(text, kind)
            results.append(text)

        for command in commands(answer):
            if SHELL.search(command):
                tell(f"could not run {command}: shell syntax (pipes, &, &&, ;, ^, %, redirection and "
                     "backquotes) cannot be used")
                continue
            if not (self.approve == "auto" and listed(command)):
                comment = verdict(ui.allow(command))
                if comment is not True:
                    ui.note(f"did not run {command}")
                    return results, [refusal(f"the user did not let {command} run", comment)], comment
            ui.activity(f"running {command}")
            try:
                code, seconds, output = run_command(command, self)
            except OSError as error:
                tell(f"could not run {command}: {error}")
                continue
            if self.stopped:
                break
            ui.ran(command, code, seconds, output)
            done = (f"stopped {command} after {RUN_TIMEOUT} s" if code is None
                    else f"ran {command}: exit {code} in {seconds:.1f} s")
            ui.note(done, "good" if code == 0 else "warn")
            results.append(f"{done}\n{clip(output, RUN_CHARS * self.size // 8192)}")
        return results, [], None

    def apply(self, answer):
        """Make the edits and deletions an answer asks for, each once the user confirms it.

        The first change the user says no to is the last one asked about: the rest
        of the answer may build on it, so the model gives them again.

        Returns (changed, problems, declined, comment): changed maps each file edited
        or deleted to "applied" or "deleted"; problems are the notes about what could
        not be done; declined are the notes about what the user said no to; comment
        is None when they said no to nothing, else False, or what they said instead.
        """
        ui, changed, problems, declined = self.ui, {}, [], []
        finished = []  # the files whose edits in this answer all applied
        comment, left = None, []  # the user's no, and the changes after it, which are not asked about

        def problem(text):
            ui.note(text, "bad")
            problems.append(text)

        edits = {}
        for path, search_text, replace in parse_edits(answer):
            edits.setdefault(path, []).append((search_text, replace))
        unread = unread_blocks(answer)
        if unread:
            lacks = "; ".join(f"{unread.count(lack)} with {lack}" for lack in dict.fromkeys(unread))
            text = f"{len(unread)} edit block(s) could not be read and were ignored: {lacks}"
            ui.note(text, "bad")
            problems.append(f"{text}. Write each change like this:\n{FORM}")  # the model is shown the form again
        removals = deletions(prose(answer))
        for path, blocks in edits.items():
            if comment is not None:
                left.append(path)
            elif not path:
                problem("could not apply an edit: the model did not say which file it is for")
            elif not inside(path):
                problem(f"could not edit {path}: it is outside the project folder")
            elif os.path.isfile(path) and not any(old or new for old, new in blocks):
                problem(f"could not apply an edit to {path}: an empty block changes nothing; "
                        f"to delete the file, write DELETE: {path}")
            else:
                try:
                    before = raw_bytes(path), missing_folders(path)
                    done, said = edit_file(path, blocks, ui, self.approve == "auto")
                except OSError as error:  # a folder, a read-only file, one that another program holds open
                    problem(f"could not edit {path}: {error.strerror or error}")
                    continue
                problems += said
                if done is True:
                    changed[path] = "applied"
                    self.remember(path, *before)
                    if not said:
                        finished.append(path)
                elif done is not None:
                    comment = done
                    declined.append(refusal(f"the user declined the edit to {path}; it was not applied", done))
        for path in dict.fromkeys(removals):
            if comment is not None:
                left.append(f"deleting {path}")
                continue
            try:
                before = raw_bytes(path) if os.path.isfile(path) else None
                done, said = delete_file(path, ui, self.approve == "auto")
            except OSError as error:
                problem(f"could not delete {path}: {error.strerror or error}")
                continue
            problems += said
            if done is True:
                changed[path] = "deleted"
                self.remember(path, before, [])
            elif done is not None:
                comment = done
                declined.append(refusal(f"the user kept {path}; it was not deleted", done))
        if left:
            declined.append(f"your later changes were not made either: {', '.join(left)}")
        # A deleted file can no longer be shown. The files whose edits all applied go first, in the order
        # they were finished, so that they are dropped first when the window is full: the files still to
        # be changed are worth more. A file this answer tried to edit is not finished unless it was again.
        self.finished = [path for path in self.finished if path not in edits] + finished
        order = {path: number for number, path in enumerate(self.finished)}
        kept = [entry for entry in self.ctx if os.path.isfile(entry[0])]
        self.ctx = (sorted((entry for entry in kept if entry[0] in order), key=lambda entry: order[entry[0]])
                    + [entry for entry in kept if entry[0] not in order])
        self.pins = [entry for entry in self.pins if os.path.isfile(entry[0])]
        return changed, problems, declined, comment


ASK_NEXT = ("Write the message the user is most likely to send next, in at most ten words. "
            "Reply with only that message.")  # what Session.suggest() asks
NOT_A_MESSAGE = re.compile(r"NEED:|RUN:|DELETE:|FILE:|DONE\b|CONTINUE\b|<{5}")  # tolken's own lines, not a user's


class Call:
    """What post() and stream() need of a session, for a call outside the conversation: where its socket is
    kept, and whether it was cancelled."""

    sock, stopped, process = None, False, None  # process: a command; such a call runs none
    cancel = Session.cancel  # sets stopped and closes the socket, which is all of a session it uses


class Brief:
    """The ui of a call that is to give one line: it ends the call once a line has come, or the model goes
    on for long without one, or keeps thinking although told not to."""

    def __init__(self, call):
        self.call, self.said, self.pieces = call, "", 0

    def text(self, piece):
        self.said, self.pieces = self.said + piece, self.pieces + 1
        if "\n" in self.said.lstrip() or self.pieces >= 40:
            self.call.stopped = True

    def thinking(self, piece):
        self.pieces += 1
        if self.pieces >= 200:
            self.call.stopped = True

    def timings(self, data):
        pass

    def note(self, text, kind="info"):
        pass


# ---------- settings and startup ----------

def strings(value):
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


# What each setting must be: (a test of its value, the words for it in the error).
KINDS = {
    "servers": (lambda value: isinstance(value, str) or strings(value), "a URL or a list of URLs"),
    "model": (lambda value: isinstance(value, str), "text"),
    "context": (lambda value: value is None or isinstance(value, int), "a whole number or null"),
    "think": (lambda value: isinstance(value, (bool, str)), 'true, false or a level such as "low"'),
    "think_max": (lambda value: value is None or isinstance(value, int), "a whole number or null"),
    "max_out": (lambda value: isinstance(value, int), "a whole number"),
    "options": (lambda value: isinstance(value, dict), "an object: {...}"),
    "approve": (lambda value: isinstance(value, str), '"ask" or "auto"'),
    "run": (strings, "a list of commands"),
    "instructions": (lambda value: isinstance(value, str), "text"),
    "suggest": (lambda value: value is False or value in ("rules", "model"), 'false, "rules" or "model"'),
}


def current_settings():
    """The settings in use, by the names they have in the file."""
    return {"servers": SERVERS, "model": MODEL, "context": CONTEXT, "think": THINK, "think_max": THINK_MAX,
            "max_out": MAX_OUT, "options": OPTIONS, "approve": APPROVE, "run": RUN, "instructions": INSTRUCTIONS,
            "suggest": SUGGEST}


def apply_settings(settings):
    """Put these settings in use: all of them, each of the kind it must be."""
    global SERVERS, MODEL, CONTEXT, THINK, THINK_MAX, MAX_OUT, OPTIONS, APPROVE, RUN, INSTRUCTIONS, SUGGEST, RULES
    servers = settings["servers"]
    SERVERS = [url.rstrip("/") for url in ([servers] if isinstance(servers, str) else servers)]
    MODEL, CONTEXT, THINK, THINK_MAX = settings["model"], settings["context"], settings["think"], settings["think_max"]
    MAX_OUT, OPTIONS, APPROVE = settings["max_out"], settings["options"], settings["approve"]
    RUN, INSTRUCTIONS, SUGGEST = settings["run"], settings["instructions"], settings["suggest"]
    RULES = rules()  # so the model knows the commands allowed and the user's instructions


def save_setting(name, value):
    """Change one setting: in the settings file, which keeps whatever else it holds, and in use.

    Raises ValueError when value is not what the setting must be, or the file is no longer
    valid JSON (it is then left as it is).
    """
    fits, kind = KINDS[name]
    if not fits(value):
        raise ValueError(f'"{name}" must be {kind}')
    saved = {}
    if os.path.isfile(SETTINGS):
        try:
            saved = json.loads(read_text(SETTINGS))
        except ValueError as error:  # not JSON, or not UTF-8
            raise ValueError(f"{SETTINGS}: {error}") from error
        if not isinstance(saved, dict):
            raise ValueError(f"{SETTINGS}: the settings must be one {{...}} object")
    with open(SETTINGS, "w", encoding="utf-8") as f:
        f.write(json.dumps({**saved, name: value}, indent=2, ensure_ascii=False) + "\n")  # letters as they are
    apply_settings({**current_settings(), name: value})


def load_settings(path=SETTINGS):
    """Replace the default settings with those of the file, which is written with the defaults if missing.

    Exits with the error when the file is not valid JSON, or a setting is of the wrong kind.
    """
    defaults = current_settings()
    if not os.path.isfile(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(defaults, indent=2) + "\n")
        print(f"[wrote the default settings to {path}]")
    try:
        settings = json.loads(read_text(path))
    except ValueError as error:  # not JSON, or not UTF-8
        sys.exit(f"{path}: {error}")
    if not isinstance(settings, dict):
        sys.exit(f"{path}: the settings must be one {{...}} object")
    unknown = sorted(settings.keys() - defaults.keys())
    if unknown:
        print(f"[{path}: unknown settings ignored: {', '.join(unknown)}]")
    settings = {**defaults, **settings}
    for name, (fits, kind) in KINDS.items():
        if not fits(settings[name]):
            sys.exit(f'{path}: "{name}" must be {kind}')
    apply_settings(settings)


def startup():
    """The model to start with, named as for /model: the one the settings name, else the first listed.

    The servers are asked only when the settings name no model; "" when none of them answers.
    """
    if MODEL:
        return MODEL
    found = models()
    if not found:
        print("[no model server answers]")
    return found[0].name if found else ""


# ---------- plain prompt ----------

def read_message():
    """One message from the terminal; a line of three quotes starts and ends a multi-line one."""
    line = input("> ").strip()
    if line != '"""':
        return line
    lines = []
    while (line := input()).strip() != '"""':
        lines.append(line)
    return "\n".join(lines)


def choose_model(session, wanted):
    """The plain prompt's /model: list the models, or switch to the one a number or name refers to."""
    found = models()
    if not wanted:
        for number, model in enumerate(found, 1):
            current = "  (current)" if model.label == session.model.label else ""
            print(f"{number}. {model.label} · {model.detail}{current}")
        print("[/model NUMBER or /model NAME switches]" if found else "[no model server answers]")
    elif model := pick(found, wanted):
        session.use(model)
    else:
        print(f"[no single model matches {wanted!r}]")


def main():
    sys.stdout.reconfigure(errors="replace")
    session = Session(PLAIN)
    start = startup()
    if start:
        choose_model(session, start)
    print(f"{session.model.url} | {session.size} tokens | approve: {session.approve} | "
          "@file or @file:10-50 pins code, /model, /undo, /clear, /exit")
    while True:
        try:
            question = read_message()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question == "/exit":
            return
        if question == "/clear":
            session.clear()
        elif question == "/undo":
            for note in session.undo():
                print(f"[{note}]")
        elif question.split()[:1] == ["/model"]:
            choose_model(session, question[6:].strip())
        elif question:
            try:
                session.send(question)
            except KeyboardInterrupt:  # Ctrl+C outside a streamed answer: before it began, or while a command ran
                session.log = []  # the next call builds the prompt afresh
                print("\n[stopped]")


if __name__ == "__main__":
    import tolken  # this file again, as the module tui.py imports, so that both see the settings loaded here
    tolken.isolate_commands()
    tolken.load_settings()
    if "--plain" in sys.argv or not (sys.stdin.isatty() and sys.stdout.isatty()):
        tolken.main()
    else:
        try:
            import tui
        except ImportError:
            print("[Textual is not installed (pip install textual); using the plain prompt]")
            tolken.main()
        else:
            tui.run()
