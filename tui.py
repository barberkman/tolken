"""Full-screen interface for tolken.py, built with Textual.

`python tolken.py` starts it when run in a terminal. Enter sends a message;
Shift+Enter starts a new line (on Windows, or where the terminal reports that
key), and Ctrl+J does so everywhere.
Esc cancels the running one, PageUp/PageDown scroll, Ctrl+Q or /exit quits.
Ctrl+C copies the text selected with the mouse; with nothing selected,
pressing it twice quits.
Up and Down bring back earlier messages.
Typing @ suggests project files: Up/Down choose, Tab or Enter inserts.
Typing / suggests the commands: Up/Down choose, Enter runs one, Tab completes it.
/model lists the models of llama-server and the Ollama servers; Enter switches to one.
The model named in settings.json is switched to at startup; without one, the
first listed (Ollama's come before llama-server's).
Thinking, when it is on, is shown dim above the answer. The answer is shown as
markdown (headings, lists, tables, highlighted code), each line break kept.
/undo puts back the files that the last message to change files changed.
An edit is shown as a card; Enter applies it, Esc skips it. A file the model
wants to delete is shown the same way; Enter deletes it, Esc keeps it. So is a
command it wants to run: Enter runs it, Esc refuses; its output then follows as
a card, and Esc stops a command that is running.
Shift+Tab switches between asking before every change ("ask") and making the
changes at once, still showing each card ("auto"); a card waiting for Enter is
then applied. The input box's lower border shows the mode; settings.json's
approve sets the one to start with.
A file dragged onto the window (its path pasted) is attached, as [file name]
in the input: a PDF or Word file goes to the model as its text, an image to a
model that can see. Alt+V attaches the files copied in Explorer, or an image
on the clipboard (saved to a temporary folder), and pastes text as usual.
A file whose full path is typed in single quotes is attached the same way:
that is what Terminator types for an image pasted with Ctrl+V, or a file dropped.
While a message is answered, a random Middle-earth phrase is shown above the input,
with the tokens streamed so far and the speed of the current call.
"""
import asyncio
import collections
import difflib
import itertools
import os
import random
import re
import subprocess
import tempfile
import threading
import time

from markdown_it import MarkdownIt
from rich.text import Text
from textual import events
from textual.app import App
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.markup import escape
from textual.message import Message
from textual.widgets import Markdown, Static, TextArea

import tolken

PLACEHOLDER = "Do you remember the Shire, Mr. Frodo? · @file pins code · / for commands"
COMMANDS = {  # what typing / offers, in this order
    "/model": "choose the model: llama-server's or one of Ollama's",
    "/undo": "put back the files that the last change of files changed",
    "/clear": "forget the loaded code and the conversation",
    "/exit": "quit",
}
EDIT_KEYS = "enter apply · esc skip"
DELETE_KEYS = "enter delete · esc keep"
RUN_KEYS = "enter run · esc refuse"
OUTPUT_LINES = 12  # lines of a command's output shown in its card (the model gets more)
NEWLINE_KEYS = ("shift+enter", "ctrl+j")  # Shift+Enter only where the terminal reports it; Ctrl+J everywhere
PHRASES = (  # shown above the input while the model works, one picked at random per message
    "Consulting the Palantír…", "Riddling in the dark…", "Pondering the Red Book…", "Smoking pipe-weed on it…",
    "Entmooting…", "Lighting the beacons…", "Speaking friend and entering…", "Delving too greedily and too deep…",
    "Tracking like a Ranger…", "Seeking the Arkenstone…", "Saddling Shadowfax…", "Waking the Ents…",
    "Rekindling the forges…", "Writing in the Red Book…", "Translating from Elvish…", "Singing of great deeds…",
    "Having second breakfast…", "Nibbling on lembas…", "Walking into Mordor…", "Following the Road ever on…",
    "Wandering, but not lost…", "Asking the Eagles…", "Reading Thrór's map…", "Searching for moon-letters…",
    "Climbing the stairs of Cirith Ungol…", "Crossing the Misty Mountains…", "Holding council at Rivendell…",
    "Asking Tom Bombadil…", "Gathering the Fellowship…", "Sneaking past Smaug…", "Mustering the Rohirrim…",
    "Reforging Narsil…", "Polishing Sting…", "Hiding from the Eye…", "Arguing with Gollum…",
    "Taking the hobbits to Isengard…",
    "Counting tokens, my precious…", "Forging the One Token…", "One token to rule them all…",
    "Tokenizing the Shire…", "Detokenizing Elvish…", "Quantizing the Silmarils…", "Compressing the Silmarillion…",
    "Decoding the runes…", "Embedding Elvish runes…", "Attending to every token…", "Prompting the Ents…",
    "Raising the temperature of Mount Doom…", "Hallucinating a second breakfast…", "Unrolling the map of Middle-earth…",
)


def shift_held():
    """True while a Shift key is physically down (Windows only).

    Most terminals report Shift+Enter as a plain Enter, so on Windows the keyboard is asked directly.
    """
    if os.name != "nt":
        return False
    import ctypes
    return bool(ctypes.windll.user32.GetAsyncKeyState(0x10) & 0x8000)  # VK_SHIFT, "down now" bit
RED, GREEN, GREY = "#e06c75", "#98c379", "#777777"


class Display:
    """What tolken.Session reports to. Its methods are called from the worker thread."""

    def __init__(self, app):
        self.app = app

    def on_screen(self, function, *args):
        try:
            self.app.call_from_thread(function, *args)
        except Exception:  # the app is closing
            pass

    def note(self, text, kind="info"):
        self.on_screen(self.app.add, {"good": "✓ ", "bad": "✗ ", "warn": "⚠ "}.get(kind, "") + text, kind)

    def activity(self, text, call=None, used=None):
        self.on_screen(self.app.set_activity, text, call, used)

    def thinking(self, piece):
        self.app.thoughts.append(piece)  # shown by the app's timer

    def text(self, piece):
        self.app.pieces.append(piece)  # shown by the app's timer

    def timings(self, data):
        self.app.call_timings.append(data)

    def show(self, path, old, new):
        self.on_screen(self.app.show_change, path, old, new)

    def confirm(self, path, old, new):
        asked = threading.Event()
        self.on_screen(self.app.ask, path, old, new, asked)
        return self.answer(asked)

    def allow(self, command):
        asked = threading.Event()
        self.on_screen(self.app.ask_run, command, asked)
        return self.answer(asked)

    def ran(self, command, code, seconds, output):
        self.on_screen(self.app.show_run, command, code, seconds, output)

    def answer(self, asked):
        """Wait for the user to answer a question on screen (Enter or Esc); False if the app closes first."""
        while not asked.wait(0.2):
            if not self.app.is_running:
                return False
        return self.app.reply


def copy_windows(text):
    """Put text on the Windows clipboard through the Win32 API. False if the clipboard is busy."""
    import ctypes
    from ctypes import wintypes
    user32, kernel32 = ctypes.WinDLL("user32"), ctypes.WinDLL("kernel32")
    kernel32.GlobalAlloc.argtypes, kernel32.GlobalAlloc.restype = [wintypes.UINT, ctypes.c_size_t], wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes, kernel32.GlobalLock.restype = [wintypes.HGLOBAL], wintypes.LPVOID
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.SetClipboardData.argtypes, user32.SetClipboardData.restype = [wintypes.UINT, wintypes.HANDLE], wintypes.HANDLE
    data = text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-16-le") + b"\0\0"
    if not user32.OpenClipboard(None):
        return False
    try:
        user32.EmptyClipboard()
        memory = kernel32.GlobalAlloc(0x0002, len(data))  # GMEM_MOVEABLE
        ctypes.memmove(kernel32.GlobalLock(memory), data, len(data))
        kernel32.GlobalUnlock(memory)
        return bool(user32.SetClipboardData(13, memory))  # CF_UNICODETEXT
    finally:
        user32.CloseClipboard()


def read_clipboard():
    """What the Windows clipboard holds: ("files", paths), ("image", PNG bytes) or ("text", text).

    Files copied in Explorer come first, then an image (as PNG, or a bitmap made into one),
    then text. None when it holds none of these, the clipboard is busy, or not on Windows.
    """
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes
    user32, kernel32, shell32 = ctypes.WinDLL("user32"), ctypes.WinDLL("kernel32"), ctypes.WinDLL("shell32")
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
    user32.GetClipboardData.argtypes, user32.GetClipboardData.restype = [wintypes.UINT], wintypes.HANDLE
    user32.RegisterClipboardFormatW.argtypes, user32.RegisterClipboardFormatW.restype = [wintypes.LPCWSTR], wintypes.UINT
    kernel32.GlobalLock.argtypes, kernel32.GlobalLock.restype = [wintypes.HGLOBAL], wintypes.LPVOID
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalSize.argtypes, kernel32.GlobalSize.restype = [wintypes.HGLOBAL], ctypes.c_size_t
    shell32.DragQueryFileW.argtypes = [wintypes.HANDLE, wintypes.UINT, wintypes.LPWSTR, wintypes.UINT]
    shell32.DragQueryFileW.restype = wintypes.UINT

    def held(kind, read):
        """The clipboard's data of one kind, read from its locked memory by read(pointer, size)."""
        handle = user32.GetClipboardData(kind)
        pointer = kernel32.GlobalLock(handle)
        try:
            return read(pointer, kernel32.GlobalSize(handle))
        finally:
            kernel32.GlobalUnlock(handle)

    if not user32.OpenClipboard(None):
        return None
    try:
        png = user32.RegisterClipboardFormatW("PNG")
        if user32.IsClipboardFormatAvailable(15):  # CF_HDROP: files copied in Explorer
            handle, paths = user32.GetClipboardData(15), []
            for index in range(shell32.DragQueryFileW(handle, 0xFFFFFFFF, None, 0)):
                name = ctypes.create_unicode_buffer(shell32.DragQueryFileW(handle, index, None, 0) + 1)
                shell32.DragQueryFileW(handle, index, name, len(name))
                paths.append(name.value)
            return "files", paths
        if user32.IsClipboardFormatAvailable(png):
            return "image", held(png, ctypes.string_at)
        if user32.IsClipboardFormatAvailable(8):  # CF_DIB
            return "image", bitmap_png(held(8, ctypes.string_at))
        if user32.IsClipboardFormatAvailable(13):  # CF_UNICODETEXT
            return "text", held(13, lambda pointer, size: ctypes.wstring_at(pointer))
        return None
    finally:
        user32.CloseClipboard()


def bitmap_png(dib):
    """A clipboard bitmap (a device-independent bitmap: a .bmp file without its file header) as PNG bytes."""
    import struct
    import fitz
    header, bits, compression, colors = (struct.unpack_from("<I", dib, 0)[0], struct.unpack_from("<H", dib, 14)[0],
                                         struct.unpack_from("<I", dib, 16)[0], struct.unpack_from("<I", dib, 32)[0])
    colors = colors or (1 << bits if bits <= 8 else 0)
    masks = 12 if compression == 3 and header == 40 else 0  # BI_BITFIELDS keeps three colour masks after the header
    bmp = b"BM" + struct.pack("<IHHI", 14 + len(dib), 0, 0, 14 + header + masks + colors * 4) + dib
    return fitz.Pixmap(bmp).tobytes("png")


def dropped(text):
    """The files a drag and drop pasted: text that is nothing but paths of existing files, maybe quoted.

    [] for any other text.
    """
    whole = text.strip().strip('"')
    if whole and os.path.isfile(whole):  # one file, its path not quoted although it may hold spaces
        return [whole]
    paths = [quoted or bare for quoted, bare in re.findall(r'"([^"]+)"|(\S+)', text)]
    return paths if paths and all(os.path.isfile(path) for path in paths) else []


# An absolute Windows path in single quotes, '' standing for a quote, as PowerShell quotes it; maybe a space after.
TYPED_PATH = re.compile(r"'((?:[A-Za-z]:\\|\\\\)(?:[^'\n]|'')+)'( ?)")


def save_image(png):
    """Save an image from the clipboard as a PNG in a temporary folder; return its path."""
    folder = os.path.join(tempfile.gettempdir(), "tolken")
    os.makedirs(folder, exist_ok=True)
    stem = os.path.join(folder, time.strftime("clipboard-%Y%m%d-%H%M%S"))
    path = next(f"{stem}{suffix}.png" for suffix in itertools.chain([""], (f"-{n}" for n in itertools.count(2)))
                if not os.path.exists(f"{stem}{suffix}.png"))
    with open(path, "wb") as f:
        f.write(png)
    return path


def copy(text):
    """Put text on the system clipboard. False if no way to do it is available."""
    if os.name == "nt":
        return copy_windows(text)
    commands = (("clip.exe", "utf-16-le"), ("pbcopy", "utf-8"), ("wl-copy", "utf-8"),
                ("xclip -selection clipboard", "utf-8"))
    for command, encoding in commands:
        try:
            subprocess.run(command.split(), input=text.encode(encoding), check=True, timeout=5)
            return True
        except (OSError, subprocess.SubprocessError):
            pass
    return False


def diff_card(path, old, new):
    """Title and body of the card that shows how a file would change.

    The body has numbered lines: removed ones marked - and red, added ones + and green.
    With new as None the file is to be deleted: the card shows how it begins.
    """
    if new is None:
        lines, shown = old.splitlines(), 12
        width = len(str(max(len(lines), 1)))
        body = Text("\n".join(f"{number:>{width}} - {line}" for number, line in enumerate(lines[:shown], 1)), RED)
        if len(lines) > shown:
            body.append(f"\n{' ' * width} ⋮ {len(lines) - shown} more lines", GREY)
        return f"delete {path} · {len(lines)} lines", body
    before, after = old.splitlines(), new.splitlines()
    width = len(str(max(len(before), len(after), 1)))
    body, added, removed = Text(), 0, 0
    groups = difflib.SequenceMatcher(None, before, after, autojunk=False).get_grouped_opcodes(2)
    for index, group in enumerate(groups):
        if index:
            body.append(" " * width + " ⋮\n", GREY)
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                for number in range(j1, j2):
                    body.append(f"{number + 1:>{width}}   ", GREY)
                    body.append(after[number] + "\n")
            else:
                for number in range(i1, i2):
                    body.append(f"{number + 1:>{width}} - {before[number]}\n", RED)
                for number in range(j1, j2):
                    body.append(f"{number + 1:>{width}} + {after[number]}\n", GREEN)
                added, removed = added + j2 - j1, removed + i2 - i1
    body.rstrip()
    return f"{'edit' if old else 'new file'} {path} · +{added} −{removed}", body


def without_blocks(text, path):
    """text without the SEARCH/REPLACE blocks for path (and the code fence around each)."""
    for block, owner in reversed(tolken.edit_blocks(text)):
        if owner == path:
            head = re.sub(r"(^|\n)```[^\n]*\n\Z", r"\1", text[:block.start()])
            tail = re.sub(r"\A\n```[ \t]*(?=\n|\Z)", "", text[block.end():])
            text = head + tail
    return re.sub(r"\n{3,}", "\n\n", text)


PROTOCOL = re.compile(r"^(?:NEED:|RUN:|DELETE:|FILE:|CONTINUE\b|DONE\b)")  # tolken's lines in an answer
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def displayed(text):
    """An answer as the markdown to show: edit blocks outside code fences are put in one, so that
    their ======= and >>>>>>> lines do not show as a heading and quotes, and tolken's lines
    (NEED:, RUN:...) stand as paragraphs of their own.

    Only whole lines are changed, so the result grows as the answer streams in; a block still
    arriving is left in an open fence, which runs to the end.
    """
    lines, out, fence, block = text.split("\n"), [], None, False
    for line in lines[:-1]:  # the last line may still be arriving
        out.append(line)
        if block:
            if re.match(r"^>{5,9} REPLACE", line):
                out.append("~~~~")
                block = False
        elif fence:
            if re.match(rf"^ {{0,3}}{re.escape(fence[0])}{{{len(fence)},}}[ \t]*$", line):
                fence = None
        elif opening := FENCE.match(line):
            fence = opening[1]
        elif re.match(r"^<{5,9} SEARCH", line):
            out[-1:] = ["~~~~", line]  # ~~~~ is not closed by a ``` line inside the block
            block = True
        elif PROTOCOL.match(line):
            out.append("")
    return "\n".join(out + lines[-1:])


def parser():
    """The markdown parser for answers: without HTML, so that text such as Optional<User> is shown and
    not taken for a tag, and keeping each line break, as models often write plain lines that markdown
    would run into one paragraph."""
    md = MarkdownIt("gfm-like", {"html": False})

    def hard_breaks(state):
        for token in state.tokens:
            for child in token.children or ():
                if child.type == "softbreak":
                    child.type = "hardbreak"

    md.core.ruler.push("hard_breaks", hard_breaks)
    return md


class Answer(Markdown):
    """An answer shown as markdown while it streams in.

    show() can be called at any time; the widget catches up in its own task, one change at a
    time: it appends what is new, or shows it all again when text before the end changed.
    """

    def __init__(self):
        super().__init__(parser_factory=parser, classes="answer")
        self.text, self.changed = "", asyncio.Event()

    def on_mount(self):
        self.run_worker(self.follow())

    def show(self, text):
        self.text = text
        self.changed.set()

    async def follow(self):
        while await self.changed.wait():
            self.changed.clear()
            text, shown = self.text, self.source
            if text != shown:
                await (self.append(text[len(shown):]) if text.startswith(shown) else self.update(text))


def matching(files, word, limit=8):
    """The files whose path contains word, best first: file-name matches before folder matches."""
    word = word.lower().replace("\\", "/")

    def rank(path):
        name = os.path.basename(path).lower()
        return not name.startswith(word), word not in name, len(path), path

    return sorted((path for path in files if word in path.lower()), key=rank)[:limit]


class Prompt(TextArea):
    """The input box. Enter sends; Shift+Enter or Ctrl+J starts a new line.

    A multi-line paste is kept whole and shown as a short marker.
    """

    BINDINGS = [
        Binding("up", "choose(-1)", show=False),
        Binding("down", "choose(1)", show=False),
        Binding("tab", "complete", show=False),
    ]

    class Submitted(Message):
        """Enter was pressed and there was no file suggestion to pick."""

    def __init__(self, placeholder):
        super().__init__(placeholder=placeholder, highlight_cursor_line=False)
        self.pastes = {}  # marker in the input -> the pasted text

    @property
    def value(self):
        return self.text

    @value.setter
    def value(self, text):
        self.load_text(text)
        self.move_cursor(self.document.end)

    def line_before_cursor(self):
        row, column = self.cursor_location
        return self.document.get_line(row)[:column]

    async def _on_key(self, event):
        if event.key == "enter" or event.key in NEWLINE_KEYS:
            event.prevent_default()  # TextArea itself would add a line for every Enter
            event.stop()
            if event.key != "enter" or shift_held():
                self.insert("\n")
            elif not self.app.complete(enter=True):  # while a list is open, Enter picks from it first
                self.post_message(self.Submitted())

    def action_choose(self, step):
        """Up or Down: move the cursor, or the suggestions or history once it is on the first or last line."""
        line = self.wrapped_document.location_to_offset(self.cursor_location).y
        if self.app.matches or line == (0 if step < 0 else self.wrapped_document.height - 1):
            self.app.choose(step)
        elif step < 0:
            self.action_cursor_up()
        else:
            self.action_cursor_down()

    def action_complete(self):
        self.app.complete()

    def on_paste(self, event):
        event.stop()  # handled here; otherwise it would travel on to the app and come back a second time
        files, lines = dropped(event.text), event.text.splitlines()
        if files:  # what a drag and drop pastes: the files' paths
            event.prevent_default()
            for path in files:
                self.attach(path)
        elif len(lines) > 1:
            event.prevent_default()  # TextArea itself would insert all the lines
            marker = f"[pasted {len(lines)} lines #{len(self.pastes) + 1}]"
            self.pastes[marker] = event.text
            self.insert(marker)

    def on_text_area_changed(self, event):
        """Attach the file of a quoted path typed into the input: Terminator types one for a pasted image or a drop."""
        for match in TYPED_PATH.finditer(self.text):
            path = match[1].replace("''", "'")
            if os.path.isfile(path):
                start, end = (self.document.get_location_from_index(index) for index in match.span())
                self.replace(self.marker_for(path) + match[2], start, end)  # the space after it, if typed yet
                return  # replacing is a change too, which attaches the next path

    def attach(self, path):
        """Put a short marker for a file in the input; take() turns it into a pin of the file."""
        self.insert(self.marker_for(path) + " ")

    def marker_for(self, path):
        """A new marker for a file, [image name] or [file name], remembered with the file's path."""
        marker = f"[{'image' if tolken.kind(path) == 'image' else 'file'} {os.path.basename(path)}]"
        if marker in self.pastes:
            marker = f"{marker[:-1]} #{len(self.pastes) + 1}]"
        self.pastes[marker] = (os.path.abspath(path),)  # a tuple: a file, where a pasted text is a string
        return marker

    def take(self):
        """Empty the input; return the message as typed and with pasted text and attached files put back in."""
        typed = message = self.value.strip()
        for marker, text in self.pastes.items():
            if isinstance(text, tuple):
                message = message.replace(marker, f'@"{text[0]}"')
            else:
                message = message.replace(marker, "\n" + text.strip("\n") + "\n")
        self.value, self.pastes = "", {}
        return typed, message.strip()


class TolkenApp(App):
    TITLE = "tolken"
    CSS = """
    #header { height: 1; padding: 0 1; color: $text-muted; text-wrap: nowrap; text-overflow: ellipsis; }
    #conversation {
        height: 1fr; padding: 0 1;
        scrollbar-size-vertical: 1;
        scrollbar-color: #555555; scrollbar-color-hover: #777777; scrollbar-color-active: #777777;
        scrollbar-background: $background; scrollbar-background-hover: $background;
        scrollbar-background-active: $background;
    }
    #suggestions { display: none; height: auto; max-height: 8; padding: 0 2; }
    Prompt, Prompt:focus {
        height: auto; max-height: 10;
        border: round #555555; background: $background; background-tint: $foreground 0%;
    }
    Prompt:disabled { opacity: 1; border: round #aaaaaa; }
    Prompt { border-subtitle-color: $text-muted; }
    Prompt.auto { border-subtitle-color: #e5c07b; }
    #status { height: 1; padding: 0 1; color: $text-muted; }
    #flavor { display: none; height: 1; padding: 0 1; color: #e5c07b; text-style: italic; }
    .user { margin-top: 1; text-style: bold; }
    .info { color: $text-muted; }
    .thinking { color: $text-muted; text-style: italic; }
    .bad { color: #e06c75; }
    .warn { color: #e5c07b; }
    .card { border: round #555555; border-title-color: #aaaaaa; padding: 0 1; }
    #conversation Markdown { margin: 0; padding: 0; background: $background; }
    """
    BINDINGS = [
        Binding("escape", "cancel", "cancel", priority=True),
        Binding("pageup", "page(-1)", "scroll up", priority=True),
        Binding("pagedown", "page(1)", "scroll down", priority=True),
        Binding("ctrl+c", "copy_selection", show=False, priority=True),
        Binding("shift+tab", "cycle_approval", show=False, priority=True),
        Binding("alt+v", "paste_clipboard", show=False, priority=True),
        # Only acts while an edit waits for an answer (see check_action); otherwise Enter sends the message.
        Binding("enter", "accept", show=False, priority=True),
    ]

    def __init__(self, size, model=""):
        super().__init__()
        self.session = tolken.Session(Display(self), size, model)
        self.models = None  # the models on offer while the /model list is open
        self.commands = None  # the command names on offer while the / list is open
        self.loading = False  # models are being listed or one is being loaded; Esc cannot cancel that
        self.pieces, self.thoughts = collections.deque(), collections.deque()  # streamed answer and thinking not yet on screen
        self.answer, self.answer_text = None, ""  # the widget of the answer being streamed, and its text
        self.thought, self.thought_text = None, ""  # the same for the thinking before it
        self.shown, self.shown_text = None, ""  # the same for the last answer that finished
        self.busy = False  # a message is being answered
        self.asking, self.reply = None, False  # the Event of an edit waiting for Enter / Esc, and the answer given
        self.activity, self.call, self.used = "ready", None, None
        self.started = self.call_started = 0.0
        self.calls, self.call_timings = 0, []
        self.hint_text, self.hint_until = "", 0.0  # a short message in the status bar, and when it expires
        self.phrase = self.last_phrase = ""  # the phrase shown while working ("" when idle), and the one shown last
        self.generated = 0  # pieces (each about a token) of thinking and answer streamed for this message
        self.call_from = 0  # self.generated when the current call started
        self.call_first = self.call_last = 0.0  # when the current call's first and latest pieces arrived
        self.quit_until = 0.0  # a second Ctrl+C with nothing selected before this time quits
        self.history = []  # the messages sent so far, each as (typed text, its pasted blocks)
        self.history_at = None  # which of them Up/Down has put in the input; None when it holds new text
        self.draft = ("", {})  # the unsent text that was in the input when Up was first pressed
        self.files = None  # the project's files while @ suggestions are open
        self.matches, self.choice = [], 0  # the lines of the open list (files or models) and the highlighted one

    def header_text(self):
        folder, home = os.getcwd(), os.path.expanduser("~")
        if folder.startswith(home):
            folder = "~" + folder[len(home):]
        if len(folder) > 40:  # keep the end of a long path, so the model and context size stay visible
            folder = "…" + folder[-39:]
        return f"tolken · {folder} · {self.session.model.label} · {self.session.size:,} tokens"

    def compose(self):
        self.header = Static(self.header_text(), id="header", markup=False)
        self.conversation = VerticalScroll(id="conversation", can_focus=False)
        self.prompt = Prompt(PLACEHOLDER)
        self.status_bar = Static(id="status", markup=False)
        self.suggestions = Static(id="suggestions", markup=False)
        self.flavor = Static(id="flavor", markup=False)
        yield self.header
        yield self.conversation
        yield self.suggestions
        yield self.flavor
        yield self.prompt
        yield self.status_bar

    def on_mount(self):
        self.prompt.focus()
        # Stay at the end as answers grow (they are drawn after they arrive), until the user scrolls up.
        self.conversation.anchor()
        self.show_approval()
        self.set_interval(0.1, self.tick)
        self.tick()
        if tolken.MODEL:  # the model the settings start with, switched to as /model NAME would
            self.run_model(lambda: self.find_model(tolken.MODEL), f"loading {tolken.MODEL}")

    def on_unmount(self):
        self.session.cancel()
        if self.asking:
            self.asking.set()

    # ----- the conversation area -----

    def at_end(self):
        """True while the conversation is scrolled to its end, so new content should stay in view."""
        return self.conversation.scroll_y >= self.conversation.max_scroll_y - 1

    def end_answer(self):
        """Close the answer being streamed; text that streams after this starts a new item."""
        self.flush()
        if self.answer is not None:
            self.shown, self.shown_text = self.answer, self.answer_text
        self.answer, self.answer_text = None, ""
        self.thought, self.thought_text = None, ""

    def add(self, content, kind="info"):
        """Append one item (a message, note or edit card) to the conversation and return its widget."""
        self.end_answer()
        if self.conversation.is_attached:
            follow = self.at_end()
            item = Static(content, classes=kind, markup=False)
            self.conversation.mount(item)
            if follow:
                self.conversation.scroll_end(animate=False)
            return item

    def flush(self):
        """Put streamed thinking and answer text that has arrived on screen."""
        if not (self.pieces or self.thoughts) or not self.conversation.is_attached:  # nothing new, or closing
            return
        follow = self.at_end()
        self.call_last = time.monotonic()  # for the speed shown above the input
        self.call_first = self.call_first or self.call_last
        if self.thoughts:
            while self.thoughts:
                self.thought_text += self.thoughts.popleft()
                self.generated += 1
            if self.thought_text.strip():
                self.thought = self.show_stream(self.thought, self.thought_text, "thinking")
                if self.activity == "waiting for model":
                    self.activity = "thinking"
        if self.pieces:
            while self.pieces:
                self.answer_text += self.pieces.popleft()
                self.generated += 1
            if self.answer_text.strip():
                self.answer = self.show_stream(self.answer, self.answer_text, "answer")
                if self.activity in ("waiting for model", "thinking"):
                    self.activity = "answering"
        if follow:
            self.conversation.scroll_end(animate=False)

    def show_stream(self, widget, text, kind):
        """Show streamed text in widget, or in a new one of class kind when widget is None; return the widget.

        The answer is shown as markdown, the thinking as plain text.
        """
        if widget is None:
            widget = Answer() if kind == "answer" else Static(classes=kind, markup=False)
            self.conversation.mount(widget)
        if kind == "answer":
            widget.show(displayed(text.lstrip()))
        else:
            widget.update(text.strip())
        return widget

    # ----- the status bar -----

    def set_activity(self, text, call=None, used=None):
        self.activity = text
        if call:
            self.flush()  # pieces of the call before still waiting count for that one, not this one
            self.call, self.used, self.calls = call, used, call
            self.call_started = time.monotonic()
            self.call_from, self.call_first = self.generated, 0.0
        self.tick()

    def tick(self):
        """Runs ten times a second: show new answer text, the phrase above the input, and refresh the status bar."""
        self.flush()
        if not self.busy:
            self.phrase, self.generated = "", 0
        elif not self.phrase:  # a message (or a model load) has started: pick a phrase for it
            self.phrase = self.last_phrase = random.choice([p for p in PHRASES if p != self.last_phrase])
        line = "✦ " + self.phrase
        if self.generated:
            line += f" (↓ {self.generated:,} token{'s' * (self.generated != 1)}"
            if self.call_last - self.call_first >= 1:  # streamed long enough for a steady speed
                line += f" · {(self.generated - self.call_from) / (self.call_last - self.call_first):.0f} tok/s"
            line += ")"
        self.flavor.update(line)
        self.flavor.display = self.busy and not self.asking
        size = self.session.size
        parts = [self.activity]
        if self.busy and not self.asking:
            if self.call:
                parts.append(f"call {self.call}")
            parts.append(f"{time.monotonic() - self.call_started:.1f}s")
        context = f"{self.used or 0:,}/{size:,} ({(self.used or 0) * 100 // size}%)"
        keys = self.prompt.placeholder if self.asking else "please wait" if self.loading else "esc cancel" if self.busy \
            else "↑↓ choose · enter select · esc close" if self.models is not None \
            else "↑↓ choose · enter run · tab complete · esc close" if self.commands is not None \
            else "enter send · shift+enter new line · ctrl+c twice to exit"
        if time.monotonic() < self.hint_until:
            keys = self.hint_text
        self.status_bar.update(f"{' · '.join(parts)} │ {context} │ {keys}")

    def hint(self, text):
        """Show a short message in the status bar for two seconds."""
        self.hint_text, self.hint_until = text, time.monotonic() + 2
        self.tick()

    def summary(self):
        """One line about the message that just ended: time, calls, tokens, speed.

        The prompt time is how long the server spent reading the prompts of all the calls; a prompt
        it had cached is read in a moment. Ollama does not say how much was cached (cache_n None).
        """
        parts = [f"{time.monotonic() - self.started:.1f}s", f"{self.calls} call{'s' * (self.calls != 1)}"]
        if self.call_timings:
            last = self.call_timings[-1]
            cached = last.get("cache_n")
            prompt = (cached or 0) + last.get("prompt_n", 0)
            answer = sum(t.get("predicted_n", 0) for t in self.call_timings)
            spent = sum(t.get("predicted_ms", 0) for t in self.call_timings)
            reading = sum(t.get("prompt_ms") or 0 for t in self.call_timings)
            parts.append(f"prompt {prompt:,}" + (f" ({cached:,} cached)" if cached is not None else ""))
            if reading:
                parts.append(f"prompt time {reading / 1000:.1f}s")
            parts.append(f"answer {answer:,}")
            if spent:
                parts.append(f"{answer * 1000 / spent:.0f} tok/s")
            self.used = prompt + last.get("predicted_n", 0)  # what the window holds now
        return " · ".join(parts)

    # ----- the list above the input: @ file suggestions, / commands, or the models of /model -----

    def on_text_area_changed(self, event):
        """Suggest commands for a / at the start, or project files for the @word that is being typed.

        Typing also closes the model list.
        """
        if self.models is not None:
            self.models, self.activity = None, "ready"
        typed = self.prompt.value
        offered = [name for name in COMMANDS if name.startswith(typed)] if re.fullmatch(r"/\S*", typed) else []
        self.commands = offered or None
        word = re.search(r"(?:^|\s)@(\S*)$", self.prompt.line_before_cursor())
        if self.commands and not self.asking:
            self.files = None
            self.matches, self.choice = [f"{name:<8}{COMMANDS[name]}" for name in self.commands], 0
        elif word and not self.asking:
            if self.files is None:  # the list is opening: read the project once
                self.files = tolken.readable_files()
            self.matches, self.choice = matching(self.files, word[1]), 0
        else:
            self.files, self.matches = None, []
        self.show_matches()

    def show_matches(self):
        lines = [Text(("› " if index == self.choice else "  ") + path, "bold" if index == self.choice else "dim")
                 for index, path in enumerate(self.matches)]
        self.suggestions.update(Text("\n").join(lines))
        self.suggestions.display = bool(self.matches)

    def close_list(self):
        self.files, self.models, self.commands, self.matches = None, None, None, []
        if self.activity == "choose a model":
            self.activity = "ready"
        self.show_matches()
        self.tick()  # so the status bar changes at once, not at the next refresh

    def choose(self, step):
        """Up (-1) or Down (+1): move in the open list, otherwise in the history."""
        if self.matches:
            self.choice = (self.choice + step) % len(self.matches)
            self.show_matches()
        elif self.history and (self.history_at is not None or step < 0):
            if self.history_at is None:  # leaving new text: keep it to come back to
                self.draft = (self.prompt.value, self.prompt.pastes)
                self.history_at = len(self.history)
            self.history_at = max(0, min(len(self.history), self.history_at + step))
            if self.history_at == len(self.history):
                (value, pastes), self.history_at = self.draft, None
            else:
                value, pastes = self.history[self.history_at]
            with self.prompt.prevent(TextArea.Changed):  # a recalled @file must not open the suggestions
                self.prompt.value = value
            self.prompt.pastes = dict(pastes)

    def complete(self, enter=False):
        """Enter or Tab on the open list: switch to the highlighted model, put the highlighted command
        in the input, or put the highlighted file in place of the @word before the cursor.

        Returns False when Enter should go on to send the input: no list is open, or a command was
        chosen with Enter, which runs it.
        """
        if not self.matches:
            return False
        if self.commands is not None:
            name = self.commands[self.choice]
            self.close_list()
            with self.prompt.prevent(TextArea.Changed):  # must not open the list again
                self.prompt.value = name if enter else name + " "  # Tab leaves room for what follows the command
            return not enter
        if self.models is not None:
            model = self.models[self.choice]
            self.close_list()
            self.run_model(lambda: self.session.use(model), f"loading {model.name}")
            return True
        row, column = self.prompt.cursor_location
        start = self.prompt.line_before_cursor().rfind("@") + 1
        result = self.prompt.replace(self.matches[self.choice] + " ", (row, start), (row, column))
        self.prompt.move_cursor(result.end_location)
        return True

    # ----- running a message -----

    def on_prompt_submitted(self, event):
        if self.busy:
            self.hint("still working" if self.loading else "still working · esc cancels")
        else:
            pastes = dict(self.prompt.pastes)
            typed, message = self.prompt.take()
            if typed and (typed, pastes) != (self.history[-1:] or [None])[0]:
                self.history.append((typed, pastes))
            self.history_at = None
            if message == "/exit":
                self.exit()
            elif message == "/clear":
                self.session.clear()
                self.conversation.remove_children()
                self.answer, self.answer_text, self.shown, self.used = None, "", None, None
            elif message == "/undo":
                for note in self.session.undo():
                    self.add(note)
            elif message.split()[:1] == ["/model"]:
                wanted = message[6:].strip()
                self.run_model(lambda: self.find_model(wanted), "listing models")
            elif message:
                self.add("> " + typed, "user")
                self.conversation.scroll_end(animate=False)
                self.busy, self.activity, self.call, self.calls = True, "starting", None, 0
                self.started = self.call_started = time.monotonic()
                self.call_timings = []
                self.run_worker(lambda: self.work(message), thread=True)

    def work(self, message):
        """Runs in a worker thread."""
        try:
            self.session.send(message)
        finally:
            self.session.ui.on_screen(self.finished)

    def finished(self):
        self.add(self.summary())
        self.busy, self.activity, self.call = False, "ready", None
        self.tick()

    # ----- /model -----

    def run_model(self, job, activity):
        """Run job (listing the models, or loading one) in a worker thread, showing activity meanwhile."""
        self.busy, self.loading, self.activity, self.call = True, True, activity, None
        self.call_started = time.monotonic()
        self.run_worker(lambda: self.model_work(job), thread=True)

    def model_work(self, job):
        """Runs in a worker thread."""
        try:
            job()
        finally:
            self.session.ui.on_screen(self.model_done)

    def find_model(self, wanted):
        """Runs in a worker thread: open the model list, or switch straight to the model that wanted names."""
        ui, found = self.session.ui, tolken.models()
        if not found:
            ui.note("no model server answers", "bad")
        elif not wanted:
            ui.on_screen(self.show_models, found)
        elif model := tolken.pick(found, wanted):
            self.session.use(model)
        else:
            ui.note(f"no single model matches {wanted!r}", "bad")

    def show_models(self, found):
        current, labels = self.session.model.label, [model.label for model in found]
        self.models = found
        self.matches = [f"{model.label} · {model.detail}" + "   (current)" * (model.label == current) for model in found]
        self.choice = labels.index(current) if current in labels else 0
        self.show_matches()

    def model_done(self):
        if self.header_text() != str(self.header.content):  # another model is in use now
            self.header.update(self.header_text())
            self.used = None
        self.busy = self.loading = False
        self.activity = "choose a model" if self.models is not None else "ready"
        self.tick()

    # ----- edits -----

    def show_change(self, path, old, new):
        """Show how a file changes (new is None: is deleted) as a card in place of the answer's edit blocks."""
        self.end_answer()
        if self.shown is not None:  # the card replaces the raw SEARCH/REPLACE text of the answer
            self.shown_text = without_blocks(self.shown_text, path)
            if self.shown_text.strip():
                self.shown.show(displayed(self.shown_text.strip()))
            else:
                self.shown.remove()
                self.shown = None
        title, body = diff_card(path, old, new)
        card = self.add(body, "card")
        if card:
            card.border_title = escape(title)

    def ask(self, path, old, new, asked):
        """Show how a file would change (new is None: be deleted) and wait for Enter or Esc.

        Display.confirm waits on the event.
        """
        self.show_change(path, old, new)
        if new is None:
            self.question(asked, "delete file?", f"Delete {path}?", DELETE_KEYS)
        else:
            self.question(asked, "apply edit?", f"Apply this change to {path}?", EDIT_KEYS)

    def ask_run(self, command, asked):
        """Show a command the model wants to run and wait for Enter or Esc. Display.allow waits on the event."""
        self.end_answer()
        card = self.add(Text(command), "card")
        if card:
            card.border_title = "run"
        self.question(asked, "run command?", "Run this command?", RUN_KEYS)

    def question(self, asked, activity, title, keys):
        """Make the input box the question until Enter or Esc answers it (see answer_diff)."""
        self.asking, self.activity = asked, activity
        self.prompt.border_title = escape(title)
        self.prompt.placeholder = keys
        self.prompt.disabled = True
        self.tick()

    def show_run(self, command, code, seconds, output):
        """Show the end of a command's output as a card, titled with how it ended."""
        self.end_answer()
        lines = output.rstrip().splitlines()
        body = Text("\n".join(lines[-OUTPUT_LINES:]) or "(no output)", GREY)
        if len(lines) > OUTPUT_LINES:
            body = Text(f"⋮ {len(lines) - OUTPUT_LINES} lines above\n", GREY) + body
        ended = f"stopped after {tolken.RUN_TIMEOUT} s" if code is None else f"exit {code} · {seconds:.1f}s"
        card = self.add(body, "card")
        if card:
            card.border_title = escape(f"{command} · {ended}")

    def answer_diff(self, yes):
        asked, self.asking = self.asking, None
        self.reply, self.activity = yes, "working"
        self.prompt.border_title = ""
        self.prompt.placeholder = PLACEHOLDER
        self.prompt.disabled = False
        self.prompt.focus()
        asked.set()

    def check_action(self, action, parameters):
        """Enter applies a waiting edit; at any other time it sends the message as usual."""
        return bool(self.asking) if action == "accept" else True

    def action_accept(self):
        self.answer_diff(True)

    def show_approval(self):
        """Show on the input box's lower border whether changes wait for the user ("ask") or not ("auto")."""
        auto = self.session.approve == "auto"
        self.prompt.border_subtitle = f"{'auto' if auto else 'ask'} · shift+tab"
        self.prompt.set_class(auto, "auto")

    def action_paste_clipboard(self):
        """Alt+V: attach the files or the image the clipboard holds, or paste its text.

        The terminal's own paste (Ctrl+V) brings only text, so the clipboard is read here.
        An image is saved as a PNG in a temporary folder first.
        """
        if self.asking:
            return
        held = read_clipboard()
        if held is None:
            self.hint("the clipboard holds nothing to attach" if os.name == "nt" else "alt+v works on Windows only")
        elif held[0] == "files":
            for path in held[1]:
                self.prompt.attach(path)
        elif held[0] == "image":
            self.prompt.attach(save_image(held[1]))
        elif dropped(held[1]) or len(held[1].splitlines()) > 1:
            self.prompt.post_message(events.Paste(held[1]))  # as a paste: attached files, or a marker
        else:
            self.prompt.insert(held[1])

    def action_cycle_approval(self):
        """Shift+Tab: switch between asking before each change and making changes at once.

        A change that waits for Enter is applied: the first one made at once.
        """
        self.session.approve = "ask" if self.session.approve == "auto" else "auto"
        self.show_approval()
        if self.asking and self.session.approve == "auto":
            self.answer_diff(True)

    # ----- other keys -----

    def action_cancel(self):
        if self.matches:  # Esc first closes the open list
            self.close_list()
        elif self.asking:
            self.answer_diff(False)
        elif self.busy and not self.loading:
            self.activity = "cancelling"
            self.session.cancel()

    def action_copy_selection(self):
        """Ctrl+C: copy the selected text; with nothing selected, a second press within two seconds quits."""
        text = self.prompt.selected_text or self.screen.get_selected_text()
        if text:
            self.copy_to_clipboard(text)  # through the terminal, which not every terminal allows
            copy(text)  # and through the system's clipboard command
            self.hint("copied")
        elif time.monotonic() < self.quit_until:
            self.exit()
        else:
            self.quit_until = time.monotonic() + 2
            self.hint("press ctrl+c again to exit")

    def action_page(self, direction):
        self.conversation.scroll_page_down() if direction > 0 else self.conversation.scroll_page_up()


def run():
    """Start the interface. Finds the model to start with first, while printing still works."""
    tolken.MODEL = tolken.startup()  # on_mount switches to it
    TolkenApp(8192).run()
