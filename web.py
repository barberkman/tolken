#!/usr/bin/env python3
"""Web chat for the local models, with images, PDFs, Word documents and text files.

  python web.py      serves http://127.0.0.1:8765 (WEB_PORT changes it) and opens it

It talks to the model servers that harness.py knows (Ollama and llama-server),
with the same settings.json, and starts on the same model: the one the settings
name, else the first listed. Images reach a model only if it can read them (an
Ollama model with the "vision" capability). The page's ASCII button asks the
model to redraw each attached image as a text diagram, one call per image. The
conversation lives in the page; this server keeps only which model is in use.
"""
import json
import logging
import os
import queue
import threading
import webbrowser

import flask

import harness
from documents import read_docx, read_pdf

PORT = int(os.environ.get("WEB_PORT", "8765"))
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web.html")

app = flask.Flask(__name__)
current = None  # the model in use; main() starts on the settings' model, and the page can switch


def vision(model):
    """Whether the model can read images."""
    try:
        if isinstance(model, harness.Ollama):
            with harness.post("/api/show", {"model": model.name}, model.url) as response:
                return "vision" in (json.load(response).get("capabilities") or [])
        return bool(harness.props(model.url).get("modalities", {}).get("vision"))
    except (OSError, ValueError):
        return False


def listing():
    """The models that can be chosen, for the page's model list."""
    return [{"label": model.label, "detail": model.detail, "vision": vision(model),
             "current": current is not None and model.label == current.label} for model in harness.models()]


def use(label):
    """Switch to the model with this label, like Session.use in harness.py."""
    global current
    model = next((model for model in harness.models() if model.label == label), None)
    if model is None:
        raise ValueError(f"no model is called {label}")
    if current is not None and current.label != model.label:
        current.leave()  # first, so that the new model finds the GPU memory free
    model.enter()
    model.vision = vision(model)
    current = model
    return {"label": model.label, "size": model.size, "vision": model.vision}


def tokens(messages, model):
    """The prompt size of messages: the model's count of the text, plus an estimate for each image."""
    images = sum(len(message.get("images") or ()) for message in messages)
    return model.count(messages) + harness.IMAGE_TOKENS * images + harness.MARGIN


def fit(messages, model):
    """(messages without their oldest exchanges so that they fit the window, tokens).

    The messages are None when the newest one does not fit on its own.
    """
    while (used := tokens(messages, model)) > model.size - harness.answer_room(model.size):
        if len(messages) < 3:
            return None, used
        messages = messages[2:]  # one user message and its answer
    return messages, used


class Writer:
    """The ui that harness.stream passes the answer to: it puts each piece in a queue."""

    def __init__(self, pieces):
        self.pieces = pieces

    def text(self, piece):
        self.pieces.put(piece)

    def thinking(self, piece):
        pass  # only the answer is shown

    def timings(self, data):
        pass


class Call:
    """What harness.stream needs so that the call can be ended from another thread."""
    sock, stopped, process = None, False, None  # process: a command; the web chat runs none
    cancel = harness.Session.cancel  # sets stopped and closes sock, which is all it uses


def answer(model, messages):
    """The model's answer, piece by piece. An error after the answer has started comes as NUL and its text.

    The model is called in a thread, since harness.stream hands out the pieces
    instead of returning them. When the page goes away (Stop), the generator is
    closed and the call is ended.
    """
    pieces, call = queue.Queue(), Call()

    def run():
        try:
            model.chat(messages, Writer(pieces), call)
        except OSError as error:
            if not call.stopped:
                pieces.put("\0" + harness.explain(error))
        finally:
            pieces.put(None)

    threading.Thread(target=run, daemon=True).start()
    try:
        while (piece := pieces.get()) is not None:
            yield piece
    finally:
        call.cancel()


@app.get("/")
def page():
    return flask.send_file(PAGE)


@app.get("/models")
def list_models():
    return listing()


@app.post("/model")
def switch_model():
    return use(flask.request.get_json(force=True)["label"])


@app.post("/pdf")
def pdf_text():
    return read_pdf(flask.request.get_data())


@app.post("/docx")
def docx_text():
    return read_docx(flask.request.get_data())


@app.post("/chat")
def chat():
    messages, model = flask.request.get_json(force=True)["messages"], current
    if model is None:
        raise ValueError("no model is selected")
    if not model.vision and any(message.get("images") for message in messages):
        raise ValueError(f"{model.label} can't read images")
    fitted, used = fit(messages, model)
    if fitted is None:
        raise ValueError(f"the message is too long for {model.label}: "
                         f"about {used:,} tokens, and {model.size - harness.answer_room(model.size):,} fit")
    return flask.Response(answer(model, fitted), mimetype="text/plain",
                          headers={"X-Dropped": str(len(messages) - len(fitted))})  # earlier messages left out


@app.errorhandler(OSError)
@app.errorhandler(ValueError)
@app.errorhandler(KeyError)
@app.errorhandler(RuntimeError)  # PyMuPDF's errors
def refuse(error):
    return {"error": harness.explain(error)}, 400


def start():
    """Switch to the model to start with, as harness.py does: the one settings.json names, else the first listed."""
    name = harness.startup()
    model = harness.pick(harness.models(), name) if name else None
    if name and not model:
        print(f"[no single model matches {name!r}]")
    elif model:
        print(f"[loading {model.label}]")
        try:
            use(model.label)
        except (OSError, KeyError, ValueError) as error:
            print(f"[could not load {model.label}: {harness.explain(error)}]")


def main():
    harness.load_settings()  # the same servers and options as the harness
    start()
    logging.getLogger("werkzeug").setLevel(logging.ERROR)  # no line per request
    url = f"http://127.0.0.1:{PORT}"
    print(f"{url} | Ctrl+C stops")
    threading.Timer(1, webbrowser.open, [url]).start()  # once the server below is listening
    app.run("127.0.0.1", PORT, threaded=True)


if __name__ == "__main__":
    main()
