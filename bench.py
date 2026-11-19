#!/usr/bin/env python3
"""Benchmark: run the same small coding tasks through the harness on several models.

  python bench.py                           llama-server's model, and Opus at xhigh effort as Claude Code
  python bench.py --models qwen3:8b cc:opus choose the rows (see below)
  python bench.py --hard                    the harder tasks, which tell models apart (default: the easy ones)
  python bench.py --runs 1 --tasks create rename
  python bench.py --save report.md

Rows: a name as in /model runs that model through the harness. cc:opus or
cc:opus:xhigh is Claude Code itself, working in the task's folder with its own
tools. opus or opus:xhigh is Claude inside the harness, on the same prompts as
the local models and with no tools. A full model name works in place of opus,
for example cc:claude-opus-4-8:xhigh.

Every task runs in a throwaway project and the result is checked. Through the
harness, every edit the model offers is accepted. Claude is reached with the
`claude` command and your Claude Code login.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import harness

WINDOW = 8192  # the context size prompts are built for when a Claude model is used
CLAUDE_NAMES = ("opus", "fable", "sonnet", "haiku")  # model names the claude command understands

# ---------- the projects the tasks work in ----------

JOYSTICK = {"joystick_viewer.py": '''"""Show the state of the first connected joystick."""
import pygame


def main():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("no joystick found")
        return
    stick = pygame.joystick.Joystick(0)
    stick.init()
    screen = pygame.display.set_mode((400, 300))
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill((0, 0, 0))
        for axis in range(stick.get_numaxes()):
            value = stick.get_axis(axis)
            pygame.draw.rect(screen, (0, 200, 0), (20, 20 + axis * 30, int(180 + value * 160), 20))
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
'''}

CART = {
    "src/cart.cpp": '''#include "cart.h"
#include "util/money.h"
#include <vector>

double Cart::total() const {
    double sum = 0;
    for (const Item& item : items_) {
        sum += item.price * item.quantity;
    }
    return roundMoney(sum * (1 + kTaxRate));
}
''',
    "src/cart.h": '''#pragma once
#include <string>
#include <vector>

struct Item {
    std::string name;
    double price;
    int quantity;
};

class Cart {
public:
    double total() const;
private:
    std::vector<Item> items_;
};
''',
    "include/util/money.h": '''#pragma once
#include <cmath>

const double kTaxRate = 0.18;

inline double roundMoney(double value) {
    return std::round(value * 100) / 100;
}
''',
}

STATS = {
    "stats.py": '''"""Small statistics helpers."""


def mean(values):
    return sum(values) / len(values)


def median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return ordered[middle] + ordered[middle + 1] / 2
''',
    "test_stats.py": '''from stats import mean, median

assert mean([1, 2, 3]) == 2
assert median([3, 1, 2]) == 2
assert median([4, 1, 3, 2]) == 2.5
assert median([10, 20]) == 15
print("ok")
''',
}


# ---------- the tasks: each check returns "" for a pass, or the reason it failed ----------

def read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def runs_clean(*arguments):
    """Run Python in the task's folder; "" if it exits cleanly within ten seconds, otherwise why not."""
    try:
        done = subprocess.run([sys.executable, *arguments], capture_output=True, timeout=10)
    except subprocess.TimeoutExpired:
        return "the code did not finish within 10 seconds"
    return "" if done.returncode == 0 else (done.stderr.decode(errors="replace").strip().splitlines() or ["failed"])[-1]


def check_greet(answer, run):
    if run.calls != 1:
        return f"made {run.calls} model calls for a greeting"
    return "offered an edit for a greeting" if run.edits else "" if answer.strip() else "gave no answer"


def check_read(answer, run):
    return "" if re.search(r"pygame|\bax[ie]s\b", answer, re.I) else "the answer shows no sign of the file's contents"


def check_fact(answer, run):
    return "" if re.search(r"0\.18|\b18\s?(%|percent)", answer) else "the answer does not give the rate 0.18"


def check_create(answer, run):
    if not os.path.isfile("fib.py"):
        return "fib.py was not created"
    return runs_clean("-c", "import fib; assert fib.fib(1) == 1 and fib.fib(10) == 55, 'fib gives wrong numbers'")


def check_rename(answer, run):
    def plain(text):
        return [line.rstrip() for line in text.strip().splitlines()]

    text = read("src/cart.cpp")
    if re.search(r"\bsum\b", text) or "subtotal" not in text:
        return "sum was not renamed to subtotal everywhere"
    return "" if plain(text) == plain(CART["src/cart.cpp"].replace("sum", "subtotal")) else "other lines changed too"


def check_method(answer, run):
    header, source = read("src/cart.h"), read("src/cart.cpp")
    if not re.search(r"\bisEmpty\s*\(", header):
        return "isEmpty is not declared in cart.h"
    defined = re.search(r"Cart::isEmpty\s*\(", source) or re.search(r"isEmpty\s*\(\s*\)\s*(const\s*)?\{", header)
    return "" if defined else "isEmpty is declared but not defined"


def check_value(answer, run):
    return "" if re.search(r"kTaxRate\s*=\s*0?\.20?\s*;", read("include/util/money.h")) else "kTaxRate is not 0.20"


def check_fix(answer, run):
    return runs_clean("test_stats.py")


def check_trace(answer, run):
    numbers = re.findall(r"-?\d+", answer)
    return "" if numbers and numbers[-1] == "40" else "the answer does not end in the right number, 40"


def tests_pass(files, test):
    """A check that puts the project's test file back as it was (a model may have edited it) and runs it."""
    def check(answer, run):
        with open(test, "w", encoding="utf-8") as f:
            f.write(files[test])
        return runs_clean(test)
    return check


# The harder tasks: each project has a test file that says exactly what is wanted.

ROMAN = {"test_roman.py": '''from roman import from_roman, to_roman

cases = {1: "I", 4: "IV", 9: "IX", 14: "XIV", 40: "XL", 90: "XC", 400: "CD", 944: "CMXLIV",
         1994: "MCMXCIV", 3999: "MMMCMXCIX"}
for number, text in cases.items():
    assert to_roman(number) == text, (number, to_roman(number))
    assert from_roman(text) == number, (text, from_roman(text))
print("ok")
'''}

INTERVALS = {"test_intervals.py": '''from intervals import merge

assert merge([]) == []
assert merge([(1, 3), (2, 6), (8, 10), (15, 18)]) == [(1, 6), (8, 10), (15, 18)]
assert merge([(5, 7), (1, 2), (2, 4)]) == [(1, 4), (5, 7)]  # unsorted input; touching intervals merge
assert merge([(1, 10), (2, 3), (4, 5)]) == [(1, 10)]
assert merge([(1, 2), (3, 4)]) == [(1, 2), (3, 4)]
print("ok")
'''}

CALC = {"test_calc.py": '''import re

from calc import calc

assert calc("2+3*4") == 14
assert calc("(2+3)*4") == 20
assert calc(" 10 / 4 ") == 2.5
assert calc("-3+5") == 2
assert calc("2*(3+(4-1))/3") == 4
assert calc("2--3") == 5
assert calc("-(2+3)*2") == -10
assert calc("1.5*4") == 6
assert calc("8/2/2") == 2
assert calc("2-3-4") == -5
for bad in ("2+", "(1+2", "1 2", "", "2*/3", "abc"):
    try:
        calc(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"{bad!r} should raise ValueError")
with open("calc.py") as source:
    assert not re.search(r"\\b(eval|exec)\\s*\\(", source.read()), "calc.py must not use eval or exec"
print("ok")
'''}

LRU = {
    "lru.py": '''"""A least-recently-used cache."""
from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = OrderedDict()

    def get(self, key, default=None):
        if key not in self.items:
            return default
        return self.items[key]

    def put(self, key, value):
        if len(self.items) >= self.capacity:
            self.items.popitem(last=False)
        self.items[key] = value

    def __len__(self):
        return len(self.items)
''',
    "test_lru.py": '''from lru import LRUCache

cache = LRUCache(2)
cache.put("a", 1)
cache.put("b", 2)
assert cache.get("a") == 1
cache.put("c", 3)  # b goes, because a was used more recently
assert cache.get("b") is None and cache.get("a") == 1 and cache.get("c") == 3
cache.put("a", 10)  # changing a value removes nothing
assert len(cache) == 2 and cache.get("c") == 3 and cache.get("a") == 10
cache.put("d", 4)  # c was used before a, so c goes
assert cache.get("c") is None and cache.get("a") == 10 and cache.get("d") == 4
print("ok")
''',
}

MYSTERY = {"mystery.py": '''def mystery(n):
    total = 0
    while n > 0:
        if n % 3 == 0:
            total += n
            n //= 3
        else:
            total -= 1
            n -= 1
    return total
'''}

WRAP = {"test_wrap.py": '''from wrap import wrap

assert wrap("", 10) == []
assert wrap("hello", 10) == ["hello"]
assert wrap("the quick brown fox", 10) == ["the quick", "brown fox"]
assert wrap("the quick brown fox", 9) == ["the quick", "brown fox"]
assert wrap("the quick brown fox", 5) == ["the", "quick", "brown", "fox"]
assert wrap("a  b   c", 3) == ["a b", "c"]
assert wrap("abcdefghij kl", 4) == ["abcd", "efgh", "ij", "kl"]
assert wrap("ab cdefghi", 4) == ["ab", "cdef", "ghi"]
print("ok")
'''}

ROOMS = {"test_rooms.py": '''from rooms import rooms

assert rooms([]) == 0
assert rooms([(9, 10)]) == 1
assert rooms([(9, 10), (10, 11)]) == 1  # one ends exactly when the next starts
assert rooms([(9, 11), (10, 12), (11, 13)]) == 2
assert rooms([(1, 5), (2, 6), (3, 7), (4, 8)]) == 4
assert rooms([(1, 10), (2, 3), (4, 5), (6, 7)]) == 2
assert rooms([(5, 8), (1, 4), (3, 6), (7, 9)]) == 2
print("ok")
'''}

SHOP = {
    "models.py": '''"""What the shop sells."""
from dataclasses import dataclass


@dataclass
class Item:
    name: str
    price: float
    quantity: int = 1
''',
    "store.py": '''"""The shop's stock."""
from models import Item


class Store:
    def __init__(self):
        self.items = []

    def add(self, name, price, quantity=1):
        self.items.append(Item(name, price, quantity))

    def total_value(self):
        return sum(item.price * item.quantity for item in self.items)
''',
    "report.py": '''"""Text reports about the stock."""


def line(item):
    return f"{item.name} x{item.quantity} @ {item.price:.2f}"


def report(store):
    lines = [line(item) for item in store.items]
    lines.append(f"total {store.total_value():.2f}")
    return "\\n".join(lines)
''',
    "test_shop.py": '''from report import report
from store import Store

store = Store()
store.add("apple", 2.0, 3)
store.add("pear", 4.0, 1, discount=25)
store.add("fig", 10.0, 2, discount=10)
assert store.items[0].discount == 0
assert abs(store.total_value() - 27.0) < 1e-9, store.total_value()
assert report(store) == "apple x3 @ 2.00\\npear x1 @ 4.00 (-25%)\\nfig x2 @ 10.00 (-10%)\\ntotal 27.00", report(store)
print("ok")
''',
}

QUEENS = {"test_queens.py": '''from queens import solutions

assert [solutions(n) for n in (1, 2, 3, 4, 6, 8)] == [1, 0, 0, 2, 4, 92]
print("ok")
'''}

DURATION = {"test_duration.py": '''from duration import parse_duration

assert parse_duration("45s") == 45
assert parse_duration("90m") == 5400
assert parse_duration("2h") == 7200
assert parse_duration("1h30m15s") == 5415
assert parse_duration("1h5s") == 3605
assert parse_duration("0s") == 0
for bad in ("", "h", "10", "5m5m", "1s2h", "1.5h", "-5m", "1h 30m"):
    try:
        parse_duration(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"{bad!r} should raise ValueError")
print("ok")
'''}


TEXT = {  # two bugs, found most easily by running the tests (RUN: lines)
    "text.py": '''"""Small text helpers."""


def slug(title):
    """'Hello, World!' gives 'hello-world'."""
    words = "".join(c if c.isalnum() else " " for c in title).split()
    return "-".join(words)


def initials(name):
    """'ada lovelace' gives 'A.L.'."""
    return "".join(part[0].upper() + "." for part in name.split(" "))
''',
    "test_text.py": '''from text import initials, slug


def test_slug():
    assert slug("Hello, World!") == "hello-world"
    assert slug("  Many   spaces ") == "many-spaces"


def test_initials():
    assert initials("ada lovelace") == "A.L."
    assert initials("  grace  brewster hopper ") == "G.B.H."


if __name__ == "__main__":
    test_slug()
    test_initials()
    print("ok")
'''}


# The longer tasks: several messages in one conversation, for what a larger window allows.

def check_recall(answer, run):
    if not os.path.isfile("mode.py"):
        return "mode.py was not created"
    return runs_clean("-c", "import mode; assert mode.release_mode() == 'strict', 'release_mode() is not strict'")


def warehouse(part):
    """Plain helper code that a larger project carries but the task does not need."""
    lines = [f'"""Helpers for stage {part} of the warehouse jobs."""', ""]
    for step in range(12):
        lines += [f"def step_{part}_{step}(values, factor={step + 1}):",
                  f'    """Scale the values for step {step} and keep those above its limit."""',
                  "    kept = []", "    for value in values:", f"        scaled = value * factor + {part}",
                  f"        if scaled > {step * 10}:", "            kept.append(scaled)", "    return kept", "", ""]
    return "\n".join(lines)


LARGE = {
    **{f"warehouse/stage_{part}.py": warehouse(part) for part in range(24)},
    "pricing.py": '''"""Price helpers."""


def apply_discount(amount, percent):
    """The amount lowered by percent (0 to 100)."""
    return amount * (100 - percent) / 100
''',
    "orders.py": '''"""Orders: lines of products with their prices and quantities."""
from pricing import apply_discount


class Order:
    def __init__(self, customer, discount=0):
        self.customer = customer
        self.discount = discount  # percent
        self.lines = []

    def add(self, product, price, quantity=1):
        self.lines.append((product, price, quantity))

    def subtotal(self):
        return sum(price * quantity for _, price, quantity in self.lines)

    def total(self):
        total = apply_discount(self.subtotal(), self.discount)
        return round(apply_discount(total, self.discount), 2)
''',
    "test_orders.py": '''from orders import Order

order = Order("ada", discount=10)
order.add("pen", 2.0, 5)
order.add("book", 15.0)
assert order.subtotal() == 25.0
assert order.total() == 22.5, order.total()
assert Order("bob").total() == 0
print("ok")
''',
}


def check_large(answer, run):
    failed = tests_pass(LARGE, "test_orders.py")(answer, run)
    return failed or runs_clean("-c", "from orders import Order; order = Order('x'); order.add('pen', 2.0, 5); "
                                      "order.add('book', 15.0); assert order.item_count() == 6, 'item_count() is wrong'")


TASKS = {  # name: (project files, the messages to send, check)
    "greet": (JOYSTICK, ["Hi"], check_greet),
    "read": (JOYSTICK, ["What does this project do?"], check_read),
    "fact": (CART, ["What tax rate does the cart total use, as a number?"], check_fact),
    "create": (JOYSTICK, ["Create a file fib.py with a function fib(n) that returns the n-th Fibonacci number, "
                          "so that fib(1) is 1 and fib(10) is 55."], check_create),
    "rename": (CART, ["In cart.cpp rename the variable sum to subtotal."], check_rename),
    "method": (CART, ["Add an isEmpty() method to the Cart class."], check_method),
    "value": (CART, ["Change the tax rate to 20%."], check_value),
    "fix": (STATS, ["Running test_stats.py fails in median. Fix the bug in stats.py."], check_fix),
    "roman": (ROMAN, ["Create roman.py with to_roman(n) and from_roman(text) for the numbers 1 to 3999. "
                      "test_roman.py must pass."], tests_pass(ROMAN, "test_roman.py")),
    "merge": (INTERVALS, ["Create intervals.py with merge(intervals), which merges overlapping or touching "
                          "(start, end) intervals and returns them sorted. test_intervals.py must pass."],
              tests_pass(INTERVALS, "test_intervals.py")),
    "calc": (CALC, ["Create calc.py with calc(text), which works out arithmetic with + - * /, brackets, a leading "
                    "minus and decimal numbers, and raises ValueError for anything malformed. Do not use eval or "
                    "exec. test_calc.py must pass."], tests_pass(CALC, "test_calc.py")),
    "lru": (LRU, ["test_lru.py fails. Fix lru.py so that it passes."], tests_pass(LRU, "test_lru.py")),
    "trace": (MYSTERY, ["What does mystery(30) in mystery.py return? Work it out and end your answer with the number."],
              check_trace),
    "wrap": (WRAP, ["Create wrap.py with wrap(text, width), which returns the text as a list of lines of at most "
                    "width characters. Words are separated by any amount of whitespace; put as many words on a line "
                    "as fit, joined by single spaces. A word longer than width starts on a new line and is cut into "
                    "pieces of width characters. test_wrap.py must pass."], tests_pass(WRAP, "test_wrap.py")),
    "rooms": (ROOMS, ["Create rooms.py with rooms(meetings), which returns how many rooms are needed so that no two "
                      "(start, end) meetings share a room at the same time. A meeting may start exactly when another "
                      "ends. test_rooms.py must pass."], tests_pass(ROOMS, "test_rooms.py")),
    "shop": (SHOP, ["Give items an optional discount in percent (default 0). It lowers the item's value in "
                    "total_value, and the report shows it after the price, like ' (-25%)', when it is not 0. "
                    "test_shop.py must pass."], tests_pass(SHOP, "test_shop.py")),
    "queens": (QUEENS, ["Create queens.py with solutions(n), which returns in how many ways n queens can stand on an "
                        "n by n board without attacking each other. test_queens.py must pass."],
               tests_pass(QUEENS, "test_queens.py")),
    "time": (DURATION, ["Create duration.py with parse_duration(text), which turns text like '1h30m15s' into "
                        "seconds. The units are h, m and s; each may appear at most once, in that order, with a "
                        "whole number before it. Anything else raises ValueError. test_duration.py must pass."],
             tests_pass(DURATION, "test_duration.py")),
    "tests": (TEXT, ["The tests in test_text.py fail. Run them, find out why, and fix text.py."],
              tests_pass(TEXT, "test_text.py")),
    "recall": (STATS, ["The release mode for this project is 'strict'. Just note it; there are no files to change.",
                       "What does mean() in stats.py return for [1, 2, 3]?",
                       "Create mode.py with a function release_mode() that returns the release mode I told you "
                       "about, as a string."], check_recall),
    "large": (LARGE, ["Where is the order discount applied?",
                      "test_orders.py fails. Fix it.",
                      "Add a method item_count() to Order that returns the total quantity of all its lines."],
              check_large),
}
EASY = ["greet", "read", "fact", "create", "rename", "method", "value", "fix"]
HARD = ["fix", "roman", "merge", "calc", "lru", "trace", "wrap", "rooms", "shop", "queens", "time", "tests"]
LONG = ["recall", "large"]  # run with --tasks


# ---------- Claude, through the claude command ----------

def fold(messages):
    """The conversation after the system message as one prompt, since the claude command takes a single one."""
    said = {"assistant": "[You answered:]\n", "user": "[The user now says:]\n"}
    return "\n\n".join([messages[0]["content"]] + [said[m["role"]] + m["content"] for m in messages[1:]])


def claude_command(model, effort, options, prompt, folder=None):
    """Run the claude command once, without the user's settings, CLAUDE.md or connectors. Returns its JSON result."""
    command = ["claude", "-p", "--model", model, "--setting-sources", "", "--strict-mcp-config",
               "--no-session-persistence", "--output-format", "json"] + options + (["--effort", effort] if effort else [])
    try:
        done = subprocess.run(command, input=prompt.encode("utf-8"), capture_output=True, cwd=folder, timeout=1200)
    except subprocess.TimeoutExpired:
        raise OSError("the claude command did not finish within 20 minutes") from None
    try:
        result = json.loads(done.stdout.decode("utf-8", "replace"))
    except ValueError:
        result = {}
    if done.returncode or result.get("is_error") or "result" not in result:
        problem = result.get("result") or done.stderr.decode("utf-8", "replace") or done.stdout.decode("utf-8", "replace")
        raise OSError(f"the claude command failed: {problem.strip()[:300]}")
    return result


def ask_claude(model, effort, system, prompt):
    """One answer from a Claude model with no tools: the given system prompt and prompt, nothing else."""
    with tempfile.TemporaryDirectory() as folder:
        system_file = os.path.join(folder, "system.txt")
        with open(system_file, "w", encoding="utf-8") as f:
            f.write(system)
        return claude_command(model, effort, ["--system-prompt-file", system_file, "--tools", ""], prompt, folder)


class ClaudeCode:
    """A Claude model working as Claude Code itself: its own instructions and tools, in the task's folder.

    It may read, create and edit files there and run Python (to try the tests); nothing else is allowed.
    """

    def __init__(self, model, effort=None):
        self.model, self.effort = model, effort
        self.name = "cc:" + (f"{model}:{effort}" if effort else model)
        self.label = "claude code · " + (f"{model} · {effort}" if effort else model)
        self.size, self.cost = WINDOW, 0.0

    def enter(self):
        pass

    def leave(self):
        pass

    def work(self, message, run):
        """Let Claude Code do one message in the current folder; record its turns and tokens on the run."""
        options = ["--permission-mode", "acceptEdits", "--allowedTools", "Bash(python *)", "PowerShell(python *)"]
        result = claude_command(self.model, self.effort, options, message)
        self.cost += result.get("total_cost_usd") or 0
        run.calls += result.get("num_turns") or 0
        run.tokens += (result.get("usage") or {}).get("output_tokens", 0)
        run.milliseconds += result.get("duration_api_ms") or 0  # time spent waiting for the model, all turns
        return result["result"]


class Claude:
    """A Claude model with the same methods as harness.Llama and harness.Ollama, so a Session can use it."""

    isolated = None  # the outcome of the isolation check, made once for all Claude rows
    tokenizer = None  # the URL of a llama-server, set by resolve(); its tokenizer counts the prompts

    def __init__(self, model, effort=None):
        self.model, self.effort = model, effort
        self.name = f"{model}:{effort}" if effort else model
        self.label = "claude · " + self.name.replace(":", " · ")
        self.detail, self.url, self.size = "Claude Code login", "the claude command", WINDOW
        self.cost = 0.0  # what the claude command reported for this row's calls, in US dollars

    def enter(self):
        """Check once that the claude command gives the model no tools and none of the user's own instructions."""
        if Claude.isolated is None:
            answer = ask_claude(self.model, None, "You are a test assistant. Answer truthfully and exactly as asked.",
                                "Reply with exactly two lines and nothing else.\nTOOLS=<how many tools you can call, "
                                "as a number>\nMEMORY=<yes or no: were you given any CLAUDE.md, user memory or "
                                "project instructions>")["result"]
            tools, memory = re.search(r"TOOLS\s*=\s*(\d+)", answer), re.search(r"MEMORY\s*=\s*(yes|no)", answer, re.I)
            clean = tools and memory and tools[1] == "0" and memory[1].lower() == "no"
            Claude.isolated = "" if clean else " ".join(answer.split())[:200]
        if Claude.isolated:
            raise OSError(f"the claude command is not isolated; the model reported: {Claude.isolated}")

    def leave(self):
        pass

    def count(self, messages):
        """llama-server's tokenizer when it is up, so prompts are trimmed exactly as for the local model."""
        if Claude.tokenizer:
            try:
                return harness.count(messages, Claude.tokenizer)
            except OSError:
                pass
        return sum(len(message["content"]) for message in messages) // 3

    def chat(self, messages, ui=harness.PLAIN, session=None):
        result = ask_claude(self.model, self.effort, messages[0]["content"], fold(messages[1:]))
        usage = result.get("usage") or {}
        self.cost += result.get("total_cost_usd") or 0
        # No predicted_ms: the call's duration includes start-up and the network, so a speed would mislead.
        ui.timings({"cache_n": 0, "prompt_n": usage.get("input_tokens", 0), "predicted_n": usage.get("output_tokens", 0)})
        ui.text(result["result"])
        ui.text("\n")
        return result["result"]


# ---------- running ----------

class Run:
    """What happened in one run of a task. Also the display the harness reports to: it accepts every edit."""

    def __init__(self):
        self.calls, self.edits, self.problems, self.tokens, self.milliseconds = 0, [], [], 0, 0.0
        self.prompt_ms = 0.0  # time the server spent reading the prompts
        self.seconds, self.reason, self.answer = 0.0, "", ""

    def note(self, text, kind="info"):
        if kind == "bad":
            self.problems.append(text)

    def activity(self, text, call=None, used=None):
        self.calls += bool(call)  # every model call, over all of the task's messages

    def thinking(self, piece):
        pass

    def text(self, piece):
        pass

    def timings(self, data):
        self.tokens += data.get("predicted_n", 0)
        self.milliseconds += data.get("predicted_ms", 0)
        self.prompt_ms += data.get("prompt_ms") or 0

    def show(self, path, old, new):
        self.edits.append(path)  # a change made at once, in the "auto" mode

    def confirm(self, path, old, new):
        self.edits.append(path)
        return True

    def allow(self, command):
        return False  # only the commands the settings allow run; nobody is there to allow others

    def ran(self, command, code, seconds, output):
        pass


def run_task(model, name):
    """One run of one task with one model, in a fresh temporary project."""
    files, messages, check = TASKS[name]
    run, home = Run(), os.getcwd()
    session = harness.Session(run, model.size)
    session.model = model
    session.approve = "auto"  # every edit is accepted anyway, and the allowed commands run
    with tempfile.TemporaryDirectory() as folder:
        folder = os.path.realpath(folder)  # the full name: Claude Code refuses to write through a short 8.3 path
        for path, text in files.items():
            os.makedirs(os.path.dirname(os.path.join(folder, path)), exist_ok=True)
            with open(os.path.join(folder, path), "w", encoding="utf-8", newline="") as f:
                f.write(text)
        os.chdir(folder)
        try:
            started = time.monotonic()
            for message in messages:
                if isinstance(model, ClaudeCode):  # Claude Code does the task itself, not through the harness
                    try:
                        run.answer = model.work(message, run)
                    except OSError as error:
                        run.problems.append(str(error))
                else:
                    session.send(message)
                    run.answer = session.history[-1][1] if session.history else ""
            run.seconds = time.monotonic() - started
            run.reason = run.problems[0] if run.problems else check(run.answer, run)
        finally:
            os.chdir(home)
    return run


def resolve(names):
    """The models that the given names refer to.

    cc:opus[:effort] is Claude Code itself; opus[:effort] is Claude inside the harness; the rest are as in /model.
    In place of opus, a full model name such as claude-opus-4-8 also works.
    """
    found, rows = harness.models(), []
    Claude.tokenizer = next((model.url for model in found if isinstance(model, harness.Llama)), None)
    for name in names:
        kind = ClaudeCode if name.startswith("cc:") else Claude
        base, _, effort = name.removeprefix("cc:").partition(":")
        if base in CLAUDE_NAMES or base.startswith("claude-"):
            rows.append(kind(base, effort or None))
        elif model := harness.pick(found, name):
            rows.append(model)
        else:
            print(f"[no single model matches {name!r}; skipped]")
    return rows


def table(results, tasks):
    """results maps a model label to {task name: [Run, ...]}. Returns the report as text."""
    width = max([len(label) for label in results] + [5])
    lines = [f"{'model':{width}}  passed  score  " + "  ".join(task.rjust(max(6, len(task))) for task in tasks)
             + "  s/task  calls  prompt s  tok/s"]
    failures = []
    for label, by_task in results.items():
        runs = [run for task in tasks for run in by_task[task]]
        passed = sum(not run.reason for run in runs)
        cells = "  ".join(f"{sum(not r.reason for r in by_task[task])}/{len(by_task[task])}".rjust(max(6, len(task)))
                          for task in tasks)
        spent = sum(run.milliseconds for run in runs)
        speed = f"{sum(run.tokens for run in runs) * 1000 / spent:5.0f}" if spent else "    -"
        reading = sum(run.prompt_ms for run in runs)  # per task, in seconds
        prompt = f"{reading / 1000 / len(runs):8.1f}" if reading else "       -"
        lines.append(f"{label:{width}}  {f'{passed}/{len(runs)}':>6}  {passed * 100 // len(runs):4}%  {cells}  "
                     f"{sum(r.seconds for r in runs) / len(runs):6.1f}  {sum(r.calls for r in runs) / len(runs):5.1f}  "
                     f"{prompt}  {speed}")
        for task in tasks:
            for number, run in enumerate(by_task[task], 1):
                if run.reason:
                    said = " ".join(run.answer.split())[:160]
                    failures.append(f"- {label} · {task} · run {number}: {run.reason}\n    answer: {said}")
    return "\n".join(lines + ["", "Failed runs:" if failures else "No failed runs."] + failures)


def main():
    parser = argparse.ArgumentParser(description="Run the harness benchmark tasks on several models.")
    parser.add_argument("--models", nargs="+", default=["llama.cpp", "cc:opus:xhigh"],
                        help="names as in /model; cc:opus[:effort] for Claude Code itself; opus[:effort] for Claude "
                             "inside the harness (default: llama.cpp cc:opus:xhigh)")
    parser.add_argument("--hard", action="store_true", help="run the harder tasks: " + " ".join(HARD))
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), metavar="TASK",
                        help="run just these tasks (default: the easy ones, " + " ".join(EASY)
                             + "); the longer ones, " + " ".join(LONG) + ", run only when named")
    parser.add_argument("--runs", type=int, default=3, help="how many times to run each task (default: 3)")
    parser.add_argument("--save", metavar="FILE", help="also write the report to this file")
    options = parser.parse_args()
    tasks = options.tasks or (HARD if options.hard else EASY)
    results, notes = {}, []
    for model in resolve(options.models):
        print(f"== {model.label}", flush=True)
        try:
            model.enter()  # loads an Ollama model, or checks that the claude command is isolated
        except (OSError, KeyError, ValueError) as error:
            print(f"[skipped: {harness.explain(error)}]")
            continue
        results[model.label] = {task: [] for task in tasks}
        for task in tasks:
            for number in range(1, options.runs + 1):
                run = run_task(model, task)
                results[model.label][task].append(run)
                print(f"   {task:8} run {number}/{options.runs}  {'pass' if not run.reason else 'FAIL'}  "
                      f"{run.seconds:5.1f}s  {run.calls} call(s)" + (f"  {run.reason}" if run.reason else ""), flush=True)
        model.leave()
        if getattr(model, "cost", 0):
            notes.append(f"{model.label}: the claude command reported ${model.cost:.2f} for its calls")
        # Report after every model, so a long run that is interrupted still leaves the finished rows.
        report = "\n".join([table(results, tasks), ""] + notes + [
            "tok/s: for local models, the answer speed their server reports. For Claude Code, output tokens "
            "(thinking included) per second spent waiting for the model. Not measured for Claude inside the harness."])
        print("\n" + report + "\n", flush=True)
        if options.save:
            with open(options.save, "w", encoding="utf-8") as f:
                f.write(report + "\n")
    if not results:
        print("[no model could be benchmarked]")


if __name__ == "__main__":
    main()
