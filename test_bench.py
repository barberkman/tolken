"""Tests for bench.py. Run: python -m unittest test_bench

Models are replaced by scripted answers; no server and no claude call is made.
"""
import json
import os
import tempfile
import unittest
from unittest import mock

import bench
import tolken

tolken.apply_settings(bench.SETTINGS)  # the tests do not depend on the defaults tolken comes with
tolken.SESSIONS = None  # the tests' sessions are not kept

BLOCK = "FILE: {0}\n<<<<<<< SEARCH\n{1}\n=======\n{2}\n>>>>>>> REPLACE\n"
FIB = "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a"
RENAMED = bench.CART["src/cart.cpp"].replace("sum", "subtotal").strip()


SOLUTIONS = {  # reference answers for the harder tasks: file name -> a correct file
    "roman.py": '''PAIRS = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
         (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]


def to_roman(n):
    text = ""
    for value, letters in PAIRS:
        while n >= value:
            text += letters
            n -= value
    return text


def from_roman(text):
    total, at = 0, 0
    for value, letters in PAIRS:
        while text.startswith(letters, at):
            total += value
            at += len(letters)
    return total''',
    "intervals.py": '''def merge(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged''',
    "calc.py": '''import re


def calc(text):
    tokens = re.findall(r"\\d+\\.?\\d*|[-+*/()]", text)
    if not tokens or "".join(tokens) != text.replace(" ", ""):
        raise ValueError(text)
    position = 0

    def peek():
        return tokens[position] if position < len(tokens) else None

    def take():
        nonlocal position
        position += 1
        return tokens[position - 1]

    def factor():
        if peek() is None:
            raise ValueError(text)
        token = take()
        if token == "-":
            return -factor()
        if token == "(":
            value = expression()
            if peek() != ")":
                raise ValueError(text)
            take()
            return value
        if token in "+*/)":
            raise ValueError(text)
        return float(token)

    def term():
        value = factor()
        while peek() in ("*", "/"):
            value = value * factor() if take() == "*" else value / factor()
        return value

    def expression():
        value = term()
        while peek() in ("+", "-"):
            value = value + term() if take() == "+" else value - term()
        return value

    result = expression()
    if position != len(tokens):
        raise ValueError(text)
    return result''',
    "lru.py": bench.LRU["lru.py"].strip().replace(
        "        return self.items[key]", "        self.items.move_to_end(key)\n        return self.items[key]").replace(
        "        if len(self.items) >= self.capacity:",
        "        if key in self.items:\n            self.items.move_to_end(key)\n        elif len(self.items) >= self.capacity:"),
    "wrap.py": '''def wrap(text, width):
    lines, line = [], ""
    for word in text.split():
        while len(word) > width:
            if line:
                lines.append(line)
                line = ""
            lines.append(word[:width])
            word = word[width:]
        if not word:
            continue
        if not line:
            line = word
        elif len(line) + 1 + len(word) <= width:
            line += " " + word
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines''',
    "rooms.py": '''def rooms(meetings):
    events = sorted([(start, 1) for start, end in meetings] + [(end, -1) for start, end in meetings])
    most = now = 0
    for _, change in events:
        now += change
        most = max(most, now)
    return most''',
    "queens.py": '''def solutions(n):
    def place(row, columns, down, up):
        if row == n:
            return 1
        return sum(place(row + 1, columns | {c}, down | {row - c}, up | {row + c})
                   for c in range(n) if c not in columns and row - c not in down and row + c not in up)
    return place(0, frozenset(), frozenset(), frozenset())''',
    "duration.py": '''import re


def parse_duration(text):
    match = re.fullmatch(r"(?:(\\d+)h)?(?:(\\d+)m)?(?:(\\d+)s)?", text)
    if not text or not match:
        raise ValueError(text)
    hours, minutes, seconds = (int(part or 0) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds''',
}
SHOP_ANSWER = (
    BLOCK.format("models.py", "    quantity: int = 1", "    quantity: int = 1\n    discount: float = 0")
    + BLOCK.format("store.py", "    def add(self, name, price, quantity=1):\n        self.items.append(Item(name, price, quantity))",
                   "    def add(self, name, price, quantity=1, discount=0):\n"
                   "        self.items.append(Item(name, price, quantity, discount))")
    + BLOCK.format("store.py", "        return sum(item.price * item.quantity for item in self.items)",
                   "        return sum(item.price * item.quantity * (100 - item.discount) / 100 for item in self.items)")
    + BLOCK.format("report.py", '    return f"{item.name} x{item.quantity} @ {item.price:.2f}"',
                   '    text = f"{item.name} x{item.quantity} @ {item.price:.2f}"\n'
                   '    return text + (f" (-{item.discount}%)" if item.discount else "")'))


class Scripted:
    """A model that gives the listed answers in turn."""

    def __init__(self, *answers):
        self.name, self.label, self.detail, self.size, self.url = "scripted", "scripted", "", 8192, "script://"
        self.answers = list(answers)

    def enter(self):
        pass

    def leave(self):
        pass

    def count(self, messages):
        return 10

    def chat(self, messages, ui=None, session=None):
        ui.timings({"predicted_n": 5, "predicted_ms": 250.0})
        answer = self.answers.pop(0)
        return answer + "DONE\n" if "REPLACE" in answer else answer  # as the rules ask, edits end with DONE


GOOD = {  # for every task, answers that solve it
    "greet": ["Hello! How can I help?"],
    "read": ["NEED: joystick_viewer.py", "It opens a pygame window and draws a bar for each joystick axis."],
    "fact": ["NEED: include/util/money.h", "0.18"],
    "create": [BLOCK.format("fib.py", "", FIB)],
    "rename": [BLOCK.format("src/cart.cpp", bench.CART["src/cart.cpp"].strip(), RENAMED)],
    "method": [BLOCK.format("src/cart.h", "    double total() const;", "    bool isEmpty() const;\n    double total() const;")
               + BLOCK.format("src/cart.cpp", "double Cart::total() const {",
                              "bool Cart::isEmpty() const {\n    return items_.empty();\n}\n\ndouble Cart::total() const {")],
    "value": ["NEED: include/util/money.h",
              BLOCK.format("include/util/money.h", "const double kTaxRate = 0.18;", "const double kTaxRate = 0.20;")],
    "fix": [BLOCK.format("stats.py", "    return ordered[middle] + ordered[middle + 1] / 2",
                         "    return (ordered[middle - 1] + ordered[middle]) / 2")],
    "roman": [BLOCK.format("roman.py", "", SOLUTIONS["roman.py"])],
    "merge": [BLOCK.format("intervals.py", "", SOLUTIONS["intervals.py"])],
    "calc": [BLOCK.format("calc.py", "", SOLUTIONS["calc.py"])],
    "lru": [BLOCK.format("lru.py", bench.LRU["lru.py"].strip(), SOLUTIONS["lru.py"])],
    "trace": ["30 gives 30, then 10 gives 29, 9 gives 38, 3 gives 41, 1 gives 40. The result is 40."],
    "wrap": [BLOCK.format("wrap.py", "", SOLUTIONS["wrap.py"])],
    "rooms": [BLOCK.format("rooms.py", "", SOLUTIONS["rooms.py"])],
    "shop": [SHOP_ANSWER],
    "queens": [BLOCK.format("queens.py", "", SOLUTIONS["queens.py"])],
    "time": [BLOCK.format("duration.py", "", SOLUTIONS["duration.py"])],
    "tests": ["RUN: python -m pytest -q test_text.py",
              BLOCK.format("text.py", '    return "-".join(words)', '    return "-".join(words).lower()')
              + BLOCK.format("text.py", 'name.split(" "))', "name.split())")],
    "recall": ["Noted.", "It returns 2.0.", BLOCK.format("mode.py", "", 'def release_mode():\n    return "strict"')],
    "large": ["In Order.total, which calls apply_discount from pricing.py.",
              BLOCK.format("orders.py", "        total = apply_discount(self.subtotal(), self.discount)\n"
                                        "        return round(apply_discount(total, self.discount), 2)",
                           "        return round(apply_discount(self.subtotal(), self.discount), 2)"),
              BLOCK.format("orders.py", "    def subtotal(self):",
                           "    def item_count(self):\n        return sum(quantity for _, _, quantity in self.lines)\n\n"
                           "    def subtotal(self):")],
}
BAD = {  # for every task, answers that do not, with part of the expected reason
    "greet": (["NEED: joystick_viewer.py", "Hi!"], "2 model calls"),
    "read": (["It is a joystick viewer."], "no sign of the file's contents"),
    "fact": (["The rate is 0.2."], "does not give the rate"),
    "create": ([BLOCK.format("fib.py", "", "def fib(n):\n    return n")], "wrong numbers"),
    "rename": ([BLOCK.format("src/cart.cpp", "    double sum = 0;", "    double subtotal = 0;")], "not renamed"),
    "method": ([BLOCK.format("src/cart.h", "    double total() const;", "    bool isEmpty() const;\n    double total() const;")],
               "declared but not defined"),
    "value": (["I changed it to 20%."], "kTaxRate is not 0.20"),
    "fix": (["The bug is in median."], "AssertionError"),
    "roman": (["I would use a table of pairs."], "No module named 'roman'"),
    "merge": ([BLOCK.format("intervals.py", "", "def merge(intervals):\n    return sorted(intervals)")], "AssertionError"),
    "calc": ([BLOCK.format("calc.py", "", "def calc(text):\n    return eval(text)")], "SyntaxError"),
    "lru": (["The cache looks fine to me."], "AssertionError"),
    "trace": (["It returns 39."], "does not end in the right number"),
    "wrap": ([BLOCK.format("wrap.py", "", "def wrap(text, width):\n    return [text]")], "AssertionError"),
    "rooms": ([BLOCK.format("rooms.py", "", "def rooms(meetings):\n    return len(meetings)")], "AssertionError"),
    "shop": ([BLOCK.format("models.py", "    quantity: int = 1", "    quantity: int = 1\n    discount: float = 0")],
             "discount"),
    "queens": ([BLOCK.format("queens.py", "", "def solutions(n):\n    return 1")], "AssertionError"),
    "time": ([BLOCK.format("duration.py", "", "def parse_duration(text):\n    return int(text[:-1])")], "AssertionError"),
    "tests": (["The tests look fine to me."], "AssertionError"),
    "recall": (["Noted.", "2.0", BLOCK.format("mode.py", "", 'def release_mode():\n    return "debug"')],
               "release_mode() is not strict"),
    "large": (["In Order.total.", "The test looks wrong to me.", "I would add a loop."], "AssertionError"),
}


class Tasks(unittest.TestCase):
    def test_every_task_passes_with_a_right_answer_and_fails_with_a_wrong_one(self):
        self.assertEqual(set(GOOD), set(bench.TASKS))
        home = os.getcwd()
        for name in bench.TASKS:
            with self.subTest(task=name):
                run = bench.run_task(Scripted(*GOOD[name]), name)
                self.assertEqual(run.reason, "")
                self.assertEqual((run.calls, run.tokens), (len(GOOD[name]), 5 * len(GOOD[name])))
                answers, reason = BAD[name]
                self.assertIn(reason, bench.run_task(Scripted(*answers), name).reason)
        self.assertEqual(os.getcwd(), home)  # every run went back to where it started

    def test_task_sets(self):
        self.assertEqual(len(bench.EASY), 8)
        self.assertEqual(len(bench.HARD), 12)
        self.assertEqual(set(bench.EASY) | set(bench.HARD) | set(bench.LONG), set(bench.TASKS))

    def test_rewriting_the_test_file_does_not_pass_a_task(self):
        answer = BLOCK.format("test_roman.py", bench.ROMAN["test_roman.py"].strip(), 'print("ok")')
        self.assertIn("No module named 'roman'", bench.run_task(Scripted(answer), "roman").reason)

    def test_using_eval_fails_the_calculator_even_when_the_sums_are_right(self):
        cheat = ("def calc(text):\n    try:\n        return eval(text)\n    except Exception:\n"
                 "        raise ValueError(text)")
        self.assertIn("must not use eval or exec", bench.run_task(Scripted(BLOCK.format("calc.py", "", cheat)), "calc").reason)

    def test_an_edit_that_cannot_be_applied_fails_the_task(self):
        failing = BLOCK.format("src/cart.cpp", "no such line", "x")
        run = bench.run_task(Scripted(failing, failing), "rename")  # the failure is retried once
        self.assertIn("could not apply an edit to src/cart.cpp", run.reason)

    def test_other_lines_changing_fails_the_rename(self):
        answer = BLOCK.format("src/cart.cpp", bench.CART["src/cart.cpp"].strip(), RENAMED.replace("#include <vector>", ""))
        self.assertEqual(bench.run_task(Scripted(answer), "rename").reason, "other lines changed too")

    def test_an_inline_definition_counts_for_the_method_task(self):
        answer = BLOCK.format("src/cart.h", "    double total() const;",
                              "    bool isEmpty() const { return items_.empty(); }\n    double total() const;")
        self.assertEqual(bench.run_task(Scripted(answer), "method").reason, "")


class ClaudeCommand(unittest.TestCase):
    def setUp(self):
        bench.Claude.isolated = None
        self.calls = []
        self.results = []  # what the fake claude command prints, one per call

        really_run = bench.subprocess.run

        def run(command, **options):
            if command[0] != "claude":  # the benchmark running Python to check a result
                return really_run(command, **options)
            system = ""
            if "--system-prompt-file" in command:
                with open(command[command.index("--system-prompt-file") + 1], encoding="utf-8") as f:
                    system = f.read()
            self.calls.append((command, system, options["input"].decode("utf-8"), options["cwd"]))
            result = self.results.pop(0)
            if callable(result):  # a fake Claude Code that does something in the folder before answering
                result = result()
            return mock.Mock(returncode=0, stdout=json.dumps(result).encode(), stderr=b"")

        patch = mock.patch.object(bench.subprocess, "run", run)
        patch.start()
        self.addCleanup(patch.stop)

    def test_command_line_prompt_and_result(self):
        self.results = [{"result": "Hello", "usage": {"input_tokens": 7, "output_tokens": 3}, "total_cost_usd": 0.02}]
        model, seen = bench.Claude("opus", "xhigh"), []
        ui = mock.Mock(timings=seen.append)
        messages = [{"role": "system", "content": "RULES"}, {"role": "user", "content": "code\n\nfirst"},
                    {"role": "assistant", "content": "an answer"}, {"role": "user", "content": "second"}]
        self.assertEqual(model.chat(messages, ui), "Hello")
        command, system, prompt, folder = self.calls[0]
        self.assertEqual(command[:11] + command[12:], [
            "claude", "-p", "--model", "opus", "--setting-sources", "", "--strict-mcp-config",
            "--no-session-persistence", "--output-format", "json", "--system-prompt-file", "--tools", "",
            "--effort", "xhigh"])
        self.assertEqual(system, "RULES")
        self.assertEqual(prompt, "code\n\nfirst\n\n[You answered:]\nan answer\n\n[The user now says:]\nsecond")
        self.assertEqual(os.path.dirname(command[11]), folder)  # runs in the temporary folder holding only that file
        self.assertFalse(os.path.exists(folder))
        self.assertEqual(seen, [{"cache_n": 0, "prompt_n": 7, "predicted_n": 3}])
        self.assertEqual((model.label, model.size, model.cost), ("claude · opus · xhigh", bench.WINDOW, 0.02))
        self.results.append({"result": "x"})
        bench.Claude("opus").chat(messages[:2], ui)  # default effort, and no earlier exchange
        self.assertNotIn("--effort", self.calls[1][0])
        self.assertEqual(self.calls[1][2], "code\n\nfirst")

    def test_a_longer_conversation_is_folded_in_order(self):
        roles = ["user", "assistant", "user", "assistant", "user"]
        messages = [{"role": role, "content": f"m{number}"} for number, role in enumerate(roles)]
        self.assertEqual(bench.fold(messages), "m0\n\n[You answered:]\nm1\n\n[The user now says:]\nm2\n\n"
                                               "[You answered:]\nm3\n\n[The user now says:]\nm4")

    def test_claude_code_works_in_the_task_folder_with_its_own_tools(self):
        def writes_the_file():
            with open("fib.py", "w", encoding="utf-8") as f:  # what Claude Code's own edit tool would do
                f.write(FIB + "\n")
            return {"result": "Created fib.py.", "num_turns": 3, "usage": {"output_tokens": 40}, "total_cost_usd": 0.05}

        model = bench.ClaudeCode("opus", "xhigh")
        self.results = [writes_the_file]
        run = bench.run_task(model, "create")
        self.assertEqual((run.reason, run.calls, run.tokens, run.answer), ("", 3, 40, "Created fib.py."))
        command, system, prompt, folder = self.calls[0]
        self.assertEqual(command, [
            "claude", "-p", "--model", "opus", "--setting-sources", "", "--strict-mcp-config",
            "--no-session-persistence", "--output-format", "json", "--permission-mode", "acceptEdits",
            "--allowedTools", "Bash(python *)", "PowerShell(python *)", "--effort", "xhigh"])
        self.assertEqual((system, folder), ("", None))  # Claude Code's own instructions, in the current (task) folder
        self.assertEqual(prompt, bench.TASKS["create"][1][0])
        self.assertEqual((model.label, model.cost), ("claude code · opus · xhigh", 0.05))
        self.results = [{"result": "I could not do it.", "num_turns": 1}]
        self.assertEqual(bench.run_task(model, "create").reason, "fib.py was not created")
        self.results = [{"result": "usage limit reached", "is_error": True}]
        self.assertIn("the claude command failed: usage limit reached", bench.run_task(model, "create").reason)

    def test_failed_calls_raise(self):
        for result in ({"result": "limit reached", "is_error": True}, {"type": "result"}):
            self.results = [result]
            with self.assertRaisesRegex(OSError, "the claude command failed"):
                bench.Claude("opus").chat([{"content": "s"}, {"content": "u"}], mock.Mock())

    def test_isolation_is_checked_once_and_a_leak_stops_the_row(self):
        self.results = [{"result": "TOOLS=0\nMEMORY=no"}]
        bench.Claude("opus").enter()
        bench.Claude("opus", "xhigh").enter()  # not asked again
        self.assertEqual(len(self.calls), 1)
        self.assertNotIn("--effort", self.calls[0][0])
        bench.Claude.isolated = None
        self.results = [{"result": "TOOLS=8\nMEMORY=no"}]
        with self.assertRaisesRegex(OSError, "not isolated; the model reported: TOOLS=8 MEMORY=no"):
            bench.Claude("opus").enter()

    def test_count_uses_the_local_tokenizer_or_an_estimate(self):
        messages = [{"content": "x" * 30}]
        with mock.patch.object(bench.Claude, "tokenizer", "http://llama"):
            with mock.patch.object(tolken, "count", lambda messages, url: 7 if url == "http://llama" else 0):
                self.assertEqual(bench.Claude("opus").count(messages), 7)
            with mock.patch.object(tolken, "count", side_effect=OSError):
                self.assertEqual(bench.Claude("opus").count(messages), 10)
        with mock.patch.object(bench.Claude, "tokenizer", None):  # no llama-server listed
            self.assertEqual(bench.Claude("opus").count(messages), 10)


class Report(unittest.TestCase):
    def run_with(self, reason="", seconds=2.0, calls=1, tokens=10, milliseconds=500.0, answer="an answer",
                 prompt_ms=0.0):
        run = bench.Run()
        run.reason, run.seconds, run.calls, run.tokens, run.milliseconds, run.answer = (
            reason, seconds, calls, tokens, milliseconds, answer)
        run.prompt_ms = prompt_ms
        return run

    def test_table(self):
        results = {
            "local": {"greet": [self.run_with(prompt_ms=300), self.run_with(prompt_ms=500)],
                      "create": [self.run_with(prompt_ms=1000), self.run_with("fib.py was not created", 4.0, 3, prompt_ms=1000)]},
            "claude · opus": {"greet": [self.run_with(milliseconds=0)] * 2, "create": [self.run_with(milliseconds=0)] * 2},
        }
        self.assertEqual(bench.table(results, ["greet", "create"]).splitlines(), [
            "model          passed  score   greet  create  s/task  calls  prompt s  tok/s",
            "local             3/4    75%     2/2     1/2     2.5    1.5       0.7     20",
            "claude · opus     4/4   100%     2/2     2/2     2.0    1.0         -      -",
            "",
            "Failed runs:",
            "- local · create · run 2: fib.py was not created",
            "    answer: an answer",
        ])
        self.assertTrue(bench.table({"local": {"greet": [self.run_with()]}}, ["greet"]).endswith("No failed runs."))

    def test_resolve_names(self):
        found = [tolken.Llama(8192, "demo.gguf"), tolken.Ollama("qwen3:8b")]
        with mock.patch.object(tolken, "models", lambda: found), mock.patch("builtins.print") as printed, \
                mock.patch.object(bench.Claude, "tokenizer", None):
            rows = bench.resolve(["llama.cpp", "qwen3:8b", "opus", "opus:xhigh", "cc:opus", "cc:opus:xhigh", "nothing",
                                  "cc:claude-opus-4-8:xhigh"])
            self.assertEqual(bench.Claude.tokenizer, found[0].url)  # the Claude rows count with llama-server's tokenizer
        self.assertEqual([row.label for row in rows],
                         ["llama.cpp · demo", "ollama · qwen3:8b", "claude · opus", "claude · opus · xhigh",
                          "claude code · opus", "claude code · opus · xhigh", "claude code · claude-opus-4-8 · xhigh"])
        self.assertEqual((rows[6].model, rows[6].effort), ("claude-opus-4-8", "xhigh"))
        self.assertEqual([type(row).__name__ for row in rows[2:]],
                         ["Claude", "Claude", "ClaudeCode", "ClaudeCode", "ClaudeCode"])
        self.assertEqual((rows[2].effort, rows[3].effort, rows[5].effort), (None, "xhigh", "xhigh"))
        printed.assert_called_once_with("[no single model matches 'nothing'; skipped]")

    def test_the_benchmark_runs_with_its_own_settings_unless_told_to_read_the_file(self):
        with mock.patch.object(bench, "resolve", return_value=[]), mock.patch("builtins.print"), \
                mock.patch.object(tolken, "apply_settings") as applied, \
                mock.patch.object(tolken, "load_settings") as loaded:
            with mock.patch.object(bench.sys, "argv", ["bench.py"]):
                bench.main()
            applied.assert_called_once_with(bench.SETTINGS)
            loaded.assert_not_called()
            with mock.patch.object(bench.sys, "argv", ["bench.py", "--settings"]):
                bench.main()
            loaded.assert_called_once_with()
            applied.assert_called_once()  # not again
        self.assertEqual((bench.SETTINGS["think"], bench.SETTINGS["max_out"], bench.SETTINGS["options"]),
                         (False, 2000, {"temperature": 0.2}))  # what BENCHMARKS.md says the figures are for


if __name__ == "__main__":
    unittest.main()
