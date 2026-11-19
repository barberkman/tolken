# Benchmarks

How well each model does small coding tasks through tolken, compared with Claude Opus working as Claude Code. Measured on 2026-11-17 with `bench.py`.

## Result

The harder task set: 11 tasks, 3 runs each, 33 runs per model.

| Model | Passed | Score | Seconds per task | Calls per task | Answer speed |
|---|---|---|---|---|---|
| Opus 5.5, `xhigh`, as Claude Code | 33 of 33 | 100% | 29.2 | 5.3 | not recorded |
| Opus 4.8, `xhigh`, as Claude Code | 33 of 33 | 100% | 34.6 | 5.5 | 75 tok/s |
| Opus 4.8, `low`, as Claude Code | 33 of 33 | 100% | 22.2 | 5.2 | 66 tok/s |
| Opus 5.5, `xhigh`, inside tolken | 33 of 33 | 100% | 9.6 | 1.1 | not measured |
| Ollama on a network server, `bonsai-2:27b`, 32,768-token window, thinking on | 26 of 33 | 78% | 37.7 | 1.1 | 47 tok/s |
| llama-server, Bonsai-2 27B, 8,192-token window | 23 of 33 | 69% | 8.4 | 1.1 | 36 tok/s |
| Ollama, `bonsai-2:27b`, 32,768-token window | 23 of 33 | 69% | 51.6 | 1.1 | 5 tok/s |
| Ollama, `qwen3:8b` | 13 of 33 | 39% | 12.2 | 1.0 | 23 tok/s |
| Ollama, `bonsai-2:latest` (9B) | 11 of 33 | 33% | 4.7 | 1.0 | 101 tok/s |

- The local 27B model solves about two thirds of what Opus solves on these tasks, and is the fastest per task.
- It scores the same in llama-server and in Ollama. In Ollama it is about seven times slower, because Ollama runs it with a 32,768-token window, which does not fit an 8 GB card.
- Every Claude row is at 100%, so this set ranks the local models against Claude but cannot separate Claude models or effort levels from each other.
- With thinking on, the 27B model passes three more runs, and is about four and a half times slower per task than in llama-server without thinking. It solves `time`, which no local model did before, and loses `calc` and `wrap` (see below).

### Passes per task, out of 3

| Task | What it asks | Claude rows | 27B thinking | llama 27B | Ollama 27B | `qwen3:8b` | 9B |
|---|---|---|---|---|---|---|---|
| `fix` | fix a wrong median | 3 | 2 | 0 | 0 | 0 | 3 |
| `roman` | Roman numerals, both ways | 3 | 3 | 3 | 2 | 3 | 1 |
| `merge` | merge overlapping intervals | 3 | 3 | 3 | 3 | 1 | 0 |
| `calc` | arithmetic parser without `eval` | 3 | 0 | 2 | 2 | 0 | 0 |
| `lru` | fix a least-recently-used cache | 3 | 3 | 3 | 3 | 0 | 0 |
| `trace` | say what a function returns | 3 | 3 | 3 | 3 | 3 | 3 |
| `wrap` | word wrap with long-word splitting | 3 | 1 | 3 | 3 | 0 | 0 |
| `rooms` | minimum meeting rooms | 3 | 3 | 2 | 2 | 3 | 1 |
| `shop` | one feature across three files | 3 | 3 | 1 | 2 | 0 | 0 |
| `queens` | count N-queens solutions | 3 | 3 | 3 | 3 | 3 | 3 |
| `time` | parse `1h30m15s` into seconds | 3 | 2 | 0 | 0 | 0 | 0 |

"Claude rows" covers all four Claude rows of the first table; each passed every run.

## 64k context on the network Ollama

Measured on 2026-11-18 with `bonsai-2:27b` on the network Ollama at a 65,536-token window, with the settings in `settings.json`: thinking on, temperature 0.6, `max_out` 8,000. "Before" is tolken at commit `318a1f7`, which kept one previous exchange and built every prompt afresh. "After" keeps the conversation and continues the previous call's prompt while it fits (see the top of `tolken.py`). 3 runs per task.

| | Before | After |
|---|---|---|
| `recall`: use a fact told two messages earlier | 0 of 3 | 3 of 3 |
| `large`: three messages on a 27-file project | 3 of 3 | 3 of 3 |
| The harder set | 27 of 33 | 27 of 33 |
| Prompt seconds per task, harder set | 0.6 | 0.6 |
| Prompt seconds per task, `recall` and `large` | 1.6 | 1.2 |
| Seconds per run, harder set without `calc` | 36.5 | 34.5 |

- **`recall` before:** every run answered that it had no release mode, because only the previous exchange was kept.
- **`calc`:** failed all three runs after and two before. After, the model thought until `max_out` ran out, about 205 seconds each, and wrote nothing. That raised the harder set's average from 43.1 to 50.0 seconds per task. One run before failed the same way.
- **Folder named after the project:** two failures after (`roman`, `time`) and one before (`calc`) wrote the file under a folder with the project's name (`tmpXXXX/roman.py`), taken from the prompt's `Project folder:` line.
- **The rest:**
  - `fix` failed once after and twice before; the model fixed only the precedence.
  - `wrap` failed once before, with wrong code.
  - `shop` failed once before: the model wrote an empty edit block for `report.py`, which tolken read as deleting it.
- **The saving from the cache hardly shows here,** because these tasks' prompts are small. It grows with the length of a conversation.

### The thinking row's failures

- **`calc`:** in two runs the model thought for the whole 8,000 tokens (about 27,000 characters) and never wrote an answer. The third wrote a parser that failed the test. An earlier run with `max_out` at 2,000 was cut off the same way in all three `calc` runs, after about 44 seconds each.
- **`wrap`:** in two runs the code was right in shape, but the model put the file under a folder named after the project (`tmpgfjy642_/wrap.py`), taking the prompt's `Project folder:` line as part of the path. tolken created that folder, so the test could not import `wrap`.
- **`fix` and `time`:** one wrong answer each.

## After the R1-R9 changes

Measured on 2026-11-19 with `bonsai-2:27b` on the network Ollama, with the settings in `settings.json`: a 65,536-token window, thinking on, temperature 0.6, `max_out` 16,000, no `think_max`. "Before" is commit `14408dd`; "after" adds the shell-syntax check, the notes about what the model was not told, `/undo`, sharper edit failures, changed parts only, and the Ollama warning. Both ran the same `bench.py`, whose hard set now has a twelfth task, `tests`, which asks the model to run the tests and fix what fails. 3 runs per task.

| | Before | After |
|---|---|---|
| Passed | 34 of 36 | 34 of 36 |
| Seconds per run | 52.1 | 78.2 |
| Seconds per run without `calc`, `wrap` and `time` | 40.3 | 38.7 |
| Calls per task | 1.9 | 2.0 |
| Prompt seconds per task | 1.1 | 1.2 |

| Task | Before | After |
|---|---|---|
| `fix` | 2 | 3 |
| `merge` | 2 | 3 |
| `shop` | 3 | 2 |
| `tests` | 3 | 2 |
| The other eight | 3 each | 3 each |

- **The time difference is thinking.** Two `calc` runs took 519 and 444 seconds after, against 67 to 299 before. One `wrap` run (231 s) and one `time` run (161 s) were single calls. Without those three tasks the two versions take the same time. `think_max` would cap such runs.
- **Failures before:** `fix` fixed only the precedence; `merge` wrote wrong code.
- **Failures after:**
  - `tests`: the model ran `python test_text.py`, which is not on the allowed list, so the benchmark refused it and the message ended.
  - `shop`: the model wrote an empty SEARCH for an existing file after taking a file for corrupted (next point).
- **A bug both versions had, fixed since:** a file found by the first search was kept as a line range, which may be the whole file. When an edit made the file longer, it was shown again cut at its old end, as "lines 1-9 of 10", and the model could decide the file was broken. A trace of four more `shop` runs showed this. An edited file is now shown whole from then on. The changed-parts form did not appear in those runs, since the files are small.

## The easy set

Eight simpler tasks (greeting, reading a file before answering, a fact from an included header, creating a file, renaming, adding a method across two files, changing a value, the median fix), 3 runs each. They were too easy to rank anything, which is why the harder set exists.

| Model | Passed | Seconds per task |
|---|---|---|
| llama-server, Bonsai-2 27B | 21 of 24 | 4.0 |
| Opus 5.5, default effort, inside tolken | 24 of 24 | 8.3 |
| Opus 5.5, `xhigh`, inside tolken | 24 of 24 | 9.3 |

The local model's three failures were all the median fix.

## How it was measured

- **Tasks:** each runs in a throwaway project of at most four small files. Every task's first prompt is under 1,000 tokens, so all of them fit the 8,192-token window. Most tasks ship a test file that defines success; a run passes when that test passes. The test file is restored before checking, so a model cannot pass by editing it.
- **Local models:** run through tolken (`tolken.Session`), with thinking off, temperature 0.2 and `max_out` 2,000. Every edit the model offers is accepted. The thinking row used the settings in `settings.json` instead: thinking on, temperature 0.6 and `max_out` 8,000, which thinking counts against.
- **Opus as Claude Code:** the `claude` command (version 2.1.326) works in the task's folder with Claude Code's own tools. It may read, create and edit files there and run Python to try the tests. User settings, `CLAUDE.md` and connectors are left out.
- **Opus inside tolken:** the same prompts the local models get, with no tools.
- **Machine:** NVIDIA RTX 3070, 8 GB. The llama-server row ran with no Ollama model loaded, and the Ollama rows ran with llama-server stopped, so each had the GPU to itself. The thinking row ran on Ollama on another computer on the network, where the whole model (10 GB at a 32,768-token window) fit in GPU memory.
- **Servers:** llama-server with `-c 8192` and a `q8_0` KV cache; Ollama 0.40.16 at its own context setting.

## Reading the numbers

- **Score is specific to tolken and these tasks.** It is not a general measure of a model.
- **Three runs per task is a small sample.** A difference of one or two runs is noise.
- **The 9B row understates that model.** Thirteen of its 22 failures were edits tolken could not read: the model closed the edit with a second `=======` line where `>>>>>>> REPLACE` belongs. The code was there, but the block was rejected.
- **Answer speed** for local models is the generation speed their server reports. For Claude Code it is output tokens (thinking included) per second of model time, so the two are not directly comparable. It was not recorded for the first Opus 5.5 run.
- **Seconds per task** for Claude includes starting the `claude` command and the network.
- **Effort is not equal:** the local models ran with thinking off, apart from the thinking row.
- **The thinking row ran on other hardware**, so its speed cannot be compared with the other local rows; its score can.

## Cost of the Claude rows

As reported by the `claude` command for the harder set:

| Row | Reported |
|---|---|
| Opus 5.5, `xhigh`, as Claude Code | $3.48 |
| Opus 4.8, `xhigh`, as Claude Code | $6.41 |
| Opus 4.8, `low`, as Claude Code | $4.91 |
| Opus 5.5, `xhigh`, inside tolken | $0.81 |

## Running it again

```
python bench.py --hard                                    # llama-server's model and Opus xhigh as Claude Code
python bench.py --hard --models bonsai-2:27b qwen3:8b     # Ollama models, named as in /model
python bench.py --hard --models cc:claude-opus-4-8:low    # another Claude model or effort, as Claude Code
python bench.py --hard --models opus:xhigh                # Claude inside tolken
python bench.py --runs 1 --tasks fix time --save report.md
python bench.py --tasks recall large --models bonsai-2:27b   # the longer tasks, several messages each
```

`bench.py` does not read `settings.json`. The thinking row and the 64k rows were run with the settings loaded first:

```
python -c "import sys, tolken, bench; tolken.load_settings(); sys.argv[1:] = ['--hard', '--models', 'bonsai-2:27b']; bench.main()"
```

Without `--hard` it runs the easy set. Stop llama-server before benchmarking Ollama models, and the other way round, or the two will share the GPU and the speed figures will be wrong.
