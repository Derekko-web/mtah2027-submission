"""Run every test file, each in its own Python process.

    python tests/run_all.py                 # all files
    python tests/run_all.py pff metrica     # only files whose path contains one of these words
    python tests/run_all.py --verbose       # also show each file's output when it passes

Torch and XGBoost each bring an OpenMP runtime, and loading both in one process can hang or crash,
so no two test files share a process. The runner needs no pytest. It collects, from
`tests/test_*.py`, `dfl_submission/test_*.py` and `experiments/test_*.py`:
- plain `test_*` functions defined in the file; a `tmp_path` parameter gets a fresh temporary
  directory, as under pytest;
- `unittest.TestCase` classes defined in the file.

It prints one line per file and a total, and exits non-zero if any test fails. Every child process
runs with `MTAH_DATA_DIR` set to a path that does not exist: no test needs the competition bundle.
"""

import argparse
import importlib.util
import inspect
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import unittest

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = ("tests/test_*.py", "dfl_submission/test_*.py", "experiments/test_*.py")
ABSENT_DATA = "/nonexistent/mtah-run-all"
RESULT = "RUN_ALL_RESULT"


def files():
    return [path for pattern in PATTERNS for path in sorted(ROOT.glob(pattern))]


def load(path):
    """Import a test file by path under its dotted name, e.g. `tests.test_pff`."""
    sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]
    name = ".".join(path.relative_to(ROOT).with_suffix("").parts)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def call(function):
    parameters = list(inspect.signature(function).parameters)
    if not parameters:
        return function()
    if parameters == ["tmp_path"]:
        with tempfile.TemporaryDirectory(prefix="run_all_") as folder:
            return function(Path(folder))
    raise TypeError(f"unsupported test parameters {parameters}; only tmp_path is provided")


def run_file(path):
    """Child process: run one file's tests and print a result line for the parent."""
    passed, failed, skipped = 0, [], 0
    try:
        module = load(path)
    except BaseException:
        traceback.print_exc()
        print(f"{RESULT} 0 1 0", flush=True)
        return 1
    functions = [(name, value) for name, value in vars(module).items()
                 if name.startswith("test_") and inspect.isfunction(value) and value.__module__ == module.__name__]
    for name, function in functions:
        try:
            call(function)
            passed += 1
        except unittest.SkipTest as reason:
            skipped += 1
            print(f"skipped {name}: {reason}", flush=True)
        except BaseException:
            failed.append(name)
            print(f"FAILED {name}", flush=True)
            traceback.print_exc()
    classes = [value for value in vars(module).values() if inspect.isclass(value)
               and issubclass(value, unittest.TestCase) and value.__module__ == module.__name__]
    for case in classes:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(case)
        outcome = unittest.TextTestRunner(stream=sys.stdout, verbosity=0).run(suite)
        broken = [test.id() for test, _ in outcome.failures + outcome.errors + outcome.unexpectedSuccesses]
        failed += broken
        skipped += len(outcome.skipped)
        passed += outcome.testsRun - len(broken) - len(outcome.skipped)
    print(f"{RESULT} {passed} {len(failed)} {skipped}", flush=True)
    return 1 if failed else 0


def parse(output):
    for line in reversed(output.splitlines()):
        if line.startswith(RESULT):
            return tuple(int(value) for value in line.split()[1:])
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("words", nargs="*", help="run only files whose path contains one of these")
    parser.add_argument("--verbose", action="store_true", help="show each file's output, not only failures")
    parser.add_argument("--timeout", type=float, default=3600, help="seconds allowed per file")
    parser.add_argument("--child", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        sys.exit(run_file(args.child))

    selected = [path for path in files()
                if not args.words or any(word in str(path.relative_to(ROOT)) for word in args.words)]
    if not selected:
        sys.exit("no test files match")
    environment = dict(os.environ, MTAH_DATA_DIR=ABSENT_DATA, PYTHONDONTWRITEBYTECODE="1")
    totals, broken_files = [0, 0, 0], []
    for path in selected:
        relative = path.relative_to(ROOT)
        started = time.monotonic()
        try:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", str(path)],
                                    cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, timeout=args.timeout)
            output, code = result.stdout, result.returncode
        except subprocess.TimeoutExpired as error:
            partial = error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else error.stdout or ""
            output, code = f"{partial}\ntimed out after {args.timeout:.0f} s", -1
        seconds = time.monotonic() - started
        counts = parse(output)
        if counts is None:
            counts = (0, 1, 0)
            output += f"\nno result line (exit code {code})"
        passed, failed, skipped = counts
        ok = failed == 0 and code == 0
        for index, value in enumerate(counts):
            totals[index] += value
        if not ok:
            broken_files.append(str(relative))
        extra = f", {skipped} skipped" if skipped else ""
        verdict = "PASS" if ok else "FAIL"
        print(f"{verdict}  {relative}  {passed} passed, {failed} failed{extra}  ({seconds:.1f} s)", flush=True)
        if args.verbose or not ok:
            shown = [line for line in output.splitlines() if not line.startswith(RESULT)]
            print("\n".join("      " + line for line in shown), flush=True)
    extra = f", {totals[2]} skipped" if totals[2] else ""
    print(f"\nTOTAL  {len(selected)} files: {totals[0]} passed, {totals[1]} failed{extra}"
          f"; {len(selected) - len(broken_files)}/{len(selected)} files pass", flush=True)
    if broken_files:
        print("Failing files: " + ", ".join(broken_files), flush=True)
    sys.exit(1 if broken_files else 0)


if __name__ == "__main__":
    main()
