"""Run every test suite and print a summary table; exit nonzero on any failure.

    python scripts/run_tests.py [--blender PATH] [--skip-blender] [--only NAME ...] [--verbose]

Plain suites run with this interpreter and PYTHONPATH set to the project root.
Blender suites run as ``blender --background --factory-startup
--python-exit-code 1 --python <suite>``; Blender is taken from ``--blender``,
then the BLENDER environment variable, then the default Windows install path.
Optional suites are skipped when their file does not exist. A suite passes only
when its process exits 0 and unittest printed a "Ran N tests" line with N > 0,
so a suite that silently ran nothing is reported as a failure.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe")
# (path relative to ROOT, runner, required)
SUITES = (
    ("tests/test_qem.py", "python", True),
    ("tests/test_raster.py", "python", True),
    ("tests/test_perception.py", "python", True),
    ("tests/test_analysis.py", "python", False),
    ("tests/test_user_study.py", "python", False),
    ("tests/test_metrics_blender.py", "blender", True),
    ("tests/test_perception_blender.py", "blender", False),
    ("tests/test_integration_blender.py", "blender", True),
)
RAN = re.compile(r"^Ran (\d+) tests? in ([\d.]+)s", re.MULTILINE)
UNITTEST_FAILED = re.compile(r"^FAILED \((.+)\)", re.MULTILINE)
UNITTEST_SKIPPED = re.compile(r"^OK \(skipped=(\d+)\)", re.MULTILINE)


def find_blender(option: str | None) -> Path | None:
    if option:  # an explicit path is never silently replaced by another Blender
        return Path(option) if Path(option).is_file() else None
    for candidate in (os.environ.get("BLENDER"), str(DEFAULT_BLENDER), shutil.which("blender")):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def run_suite(path: Path, runner: str, blender: Path | None, timeout: float) -> dict:
    if runner == "python":
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT), environment.get("PYTHONPATH"))))
        command = [sys.executable, str(path)]
    else:
        # Blender ignores PYTHONPATH by default; the Blender suites add the root themselves.
        environment = None
        command = [str(blender), "--background", "--factory-startup", "--python-exit-code", "1", "--python", str(path)]
    started = time.perf_counter()
    try:
        completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=timeout)
        output, code = completed.stdout + completed.stderr, completed.returncode
    except subprocess.TimeoutExpired as exc:
        output = f"{exc.stdout or ''}{exc.stderr or ''}\nTIMEOUT after {timeout:.0f} s"
        code = None
    seconds = time.perf_counter() - started
    ran = RAN.findall(output)
    tests = sum(int(count) for count, _ in ran) if ran else None
    failed = UNITTEST_FAILED.search(output)
    skipped = UNITTEST_SKIPPED.search(output)
    if code is None:
        status = "TIMEOUT"
    elif code != 0:
        status = "FAIL"
    elif not tests:
        status = "FAIL"
        output += "\nrun_tests: exit code 0 but no unittest 'Ran N tests' line with N > 0"
    else:
        status = "OK"
    detail = failed.group(1) if failed else (f"skipped={skipped.group(1)}" if skipped else "")
    if code not in (0, None) and not detail:
        detail = f"exit code {code}"
    return dict(status=status, tests=tests, seconds=seconds, detail=detail, output=output, command=command)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--blender", help="Blender executable (default: $BLENDER, then the Windows install path)")
    parser.add_argument("--skip-blender", action="store_true", help="run only the plain-Python suites")
    parser.add_argument("--only", nargs="+", metavar="NAME",
                        help="run only suites whose file name contains one of these strings")
    parser.add_argument("--timeout", type=float, default=1800.0, help="seconds per suite (default 1800)")
    parser.add_argument("--verbose", action="store_true", help="print the full output of every suite")
    arguments = parser.parse_args(argv)

    blender = None if arguments.skip_blender else find_blender(arguments.blender)
    results = []
    for relative, runner, required in SUITES:
        if arguments.only and not any(name in relative for name in arguments.only):
            continue
        path = ROOT / relative
        if not path.is_file():
            results.append((relative, runner, dict(status="MISSING" if required else "SKIP", tests=None,
                                                   seconds=0.0, detail="file not found", output="")))
            continue
        if runner == "blender" and arguments.skip_blender:
            results.append((relative, runner, dict(status="SKIP", tests=None, seconds=0.0,
                                                   detail="--skip-blender", output="")))
            continue
        if runner == "blender" and blender is None:
            results.append((relative, runner, dict(status="NO-BLENDER", tests=None, seconds=0.0,
                                                   detail="pass --blender PATH or set BLENDER", output="")))
            continue
        print(f"running {relative} ({runner}) ...", flush=True)
        result = run_suite(path, runner, blender, arguments.timeout)
        results.append((relative, runner, result))
        if arguments.verbose or result["status"] != "OK":
            print(result["output"] if arguments.verbose else "\n".join(result["output"].splitlines()[-40:]))

    print()
    print(f"Python  : {sys.executable} ({sys.version.split()[0]})")
    print(f"Blender : {blender if blender else 'not used'}")
    header = f"{'suite':<36} {'runner':<8} {'tests':>5}  {'result':<10} {'seconds':>8}  detail"
    print(header)
    print("-" * len(header))
    for relative, runner, result in results:
        tests = "-" if result["tests"] is None else str(result["tests"])
        print(f"{relative:<36} {runner:<8} {tests:>5}  {result['status']:<10} {result['seconds']:>8.1f}  {result['detail']}")
    bad = [relative for relative, _, result in results if result["status"] not in ("OK", "SKIP")]
    total = sum(result["tests"] or 0 for _, _, result in results)
    print("-" * len(header))
    print(f"{total} tests in {sum(1 for *_, r in results if r['status'] == 'OK')} passing suites; "
          + (f"FAILED: {', '.join(bad)}" if bad else "all suites passed"))
    return 1 if bad or not results else 0


if __name__ == "__main__":
    raise SystemExit(main())
