"""Isolated Python execution sandbox for the Data Analyst agent.

Runs untrusted code in a subprocess under ``python -I`` (isolated mode:
empty environment, no user site-packages, no PYTHONPATH influence) inside a
throwaway working directory, with a hard wall-clock timeout.

Conventions the generated code must follow:
  * final machine-readable findings  -> write ``analysis_result.json`` (JSON)
  * charts                           -> save PNGs to the ``charts/`` subfolder
"""

import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from app.config import get_settings

_BOOTSTRAP = (
    "import os, sys\n"
    "os.chdir(os.path.dirname(os.path.abspath(__file__)))\n"
    "try:\n"
    "    import matplotlib\n"
    "    matplotlib.use('Agg')\n"
    "except Exception:\n"
    "    pass\n"
)

_RESULT_FILENAME = "analysis_result.json"
_CHARTS_DIR = "charts"


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def run_code(
    code: str,
    timeout: int | None = None,
    files: dict[str, str] | None = None,
) -> dict:
    """Execute *code* (plus optional seed *files*) in a sandboxed interpreter.

    Returns a dict with keys: stdout, stderr, result, charts, ok, error.
    """
    settings = get_settings()
    timeout = timeout or settings.sandbox_timeout_seconds
    max_bytes = settings.sandbox_max_output_bytes
    python = settings.python_executable or sys.executable

    with tempfile.TemporaryDirectory(prefix="agent_sandbox_") as tmpdir:
        workdir = Path(tmpdir)
        script_path = workdir / "script.py"
        (workdir / _CHARTS_DIR).mkdir(exist_ok=True)

        for name, content in (files or {}).items():
            (workdir / name).write_text(content, encoding="utf-8")

        script_path.write_text(
            _BOOTSTRAP + code.replace("\r\n", "\n"),
            encoding="utf-8",
        )

        try:
            proc = subprocess.run(
                [python, "-I", str(script_path)],
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Execution timed out after {timeout}s.",
                "result": None,
                "charts": [],
                "ok": False,
                "error": "timeout",
            }
        except OSError as exc:
            return {
                "stdout": "",
                "stderr": f"Could not launch sandbox interpreter: {exc}",
                "result": None,
                "charts": [],
                "ok": False,
                "error": "launch_failed",
            }

        result = None
        result_file = workdir / _RESULT_FILENAME
        if result_file.exists():
            try:
                result = json.loads(result_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                result = result_file.read_text(encoding="utf-8")[:2000]

        charts: list[str] = []
        for chart in sorted((workdir / _CHARTS_DIR).glob("*.png")):
            charts.append(base64.b64encode(chart.read_bytes()).decode("ascii"))

        return {
            "stdout": _truncate(proc.stdout or "", max_bytes),
            "stderr": _truncate(proc.stderr or "", max_bytes),
            "result": result,
            "charts": charts,
            "ok": proc.returncode == 0,
            "error": None if proc.returncode == 0 else "runtime_error",
        }
