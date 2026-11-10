from app.services.sandbox import run_code

SIMPLE = """
import json
values = [10, 20, 30, 40]
result = {"count": len(values), "mean": sum(values) / len(values)}
with open("analysis_result.json", "w", encoding="utf-8") as fh:
    json.dump(result, fh)
"""

CHART = """
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
values = [1, 4, 9, 16]
plt.plot(range(len(values)), values)
plt.savefig("charts/chart_0.png")
with open("analysis_result.json", "w", encoding="utf-8") as fh:
    json.dump({"ok": True}, fh)
"""


def test_sandbox_runs_code_and_captures_result():
    out = run_code(SIMPLE)
    assert out["ok"] is True
    assert out["result"]["mean"] == 25.0
    assert out["error"] is None


def test_sandbox_captures_charts_as_base64():
    out = run_code(CHART)
    assert out["ok"] is True
    assert len(out["charts"]) == 1


def test_sandbox_timeout_is_enforced():
    out = run_code("while True:\n    pass", timeout=2)
    assert out["ok"] is False
    assert out["error"] == "timeout"


def test_sandbox_runtime_error_is_captured():
    out = run_code("raise ValueError('boom')")
    assert out["ok"] is False
    assert "boom" in out["stderr"]


def test_sandbox_seed_files_available():
    code = (
        "import json\n"
        "data = json.load(open('input.json', encoding='utf-8'))\n"
        "r = {'seen': data['hello']}\n"
        "with open('analysis_result.json', 'w', encoding='utf-8') as fh:\n"
        "    json.dump(r, fh)"
    )
    out = run_code(code, files={"input.json": '{"hello": "world"}'})
    assert out["ok"] is True
    assert out["result"]["seen"] == "world"
