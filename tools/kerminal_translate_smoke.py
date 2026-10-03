"""Translate the five-round Jev fixture using the installed Kerminal CLI."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples" / "jev_smoke"


def validate_translation(evidence, chinese_text):
    """Check structure and exact numeric preservation; semantics need a separate review."""
    if evidence.get("language") != "en" or evidence.get("sample_kind") != "synthetic_connectivity_fixture_not_real_performance_evidence":
        raise ValueError("Translation must retain the English and synthetic-data labels.")
    if "kernel" in evidence:
        raise ValueError("The translator must not replace source code.")
    if not isinstance(evidence.get("evidence_status"), str) or not evidence["evidence_status"].strip():
        raise ValueError("Translation omitted the explicit evidence-status explanation.")
    expected = []
    for line in chinese_text.splitlines():
        columns = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(columns) == 6 and columns[0].isdigit():
            expected.append((int(columns[0]), float(columns[3]), float(columns[4])))
    if [row[0] for row in expected] != [1, 2, 3, 4, 5]:
        raise ValueError("This smoke test expects the five-round Chinese fixture.")
    history = evidence.get("iteration_history")
    if not isinstance(history, list) or len(history) != len(expected):
        raise ValueError("Translation omitted or added iteration records.")
    for item, (iteration, geomean, minimum) in zip(history, expected):
        metrics = item.get("measured_effect", {})
        if type(item.get("iteration")) is not int or item["iteration"] != iteration:
            raise ValueError("Translation changed iteration order or IDs.")
        for key, value in (("geomean_speedup", geomean), ("min_case_speedup", minimum)):
            if type(metrics.get(key)) not in (int, float) or metrics[key] != value:
                raise ValueError("Translation changed a measured number.")
        for field in ("direction", "implementation_change", "adoption_outcome"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError("Translation omitted an iteration description.")
    protocol = evidence.get("measurement_protocol", {})
    if protocol.get("provider") != "cann-bench" or protocol.get("strategy") != "kernel_details":
        raise ValueError("Translation changed the measurement provider or strategy.")
    for section, fields in ((protocol, ("scope", "metric_policy")),
                            (evidence.get("profiling_analysis", {}), ("finding", "limitation", "next_validation"))):
        for field in fields:
            if not isinstance(section.get(field), str) or not section[field].strip():
                raise ValueError("Translation omitted a measurement or profiling explanation.")
    if re.search(r"[\u3400-\u9fff]", json.dumps(evidence, ensure_ascii=False, allow_nan=False)):
        raise ValueError("Chinese text remains in the English evidence.")
    return {"structure": "passed", "numeric_values": "passed", "iteration_count": 5,
            "chinese_text_absent": True, "semantic_review": "required_before_jev"}


def translate(cli, run_dir, prompt, timeout):
    """Use CLI's advertised MCP tool; no additional MCP service installation."""
    with (run_dir / "kerminal.stderr.log").open("w", encoding="utf-8") as stderr:
        proc = subprocess.Popen([str(cli), "mcp-server"], cwd=run_dir,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                                text=True, encoding="utf-8")
        messages = queue.Queue()

        def read_messages():
            for line in proc.stdout:
                messages.put(line)
            messages.put(None)

        reader = threading.Thread(target=read_messages, daemon=True)
        reader.start()
        deadline = time.monotonic() + timeout

        def send(value):
            proc.stdin.write(json.dumps({"jsonrpc": "2.0", **value}) + "\n")
            proc.stdin.flush()

        def receive(request_id):
            while time.monotonic() < deadline:
                try:
                    line = messages.get(timeout=min(20, max(0.1, deadline - time.monotonic())))
                except queue.Empty:
                    print("Kerminal translation is running...", flush=True)
                    continue
                if line is None:
                    raise RuntimeError("Kerminal closed before returning a result.")
                message = json.loads(line)
                if "method" in message and "id" in message:
                    # Do not authorize interactive requests outside this bounded read-only task.
                    send({"id": message["id"], "error": {"code": -32601, "message": "Interactive request unsupported"}})
                if message.get("id") == request_id and "method" not in message:
                    if "error" in message:
                        raise RuntimeError("Kerminal returned a protocol error.")
                    return message["result"]
            raise TimeoutError("Kerminal translation exceeded the configured timeout.")

        try:
            send({"id": 1, "method": "initialize", "params": {
                "protocolVersion": "2024-11-05", "capabilities": {},
                "clientInfo": {"name": "jev_translation_smoke", "version": "0.1.0"}}})
            server = receive(1)["serverInfo"]
            send({"method": "notifications/initialized", "params": {}})
            send({"id": 2, "method": "tools/list", "params": {}})
            if not any(tool["name"] == "kerminal" for tool in receive(2)["tools"]):
                raise RuntimeError("The installed CLI does not expose the Kerminal tool.")
            send({"id": 3, "method": "tools/call", "params": {"name": "kerminal", "arguments": {
                "prompt": prompt, "cwd": str(run_dir), "sandbox": "read-only", "approval-policy": "never",
                "developer-instructions": "Translate only the two named local fixture files. Preserve evidence and numbers. Return English JSON. Do not read credentials or other files and do not modify anything."
            }}})
            result = receive(3)
            (run_dir / "kerminal.result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            if result.get("isError"):
                raise RuntimeError("Kerminal reported a failed translation; inspect its local result.")
            text = "\n".join(block["text"] for block in result.get("content", []) if block.get("type") == "text")
            (run_dir / "kerminal.final.txt").write_text(text, encoding="utf-8")
            return json.loads(text), server
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            reader.join(timeout=2)
            proc.stdin.close()
            proc.stdout.close()


def main():
    default_cli = shutil.which("kerminal") or str(Path(os.environ.get("LOCALAPPDATA", "")) / "kerminal/bin/kerminal.exe")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", default=default_cli)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/jev_setup/kerminal")
    args = parser.parse_args()
    run_dir = (args.output_dir / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8])).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for filename in ("evidence.zh.md", "kernel.py"):
        data = (FIXTURE / filename).read_bytes()
        (run_dir / filename).write_bytes(data)
        hashes[filename] = hashlib.sha256(data).hexdigest()
    prompt = (FIXTURE / "kerminal_prepare_prompt.txt").read_text(encoding="utf-8")
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    summary = {"agent": "kerminal", "cli": str(args.cli), "source_sha256": hashes,
               "status": "running", "output_dir": str(run_dir)}
    print(json.dumps(summary), flush=True)
    start = time.monotonic()
    try:
        evidence, server = translate(args.cli, run_dir, prompt, args.timeout)
        checks = validate_translation(evidence, (run_dir / "evidence.zh.md").read_text(encoding="utf-8"))
        for filename, digest in hashes.items():
            if hashlib.sha256((run_dir / filename).read_bytes()).hexdigest() != digest:
                raise ValueError("Input fixture was modified during translation.")
        (run_dir / "evidence.en.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summary.update(status="translated_pending_semantic_review", server=server, validation=checks)
        return 0
    except Exception as exc:
        summary.update(status="failed", error_type=type(exc).__name__)
        print(f"Kerminal translation failed: {type(exc).__name__}. See the run directory.", file=sys.stderr)
        return 1
    finally:
        summary["elapsed_seconds"] = round(time.monotonic() - start, 3)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    sys.exit(main())
