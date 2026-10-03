"""Read a local text file and make one explicit Jev API request. No NPU required."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.jev_client import build_request, check_budget, evaluate, load_settings


def main():
    fixture = ROOT / "examples" / "jev_smoke"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--source", type=Path, default=fixture / "kernel.py")
    parser.add_argument("--evidence", type=Path, default=fixture / "evidence.en.json")
    parser.add_argument("--questions", type=Path, default=fixture / "questions.json")
    parser.add_argument("--path-only", action="store_true", help="Negative control: omit source text from state.")
    parser.add_argument("--dry-run", action="store_true", help="Save input and byte budget without contacting Jev.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "jev_smoke",
                        help="Parent directory; each invocation creates a unique run subdirectory.")
    args = parser.parse_args()
    summary = None
    run_dir = None
    try:
        settings = load_settings(args.config)
        request = build_request(args.source, args.evidence, args.questions, model=settings.model)
        if args.path_only:
            request["state"]["kernel"] = {"path": str(args.source.resolve())}
        budget = check_budget(request, settings)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
        run_dir = args.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "request.json").write_text(
            json.dumps(request, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summary = {"live": not args.dry_run, "path_only_control": args.path_only,
                   "timestamp_utc": datetime.now(timezone.utc).isoformat(), "budget": budget,
                   "status": "prepared", "output_dir": str(run_dir.resolve())}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if not args.dry_run:
            start = time.monotonic()
            response = evaluate(request, settings)
            summary.update(elapsed_seconds=round(time.monotonic() - start, 3),
                           model=response.get("model"), usage=response.get("usage"))
            (run_dir / "response.json").write_text(
                json.dumps(response, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(json.dumps(response, indent=2, ensure_ascii=False))
        summary["status"] = "dry_run" if args.dry_run else "success"
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        # SDK/server error bodies can contain submitted text. Print no credentials or payload.
        status = getattr(exc, "status_code", None)
        if summary is not None and run_dir is not None:
            summary.update(status="failed", error_type=type(exc).__name__, http_status=status)
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Jev smoke test failed: {type(exc).__name__}; HTTP status={status}.", file=sys.stderr)
        if isinstance(exc, (ValueError, FileNotFoundError, UnicodeError)):
            print("Check input files, English evidence, model, credentials and local byte budgets.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
