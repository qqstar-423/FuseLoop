"""Send a local human opinion from a second terminal; never invoke the model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.human_review import HumanReview, submit_message


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Submit a human direction, question or wait request to a running workflow")
    parser.add_argument("--work-dir", required=True, help="Working directory of this workflow")
    parser.add_argument("--request-id", help="Optional consultation ID; by default associates with the consultation currently waiting")
    parser.add_argument("--kind", choices=("direction", "question"), default="direction",
                        help="direction=direction suggestion (P0); question=answered first, no direct modification")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", "--message", dest="text", help="Quote short verbatim text; use --file for long feedback to avoid overlong command-line arguments")
    source.add_argument("--file", help="Read a UTF-8 feedback file in full; recommended for long feedback, unaffected by command-line argument length limits")
    source.add_argument("--status", action="store_true", help="Show the current consultation and pending feedback without submitting")
    args = parser.parse_args(argv)
    try:
        if args.status:
            runtime = HumanReview(args.work_dir, observe_messages=False)
            output = {"state": runtime.state(), "consultation": runtime.active_consultation(),
                      "pending": runtime.pending_messages()}
        else:
            text = Path(args.file).read_text(encoding="utf-8-sig") if args.file else args.text
            if text is None:
                text = input("Enter human feedback (for waiting, type \"please wait\"): ")
            output = submit_message(args.work_dir, text, args.request_id, kind=args.kind)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError) as exc:
        print(f"Submission failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
