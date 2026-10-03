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
    parser = argparse.ArgumentParser(description="向运行中的 workflow 提交人工方向、提问或等待请求")
    parser.add_argument("--work-dir", required=True, help="本次 workflow 的工作目录")
    parser.add_argument("--request-id", help="可选咨询编号；默认自动关联正在等待的咨询")
    parser.add_argument("--kind", choices=("direction", "question"), default="direction",
                        help="direction=方向建议（P0）；question=先答疑，不直接修改")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", "--message", dest="text", help="简短原话请用引号包住；长意见请用 --file，避免命令行参数过长")
    source.add_argument("--file", help="完整读取 UTF-8 意见文件；长意见推荐此方式，不受正文的命令行参数长度限制")
    source.add_argument("--status", action="store_true", help="显示当前咨询与待处理意见，不提交")
    args = parser.parse_args(argv)
    try:
        if args.status:
            runtime = HumanReview(args.work_dir, observe_messages=False)
            output = {"state": runtime.state(), "consultation": runtime.active_consultation(),
                      "pending": runtime.pending_messages()}
        else:
            text = Path(args.file).read_text(encoding="utf-8-sig") if args.file else args.text
            if text is None:
                text = input("请输入人工意见（等待请填“请等待”）：")
            output = submit_message(args.work_dir, text, args.request_id, kind=args.kind)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError) as exc:
        print(f"提交失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
