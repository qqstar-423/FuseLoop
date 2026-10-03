"""Small protocol helpers shared by Stage9, development and the human inbox.

The workflow, not a model, controls waiting, delivery and evaluation counters.
"""
import json
from pathlib import Path

from .handoff import atomic_write_json, atomic_write_text
from .prompt_files import file_hint


def evidence_sources(work_dir, iter_dirs):
    work = Path(work_dir)
    items = [
        (work / "task", "算子需求、case 和 golden", "核对接口、精度和覆盖范围"),
        (work / "ANALYSIS.md", "Stage1 需求分析", "核对实现约束"),
        (work / "device_info.json", "当前硬件", "核对核数、存储容量和编程模型"),
        (work / "impl", "本次实际实现", "对照入口、tiling 和数据流"),
        (work / "fusion", "Stage1.5 初始方案与概率", "概率是先验，结合当前实测判断"),
        (work / "selection", "最佳实现、代码绑定、窗口及 case 趋势", "核对快照指标与实现是否匹配"),
        (work / "knowledge/history.json", "完整历轮账本", "比较已尝试方向和结果"),
        (work / "knowledge/proven_patterns.md", "成功经验", "核对适用条件"),
        (work / "knowledge/regression_patterns.md", "退步经验", "避免重复已证实的问题"),
        (work / "knowledge/tech_lead_pitfalls.md", "争议裁定", "检查历史误判和正确做法"),
        (work / "develop", "历轮融合方案选择依据与自测", "区分实现选择、自测和正式评测"),
    ]
    for key, purpose in (("build", "本轮编译日志"), ("eval", "精度、性能及原始 profiling"),
                         ("profile", "Stage7 瓶颈结论"), ("search", "Stage8 搜索与建议")):
        items.append((Path(iter_dirs[key]), purpose, "先读结论，再按 case 回查原始证据"))
    return [{"path": str(path), "purpose": purpose, "read_hint": hint,
             "source_description": file_hint(work_dir, path, purpose, hint).strip()}
            for path, purpose, hint in items]


def validate_question(output, request_id):
    if not isinstance(output, dict) or output.get("request_id") != request_id:
        raise ValueError("咨询输出必须匹配本次 request_id")
    for key in ("question", "difficulty", "current_scheme", "attempts", "evidence", "recommendation_reason"):
        if not isinstance(output.get(key), str) or not output[key].strip():
            raise ValueError(f"咨询缺少 {key}")
    options = output.get("options")
    if not isinstance(options, list) or not 2 <= len(options) <= 3:
        raise ValueError("咨询必须提供 2～3 个实际选项")
    ids = set()
    for option in options:
        if not isinstance(option, dict):
            raise ValueError("咨询选项必须为对象")
        for key in ("id", "title", "benefit", "cost", "risk"):
            if not isinstance(option.get(key), str) or not option[key].strip():
                raise ValueError(f"咨询选项缺少 {key}")
        if option["id"] in ids:
            raise ValueError("咨询选项编号不能重复")
        ids.add(option["id"])
    if output.get("recommended_option") not in ids:
        raise ValueError("推荐选项必须出现在选项列表中")


def render_question(output, request, selection_status, work_dir):
    validate_question(output, request["request_id"])
    lines = ["# Stage9 请求人工判断", "", output["question"], "",
             f"当前方案：{output['current_scheme']}", f"难点：{output['difficulty']}",
             f"已尝试：{output['attempts']}", "", f"依据：{output['evidence']}"]
    best = selection_status.get("best") or {}
    lines += ["", f"程序记录的停滞触发轮次：{request.get('trigger_iterations', [])}",
              f"最佳 avg_speedup：{best.get('avg_speedup', '暂无')}；HAP：{(best.get('hap') or {}).get('performance_score', '暂无')}",
              f"窗口：{json.dumps(selection_status.get('window', {}), ensure_ascii=False)}", ""]
    for option in output["options"]:
        lines += [f"- {option['id']}：{option['title']}。收益：{option['benefit']}；代价：{option['cost']}；风险：{option['risk']}。"]
    lines += ["", f"推荐 {output['recommended_option']}：{output['recommendation_reason']}", "",
              "请通过 tools/human_review.py 向此工作目录提交选项或意见。也可以提出其他方向；纯提问使用 --kind question。",
              "通知后等待 2 分钟；回复“请等待”仅在原截止时间上加 10 分钟。超时按推荐方向继续，不能视为人工同意。", ""]
    manifest = request.get("evidence_manifest_path")
    if manifest:
        lines.append(file_hint(work_dir, manifest, "咨询时全部证据的版本清单", "按用途和读法找快照；缺失材料有明确记录"))
    return "\n".join(lines) + "\n"


def human_prompt(work_dir, messages, bundle_path=None, *, ending_reason=None):
    if not messages and not bundle_path:
        return ""
    text = "\n=== 人类意见（逐条处理，实质方向为 P0） ===\n"
    if bundle_path:
        text += file_hint(work_dir, bundle_path, "完整人工反馈包与咨询时证据", "先读原问题、全部选项和全部原话，再核对快照版本；超时不代表人工同意")
    text += json.dumps(messages, ensure_ascii=False, indent=2) + "\n"
    text += (
        "human_responses 必须逐条输出 {message_id, kind, answer}；kind 沿用提交类型 direction/question/wait。"
        "实质方向不能改成提问；与正确性或硬件冲突时 kind=conflict，解释原因并增加 alternative。"
        "每条 direction/conflict 必须有 suggest_next 的 P0，source=human，human_message_id 对应该消息编号；"
        "action 明确写 P0（人工建议）的目标或可行替代。纯提问先回答，wait 不作为方向。"
        "不得将超时推荐伪装为人工意见。\n")
    if ending_reason:
        text += f"程序已经决定退出：{ending_reason}。意见仍需处理并保留人工 P0，但不会再执行 Stage3；解释未执行原因，不能突破退出条件。\n"
    return text


def development_human_prompt(work_dir, messages, output_path):
    if not messages:
        return ""
    return (
        "\n=== 本次人工 P0 交付 ===\n" + json.dumps(messages, ensure_ascii=False, indent=2)
        + "\n遵守 history 中 Stage9 对应人工 P0；在《融合方案选择决策依据.md》中按意见编号说明实际落实、修改文件或无法落实原因。"
        + file_hint(work_dir, output_path, "本次人工意见执行回执（待生成）",
                    "写 JSON 列表，每项 message_id、status（implemented 或 not_implemented）、details（具体改动或原因）。自测通过不能替代正式性能验证")
    )


def validate_execution_receipt(path, messages):
    rows = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    expected = {message["id"] for message in messages}
    if not isinstance(rows, list):
        raise ValueError("人工执行回执必须为 JSON 列表")
    seen = set()
    for row in rows:
        if (not isinstance(row, dict) or row.get("message_id") not in expected
                or row["message_id"] in seen or row.get("status") not in {"implemented", "not_implemented"}
                or not isinstance(row.get("details"), str) or not row["details"].strip()):
            raise ValueError("人工执行回执缺少有效编号、状态或具体落实依据")
        seen.add(row["message_id"])
    if seen != expected:
        raise ValueError("人工执行回执必须覆盖本次全部意见")
    return rows
