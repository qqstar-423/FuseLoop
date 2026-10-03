"""Describe prompt artifacts without resolving linked task directories."""

import os
import re


def file_hint(work_dir, path, purpose, read_hint, *, base_dir=None, base_label="工作目录") -> str:
    """Return one line with actual location, relative/template path and reading advice.

    Paths retain their lexical locations: ``work/task`` may be a link to an
    external input directory and must remain recognizable as ``task/...``.
    Like normal file operations, relative arguments are interpreted from cwd.
    """
    actual = os.path.abspath(os.fspath(path))
    base = os.path.abspath(os.fspath(work_dir if base_dir is None else base_dir))
    try:
        relative = os.path.relpath(actual, base).replace("\\", "/")
    except ValueError:
        # Windows cannot express a relative path across drive letters.
        relative = "跨盘无法表示，请使用实际路径"
    template_parts = []
    for part in relative.split("/"):
        if re.fullmatch(r"iter\d+-[0-9a-fA-F]+", part):
            part = "<iter>-<指纹>"
        elif re.fullmatch(r"iter\d+", part):
            part = "<iter>"
        template_parts.append(part)
    template = "/".join(template_parts)
    dynamic = f"；迭代模板：`{template}`" if template != relative else ""
    purpose = " ".join(str(purpose).splitlines())
    read_hint = " ".join(str(read_hint).splitlines())
    return (f"- 文件：`{actual}`；相对{base_label}：`{relative}`{dynamic}；"
            f"用途：{purpose}；怎么看：{read_hint}\n")
