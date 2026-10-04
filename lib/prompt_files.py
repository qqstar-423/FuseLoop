"""Describe prompt artifacts without resolving linked task directories."""

import os
import re


def file_hint(work_dir, path, purpose, read_hint, *, base_dir=None, base_label="working directory") -> str:
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
        relative = "cannot be expressed across drives; use the actual path"
    template_parts = []
    for part in relative.split("/"):
        if re.fullmatch(r"iter\d+-[0-9a-fA-F]+", part):
            part = "<iter>-<fingerprint>"
        elif re.fullmatch(r"iter\d+", part):
            part = "<iter>"
        template_parts.append(part)
    template = "/".join(template_parts)
    dynamic = f"; iteration template: `{template}`" if template != relative else ""
    purpose = " ".join(str(purpose).splitlines())
    read_hint = " ".join(str(read_hint).splitlines())
    return (f"- File: `{actual}`; relative to {base_label}: `{relative}`{dynamic}; "
            f"Purpose: {purpose}; how to read: {read_hint}\n")
