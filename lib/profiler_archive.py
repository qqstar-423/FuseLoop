"""Keep each evaluation's profiler evidence inside its own work directory."""

from pathlib import Path, PurePosixPath
import shutil
import tempfile


def archive_profiler_data(report_path, eval_dir, log_callback=None):
    """Copy only the returned report's sibling prof_data, never guess other roots."""
    emit = log_callback or (lambda message: None)
    evaluation = Path(eval_dir).resolve()
    evaluation.mkdir(parents=True, exist_ok=True)
    destination = evaluation / "prof_data"
    # Never replace a link/junction that leads outside this evaluation directory.
    if destination.is_symlink() or destination.resolve().parent != evaluation:
        raise ValueError(f"prof_data 归档目标超出本轮评测目录：{destination}")
    source = Path(report_path).resolve().parent / "prof_data" if report_path else None
    if source is None or not source.is_dir():
        if destination.exists():
            shutil.rmtree(destination)
        emit(f"[profiler] 本次报告没有 prof_data：{source or '无报告'}；"
             f"本轮不关联 kernel_csv，不复用旧文件；目标={destination}")
        return ""
    if source.resolve() == destination.resolve():
        emit(f"[profiler] 数据已在本轮 work 目录：{destination}")
        return str(destination)
    # Finish copying before publishing the new directory; do not expose a partial copy.
    with tempfile.TemporaryDirectory(prefix=".profiler_copy_", dir=evaluation) as temporary:
        staged = Path(temporary) / "prof_data"
        shutil.copytree(source, staged)
        if destination.exists():
            shutil.rmtree(destination)
        staged.replace(destination)
    emit(f"[profiler] 本轮数据已保存：source={source} -> work={destination}")
    if (destination / "_batched").is_dir():
        emit("[profiler] 已保留批量采集 _batched 数据；没有逐 case 对应证据时，"
             "不把共享 CSV 当作某个 case 的 kernel_csv")
    return str(destination)


def case_kernel_csv(source_csv_dir, case_id):
    """Find an unambiguous per-case CSV, matching cann-bench's case-id layout."""
    if not source_csv_dir:
        return ""
    root = Path(source_csv_dir).resolve()
    prefix, separator, number = str(case_id).rpartition("_")
    if not separator or not number.isdigit():
        return ""
    relative = PurePosixPath(prefix.replace("\\", "/"))
    if (relative.is_absolute() or ".." in relative.parts
            or any(":" in part for part in relative.parts)):
        return ""
    directory = root.joinpath(*relative.parts, number)
    # Older callers passed an operator directory, rather than the full prof_data root.
    if not directory.is_dir() and root.name != "prof_data":
        directory = root / number
    if not directory.is_dir() or not directory.resolve().is_relative_to(root):
        return ""
    matches = sorted(path.resolve() for path in directory.rglob("kernel_details.csv")
                     if path.is_file() and path.resolve().is_relative_to(root))
    return str(matches[0]) if len(matches) == 1 else ""
