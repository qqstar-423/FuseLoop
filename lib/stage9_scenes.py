"""Compose a Stage9 role from common rules and only the current scene's tasks.

The caller persists the returned text alongside each request. Scene selection is
deterministic; old successful metrics cannot override a build/precision failure.
"""

from pathlib import Path


SCENES = frozenset({"compile", "precision", "evaluation_error", "optimization",
                    "stagnation", "all_passed"})
PERFORMANCE_SCENES = frozenset({"optimization", "stagnation", "all_passed"})
_PERFORMANCE_INPUTS = frozenset({"fusion_library", "perf_result", "profiler",
                               "perf_reports", "bottleneck", "fix_directive", "search_report",
                               "build_log", "precision_result", "precision_reports", "self_test"})
_SCENE_INPUTS = {
    "compile": frozenset({"build_log", "self_test"}),
    "precision": frozenset({"build_log", "precision_result", "precision_reports", "self_test"}),
    "evaluation_error": frozenset({"perf_result", "profiler", "perf_reports",
                                    "build_log", "precision_result", "self_test"}),
    "optimization": _PERFORMANCE_INPUTS,
    "stagnation": _PERFORMANCE_INPUTS,
    "all_passed": _PERFORMANCE_INPUTS - {"bottleneck", "fix_directive", "search_report"},
}


def classify_scene(fail_reason: str, semantic_decision=None, metrics=None, *, iteration=None) -> str:
    """Select a scene, honoring the original failure route before old metrics."""
    lines = (fail_reason or "").strip().splitlines()
    reason = lines[0] if lines else ""
    if reason.startswith(("build_fail", "compile_fail")):
        return "compile"
    if reason.startswith("precision_fail"):
        return "precision"
    if reason.startswith(("score_zero", "evaluation_error", "score_error", "perf_error")):
        return "evaluation_error"
    if reason.startswith("perf_pass"):
        return "all_passed"
    status = semantic_decision if isinstance(semantic_decision, dict) else {}
    measured = metrics if isinstance(metrics, dict) else {}
    if measured.get("score_error_code") or measured.get("avg_speedup") == 0:
        return "evaluation_error"
    current_status = iteration is None or status.get("latest_iteration") == iteration
    if status.get("eligible") and current_status:
        if status.get("review_fusion"):
            return "stagnation"
        if not reason.startswith("perf_optimize") and (status.get("window") or {}).get("status") == "passed":
            return "all_passed"
    if not reason.startswith("perf_optimize") and measured.get("all_cases_pass") is True:
        return "all_passed"
    return "optimization"


def _check_scene(scene):
    if scene not in SCENES:
        raise ValueError(f"Unknown Stage9 scene: {scene}")


def is_performance_scene(scene: str) -> bool:
    _check_scene(scene)
    return scene in PERFORMANCE_SCENES


def scene_input_keys(scene: str) -> frozenset:
    """Default input names; full versioned evidence remains available on demand.

Requirement, hardware, code/evidence binding, prior advice and relevant history
remain common inputs. This selection never deletes any original artifacts.
"""
    _check_scene(scene)
    return _SCENE_INPUTS[scene]


def build_scene_role(roles_dir, scene: str, *, phase: str = "decision",
                     perf_diff=None, has_question: bool = False,
                     has_human: bool = False) -> str:
    """Return common + scene + phase rules without all six scene task lists.

Consultation produces question.json only. Feedback and decision use the final
protocol. Performance knowledge is requested only for valid performance scenes
when the matching program-computed improvement/regression flag is true.
"""
    _check_scene(scene)
    if phase not in {"decision", "consultation", "feedback"}:
        raise ValueError(f"Unknown Stage9 phase: {phase}")
    if phase == "consultation" and scene != "stagnation":
        raise ValueError("Stage9 consultation requires the stagnation scene")
    root = Path(roles_dir)
    fragments = [root / "n4_stage9_tech_lead_guide.md", root / "stage9" / f"{scene}.md"]
    if scene in PERFORMANCE_SCENES:
        fragments.append(root / "stage9" / "performance_evidence.md")
    if phase == "consultation":
        fragments.append(root / "stage9" / "consultation.md")
    else:
        fragments.append(root / "stage9" / "decision.md")
        if has_question:
            fragments.append(root / "stage9" / "pitfall.md")
        diff = perf_diff if isinstance(perf_diff, dict) else {}
        if scene in PERFORMANCE_SCENES and (diff.get("has_improvement") or diff.get("has_regression")):
            fragments.append(root / "stage9" / "patterns.md")
        if has_human or phase == "feedback":
            fragments.append(root / "stage9" / "human.md")
    return "\n\n".join(path.read_text(encoding="utf-8-sig").strip() for path in fragments) + "\n"
