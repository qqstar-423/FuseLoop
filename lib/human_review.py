"""Durable local human inbox and consultation clock.

Only the workflow writes state/request files. Independent terminals publish one
immutable UUID-named message each, so submitting never races a state rewrite.
Waiting and stagnation counts are controlled by Python, never by model output.
Inbox files acknowledge publication; receipts acknowledge the workflow's first
logged read and do not imply that Stage9 processed or Stage3 executed an opinion.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import shutil
import time
from typing import Callable
import uuid

from lib.handoff import atomic_write_json, atomic_write_text


log = logging.getLogger("triton-ascend-workflow")
WAIT_SECONDS = 120
EXTENSION_SECONDS = 600
TRIGGER_COUNT = 3


def _read(path: Path, default=None):
    if not path.exists():
        return deepcopy(default)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write(path: Path, value: dict):
    atomic_write_json(str(path), value)


def _stamp(now: float) -> str:
    return datetime.fromtimestamp(now, timezone.utc).isoformat()


def _summary(text: str, limit: int = 120) -> str:
    """Keep a log entry on one line; the unmodified text stays in the inbox."""
    value = " ".join(re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text).split())
    return value if len(value) <= limit else value[:limit] + "…"


def is_wait_message(text: str) -> bool:
    """Only a stand-alone wait instruction is a wait; mixed advice is retained."""
    compact = re.sub(r"[\s，。！？,.!?；;：:]+", "", text).lower()
    return compact in {"请等待", "等待一会儿", "等一会儿", "稍等", "请稍等", "请等一下",
                       "等一下", "等等", "稍等一会", "稍等一会儿", "等待", "wait", "pleasewait"}


def _initial_state() -> dict:
    return {"schema_version": 1, "active_enabled": True, "consultation_enabled": True,
            "workflow_closed": False, "messages": {}, "requests": {}, "active_request_id": None,
            "stagnation": {"count": 0, "trigger_iterations": [], "seen_iterations": []}}


def submit_message(work_dir: str, text: str, request_id: str | None = None,
                   *, kind: str = "direction", now: float | None = None) -> dict:
    """Publish without modifying workflow state. CLI inputs never become commands."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Human feedback must not be empty")
    if kind not in {"direction", "question"}:
        raise ValueError("kind must be direction or question")
    work = Path(work_dir).resolve()
    if not work.is_dir():
        raise ValueError(f"Working directory does not exist: {work}")
    root = work / "human_review"
    state = _read(root / "state.json", _initial_state())
    if state.get("workflow_closed"):
        raise ValueError("This workflow has finished; feedback will not restart the task")
    active_id = state.get("active_request_id")
    active = None
    if active_id and active_id in state.get("requests", {}):
        active = _read(root / state["requests"][active_id])
    if request_id is None and active and active.get("status") in {"draft", "waiting"}:
        request_id = active_id
    if request_id is not None and request_id not in state.get("requests", {}):
        raise ValueError("Consultation request ID not found")
    if not state.get("active_enabled", True) and not (
            request_id == active_id and active and active.get("status") in {"draft", "waiting"}):
        raise ValueError("The proactive human entry is closed")
    timestamp = float(time.time() if now is None else now)
    message_id = uuid.uuid4().hex
    message = {"id": message_id, "text": text, "kind": "wait" if is_wait_message(text) else kind,
               "submitted_at": timestamp, "submitted_at_utc": _stamp(timestamp),
               "request_id": request_id}
    path = root / "inbox" / f"{message_id}.json"
    _write(path, message)
    if _read(root / "state.json", _initial_state()).get("workflow_closed"):
        # The final reader may already have sealed the run while this terminal
        # published. Do not report this late submission as accepted for execution.
        raise ValueError(f"The workflow had already finished when submitted; the verbatim message is archived but will not be executed: {path}")
    return {**message, "path": str(path)}


class HumanReview:
    def __init__(self, work_dir: str, *, clock: Callable[[], float] = time.time,
                 sleeper: Callable[[float], None] = time.sleep, observe_messages: bool = True):
        self.work_dir = Path(work_dir).resolve()
        self.root = self.work_dir / "human_review"
        self.clock, self.sleeper = clock, sleeper
        # A second terminal's --status must not acknowledge messages on behalf
        # of the workflow process (which owns workflow.log and these receipts).
        self.observe_messages = observe_messages

    def state(self) -> dict:
        return _read(self.root / "state.json", _initial_state())

    def _save(self, state: dict):
        _write(self.root / "state.json", state)

    def configure(self, *, active_enabled: bool = True, consultation_enabled: bool = True):
        state = self.state()
        state.update(active_enabled=bool(active_enabled), consultation_enabled=bool(consultation_enabled))
        self._save(state)

    def close_workflow(self, reason: str):
        state = self.state()
        state.update(workflow_closed=True, closed_reason=reason, closed_at=self.clock())
        self._save(state)

    def _messages(self) -> list[dict]:
        result = []
        for path in (self.root / "inbox").glob("*.json"):
            item = _read(path)
            if not item or item.get("id") != path.stem or not isinstance(item.get("text"), str):
                raise ValueError(f"Invalid human inbox message: {path}")
            result.append({**item, "path": str(path)})
        result.sort(key=lambda item: (item["submitted_at"], item["id"]))
        if self.observe_messages:
            for item in result:
                self._observe_message(item)
        return result

    def _observe_message(self, item: dict):
        workflow_log = self.work_dir / "log" / "workflow.log"
        if not log.isEnabledFor(logging.INFO) or not any(
                isinstance(handler, logging.FileHandler) and handler.level <= logging.INFO
                and Path(handler.baseFilename).resolve() == workflow_log
                for handler in log.handlers):
            # A library/status reader without this workflow's file handler must
            # not consume the durable "received" event before the workflow sees it.
            return
        receipt_path = self.root / "receipts" / f"{item['id']}.json"
        if receipt_path.exists():
            return
        observed = self.clock()
        _write(receipt_path, {"message_id": item["id"], "request_id": item.get("request_id"),
                              "kind": item["kind"], "original_path": item["path"],
                              "submitted_at": item["submitted_at"], "observed_at": observed,
                              "observed_at_utc": _stamp(observed)})
        log.info("Human message read message_id=%s request_id=%s kind=%s summary=%s original=%s receipt=%s",
                 item["id"], item.get("request_id") or "proactive", item["kind"],
                 _summary(item["text"]), item["path"], receipt_path)

    def all_messages(self) -> list[dict]:
        """Retain processed-but-not-executed advice when a decision is reconsidered."""
        state = self.state()
        return [{**item, **state["messages"].get(item["id"], {"status": "pending"})}
                for item in self._messages()]

    def pending_messages(self) -> list[dict]:
        state = self.state()
        result = []
        for item in self._messages():
            status = state["messages"].get(item["id"], {})
            if status.get("status") in {"processed", "executed", "not_executed"}:
                continue
            request_id = item.get("request_id")
            request = self.get_consultation(request_id) if request_id else None
            if request and request.get("status") in {"draft", "waiting"}:
                continue
            result.append({**item, "status": "pending", "late_reply": bool(
                request and item["id"] not in request.get("message_ids", []))})
        return result

    def mark_processed(self, message_ids: list[str], decision_path: str):
        self._mark(message_ids, "processed", decision_path=str(decision_path), processed_at=self.clock())

    def mark_executed(self, message_ids: list[str], iteration: int, evidence_path: str):
        self._mark(message_ids, "executed", executed_iteration=iteration,
                   execution_evidence_path=str(evidence_path), executed_at=self.clock())

    def mark_unexecuted(self, message_ids: list[str], reason: str):
        self._mark(message_ids, "not_executed", reason=reason, closed_at=self.clock())

    def _mark(self, ids: list[str], status: str, **metadata):
        state = self.state()
        known = {message["id"]: message for message in self._messages()}
        if any(message_id not in known for message_id in ids):
            raise ValueError("Unknown human message ID")
        changes = []
        for message_id in ids:
            previous = state["messages"].get(message_id, {})
            if previous.get("status") == "executed" and status == "processed":
                continue
            state["messages"][message_id] = {**previous, "status": status, **metadata}
            if previous.get("status") != status or any(
                    previous.get(key) != value for key, value in metadata.items() if not key.endswith("_at")):
                changes.append(message_id)
        self._save(state)
        for message_id in changes:
            message = known[message_id]
            handling = {"direction": "human P0 suggestion", "question": "question answer", "wait": "wait only"}[message["kind"]]
            log.info("Human message status message_id=%s request_id=%s kind=%s status=%s handling=%s "
                     "decision=%s execution evidence=%s reason=%s status record=%s",
                     message_id, message.get("request_id") or "proactive", message["kind"], status,
                     handling, state["messages"][message_id].get("decision_path", "-"),
                     metadata.get("execution_evidence_path", "-"), _summary(str(metadata.get("reason", "-"))),
                     self.root / "state.json")

    def record_evaluation(self, iteration: int, *, valid: bool, all_passed: bool,
                          review_fusion: bool, window_gain: float | None = None) -> dict:
        state = self.state()
        counter = state["stagnation"]
        if not valid or iteration in counter["seen_iterations"]:
            return {**deepcopy(counter), "consultation_due": counter["count"] >= TRIGGER_COUNT}
        counter["seen_iterations"].append(iteration)
        if all_passed or (window_gain is not None and window_gain >= 0.05):
            counter.update(count=0, trigger_iterations=[])
        elif review_fusion:
            counter["trigger_iterations"].append(iteration)
            counter["count"] += 1
        self._save(state)
        return {**deepcopy(counter), "consultation_due": counter["count"] >= TRIGGER_COUNT}

    def get_consultation(self, request_id: str) -> dict | None:
        relative = self.state()["requests"].get(request_id)
        return _read(self.root / relative) if relative else None

    def active_consultation(self) -> dict | None:
        request_id = self.state().get("active_request_id")
        return self.get_consultation(request_id) if request_id else None

    def _request_path(self, request: dict) -> Path:
        return self.root / f"iter{request['iteration']}" / request["request_id"] / "request.json"

    def _save_request(self, request: dict):
        _write(self._request_path(request), request)

    def create_consultation(self, iteration: int, scene: str, fail_reason: str,
                            evidence: list[dict], context: dict) -> dict:
        active = self.active_consultation()
        if active:
            return active
        if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
            raise ValueError("Invalid consultation iteration")
        state = self.state()
        request_id = uuid.uuid4().hex
        directory = self.root / f"iter{iteration}" / request_id
        directory.mkdir(parents=True)
        manifest = self._snapshot_evidence(directory, evidence)
        _write(directory / "evidence_manifest.json", {"files": manifest})
        request = {"schema_version": 1, "request_id": request_id, "iteration": iteration,
                   "scene": scene, "fail_reason": fail_reason, "status": "draft",
                   "created_at": self.clock(), "context": deepcopy(context),
                   "trigger_iterations": list(state["stagnation"]["trigger_iterations"]),
                   "directory": str(directory), "question_path": str(directory / "Question Document.md"),
                   "evidence_manifest_path": str(directory / "evidence_manifest.json"),
                   "message_ids": [], "extension_used": False, "deadline": None}
        self._save_request(request)
        state["requests"][request_id] = self._request_path(request).relative_to(self.root).as_posix()
        state["active_request_id"] = request_id
        self._save(state)
        log.info("Human consultation created request_id=%s iter=%s scene=%s trigger rounds=%s question=%s evidence manifest=%s",
                 request_id, iteration, scene, request["trigger_iterations"],
                 request["question_path"], request["evidence_manifest_path"])
        return request

    def _snapshot_evidence(self, directory: Path, evidence: list[dict]) -> list[dict]:
        manifest = []
        for index, specification in enumerate(evidence):
            item = deepcopy(specification)
            raw_path = item.get("path")
            source = Path(raw_path) if raw_path else None
            if source is not None and not source.is_absolute():
                source = self.work_dir / source
            if source is None or not source.exists():
                item.update(status="missing", snapshot_path=None)
                manifest.append(item)
                continue
            source = source.resolve()
            if directory.is_relative_to(source):
                raise ValueError("Evidence cannot contain the consultation output directory")
            target = directory / "evidence" / f"{index:03d}" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", ".git"))
                hashes = {p.relative_to(target).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(target.rglob("*")) if p.is_file()}
            else:
                shutil.copy2(source, target)
                hashes = {source.name: hashlib.sha256(target.read_bytes()).hexdigest()}
            item.update(path=str(source), status="available", snapshot_path=str(target), sha256=hashes)
            manifest.append(item)
        return manifest

    def create_proactive_bundle(self, iteration: int, scene: str, fail_reason: str,
                                 evidence: list[dict], context: dict, messages: list[dict]) -> str:
        """Archive the exact material reviewed with proactive advice, without waiting."""
        request_id = uuid.uuid4().hex
        directory = self.root / f"iter{iteration}" / request_id
        directory.mkdir(parents=True)
        manifest = self._snapshot_evidence(directory, evidence)
        manifest_path = directory / "evidence_manifest.json"
        _write(manifest_path, {"files": manifest})
        request = {"schema_version": 1, "request_id": request_id, "iteration": iteration,
                   "scene": scene, "fail_reason": fail_reason, "status": "proactive",
                   "created_at": self.clock(), "directory": str(directory),
                   "evidence_manifest_path": str(manifest_path),
                   "message_ids": [message["id"] for message in messages]}
        bundle = {"request": request, "messages": deepcopy(messages),
                  "human_response_received": any(message.get("kind") != "wait" for message in messages),
                  "question_text": "", "context": deepcopy(context), "evidence": manifest,
                  "instructions": "Substantive human feedback becomes P0; pure questions are answered first, and a wait expression is not a scheme selection."}
        _write(directory / "request.json", request)
        path = directory / "feedback.json"
        _write(path, bundle)
        log.info("Proactive human feedback packaged request_id=%s iter=%s scene=%s message_ids=%s bundle=%s evidence manifest=%s",
                 request_id, iteration, scene, request["message_ids"], path, manifest_path)
        return str(path)

    def start_wait(self, request_id: str, question_path: str | None = None) -> dict:
        """Call after showing the question notification. Re-entry retains deadline."""
        request = self.get_consultation(request_id)
        if not request:
            raise ValueError("Unknown consultation")
        if request["status"] != "draft":
            return request
        source = Path(question_path or request["question_path"])
        text = source.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError("The consultation question document must not be empty")
        target = Path(request["question_path"])
        if source.resolve() != target.resolve():
            atomic_write_text(str(target), text)
        now = self.clock()
        request.update(status="waiting", question_text=text, notified_at=now,
                       original_deadline=now + WAIT_SECONDS, deadline=now + WAIT_SECONDS)
        self._save_request(request)
        log.info("Human question ready request_id=%s iter=%s scene=%s question=%s deadline=%s wait seconds=%s",
                 request_id, request["iteration"], request["scene"], target,
                 _stamp(request["deadline"]), WAIT_SECONDS)
        return request

    def poll_consultation(self, request_id: str) -> dict:
        request = self.get_consultation(request_id)
        if not request:
            raise ValueError("Unknown consultation")
        if request["status"] != "waiting":
            return request
        now = self.clock()
        messages = self._messages()
        already = set(request["message_ids"])
        state = self.state()
        substantive = []
        for message in messages:
            if message["id"] in already or message.get("request_id") != request_id:
                continue
            if state["messages"].get(message["id"], {}).get("status") in {"processed", "executed", "not_executed"}:
                continue
            submitted = message["submitted_at"]
            if submitted > now or submitted > request["deadline"]:
                continue
            request["message_ids"].append(message["id"])
            if message["kind"] == "wait":
                if not request["extension_used"]:
                    request["extension_used"] = True
                    request["deadline"] = request["original_deadline"] + EXTENSION_SECONDS
                    log.info("Human consultation extended request_id=%s message_id=%s extension seconds=%s deadline=%s request=%s",
                             request_id, message["id"], EXTENSION_SECONDS,
                             _stamp(request["deadline"]), self._request_path(request))
            else:
                substantive.append(message["id"])
        if substantive:
            request.update(status="replied", response_message_ids=substantive, closed_at=now)
        elif now >= request["deadline"]:
            request.update(status="timed_out", response_message_ids=[], closed_at=now)
        self._save_request(request)
        if request["status"] != "waiting":
            log.info("Human consultation wait finished request_id=%s status=%s reply_ids=%s deadline=%s request=%s",
                     request_id, request["status"], request["response_message_ids"],
                     _stamp(request["deadline"]), self._request_path(request))
        return request

    def wait_consultation(self, request_id: str, *, poll_interval: float = 1.0,
                          on_update: Callable[[dict], None] | None = None) -> dict:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        previous = None
        while True:
            request = self.poll_consultation(request_id)
            if on_update and request != previous:
                on_update(deepcopy(request))
            if request["status"] != "waiting":
                return request
            previous = deepcopy(request)
            self.sleeper(min(poll_interval, max(0.0, request["deadline"] - self.clock())))

    def build_feedback_bundle(self, request_id: str) -> str:
        request = self.get_consultation(request_id)
        if not request or request["status"] not in {"replied", "timed_out", "completed"}:
            raise ValueError("Consultation has not finished waiting")
        ids = set(request["message_ids"])
        messages = [message for message in self._messages() if message["id"] in ids]
        manifest = _read(Path(request["evidence_manifest_path"]))
        bundle = {"request": request, "messages": messages,
                  "human_response_received": bool(request.get("response_message_ids")),
                  "question_text": request["question_text"], "context": request["context"],
                  "evidence": manifest["files"],
                  "instructions": "Substantive human feedback becomes P0; pure questions are answered first, and waiting is not feedback. On timeout continue per the recommendation; it must not be treated as human consent."}
        path = Path(request["directory"]) / "feedback.json"
        first_bundle = not path.exists()
        _write(path, bundle)
        if first_bundle:
            log.info("Human consultation feedback packaged request_id=%s status=%s human_response_received=%s message_ids=%s bundle=%s",
                     request_id, request["status"], bundle["human_response_received"], request["message_ids"], path)
        return str(path)

    def complete_consultation(self, request_id: str, decision_path: str) -> dict:
        request = self.get_consultation(request_id)
        if not request or request["status"] not in {"replied", "timed_out", "completed"}:
            raise ValueError("Consultation cannot be completed before feedback/timeout")
        if request["status"] != "completed":
            request.update(status="completed", decision_path=str(decision_path), completed_at=self.clock())
            self._save_request(request)
            log.info("Human consultation produced the final decision request_id=%s message_ids=%s decision=%s request=%s",
                     request_id, request["message_ids"], decision_path, self._request_path(request))
        self.mark_processed(request["message_ids"], decision_path)
        state = self.state()
        if state.get("active_request_id") == request_id:
            state["active_request_id"] = None
            state["stagnation"].update(count=0, trigger_iterations=[])
        self._save(state)
        return request
