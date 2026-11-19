"""
Assignment 11 — Audit Log (forensics, never blocks).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store input + start timestamp keyed by request_id; return the id."""
        request_id = request_id or f"req-{len(self.logs) + len(self._open) + 1:04d}"
        self._open[request_id] = {
            "request_id": request_id,
            "user_id": user_id,
            "input": text,
            "started_at": utc_now_iso(),
            # perf_counter là monotonic — dùng đo latency, không dùng để hiển thị.
            "_started": time.perf_counter(),
        }
        return request_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store output, layer decision, latency; append to self.logs."""
        pending = self._open.pop(request_id, None) if request_id else None
        if pending is None:
            # Không có record_input tương ứng — vẫn ghi để không mất dấu vết.
            pending = {
                "request_id": request_id or f"req-{len(self.logs) + 1:04d}",
                "user_id": user_id,
                "input": None,
                "started_at": None,
                "_started": None,
            }

        latency_ms = (
            round((time.perf_counter() - pending["_started"]) * 1000, 2)
            if pending["_started"] is not None
            else None
        )
        entry = {
            "request_id": pending["request_id"],
            "user_id": user_id or pending["user_id"],
            "input": pending["input"],
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "started_at": pending["started_at"],
            "completed_at": utc_now_iso(),
            "latency_ms": latency_ms,
        }
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
