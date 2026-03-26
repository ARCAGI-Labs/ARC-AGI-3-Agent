"""JSONL session recorder for Solana agents — mirrors agents/recorder.py."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

RECORDING_SUFFIX = ".solana.recording.jsonl"


def get_recordings_dir() -> str:
    return os.environ.get("SOLANA_RECORDINGS_DIR", os.environ.get("RECORDINGS_DIR", ""))


class SolanaRecorder:
    """Records agent actions and state to JSONL files."""

    def __init__(
        self, prefix: str, filename: Optional[str] = None, guid: Optional[str] = None
    ) -> None:
        self.guid = guid or str(uuid.uuid4())
        self.prefix = prefix
        recordings_dir = get_recordings_dir()
        self.filename = (
            os.path.join(recordings_dir, filename)
            if filename
            else os.path.join(
                recordings_dir,
                f"{self.prefix}.{self.guid}{RECORDING_SUFFIX}",
            )
        )
        if recordings_dir:
            os.makedirs(recordings_dir, exist_ok=True)

    def record(self, data: dict[str, Any]) -> None:
        """Append a timestamped event to the recording file."""
        event: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        with open(self.filename, "a", encoding="utf-8") as f:
            json.dump(event, f)
            f.write("\n")

    def get(self) -> list[dict[str, Any]]:
        """Load all recorded events."""
        if not os.path.isfile(self.filename):
            return []
        events: list[dict[str, Any]] = []
        with open(self.filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events

    @classmethod
    def list(cls) -> list[str]:
        recordings_dir = get_recordings_dir()
        if recordings_dir:
            os.makedirs(recordings_dir, exist_ok=True)
            filenames = os.listdir(recordings_dir)
        else:
            filenames = []
        return [f for f in filenames if f.endswith(RECORDING_SUFFIX)]

    def __repr__(self) -> str:
        return f"<SolanaRecorder guid={self.guid} file={self.filename}>"
