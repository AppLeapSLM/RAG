from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SyncStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ConnectorFile:
    """A file fetched from an external source, ready for ingestion."""

    file_path: str  # local temp path to downloaded content
    filename: str  # original name (e.g. "runbook.pdf")
    mime_type: str  # e.g. "application/pdf", "text/html"
    source_id: str  # unique ID in the source system (e.g. Drive file ID)
    source: str  # connector name (e.g. "google_drive")
    metadata: dict[str, Any] = field(default_factory=dict)
    permissions: list[str] = field(default_factory=list)


@dataclass
class SyncResult:
    """Summary of a connector sync run."""

    status: SyncStatus
    files_found: int = 0
    files_ingested: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    chunks_stored: int = 0
    errors: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "files_found": self.files_found,
            "files_ingested": self.files_ingested,
            "files_skipped": self.files_skipped,
            "files_failed": self.files_failed,
            "chunks_stored": self.chunks_stored,
            "errors": self.errors[:20],  # cap error list in response
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


class BaseConnector(ABC):
    """Interface that all connectors implement."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Short identifier stored in Document.source (e.g. 'google_drive')."""
        ...

    @abstractmethod
    async def authenticate(self) -> None:
        """Validate credentials and establish connection to the source."""
        ...

    @abstractmethod
    async def list_files(self) -> list[ConnectorFile]:
        """List all files available for ingestion from the source.

        Each ConnectorFile should have permissions populated.
        Does NOT download content yet — file_path may be empty.
        """
        ...

    @abstractmethod
    async def download_file(self, connector_file: ConnectorFile) -> ConnectorFile:
        """Download/export file content to a local temp path.

        Returns the same ConnectorFile with file_path set to the temp file.
        Caller is responsible for cleaning up the temp file.
        """
        ...
