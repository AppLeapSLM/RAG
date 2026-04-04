from __future__ import annotations

import asyncio
import io
import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Any

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from backend.config import settings
from backend.connectors.base import (
    BaseConnector,
    ConnectorFile,
    SyncResult,
    SyncStatus,
)

logger = logging.getLogger(__name__)

# Scopes needed: read-only access to Drive files + metadata
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Google Workspace native types that must be EXPORTED (not downloaded)
# Maps native MIME type -> export MIME type + file extension
EXPORT_MAP: dict[str, tuple[str, str]] = {
    "application/vnd.google-apps.document": ("text/html", ".html"),
    "application/vnd.google-apps.spreadsheet": ("text/csv", ".csv"),
    "application/vnd.google-apps.presentation": ("text/plain", ".txt"),
}

# File types we can parse via Unstructured (matches ALLOWED_EXTENSIONS in main.py)
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
    ".md", ".txt", ".html", ".htm", ".rst", ".csv", ".json", ".xml",
}

# Max export size for Google native docs (10MB Google API limit)
MAX_EXPORT_BYTES = 10 * 1024 * 1024

# Page size for Drive API list calls (max 1000)
LIST_PAGE_SIZE = 100


def _get_extension(filename: str) -> str:
    """Extract lowercase file extension."""
    _, ext = os.path.splitext(filename)
    return ext.lower()


def _build_folder_path(file_meta: dict, folder_names: dict[str, str]) -> str:
    """Build a human-readable folder path from parent IDs."""
    parents = file_meta.get("parents", [])
    if not parents:
        return "/"
    parts = []
    for pid in parents:
        parts.append(folder_names.get(pid, pid))
    return "/" + "/".join(parts)


class GoogleDriveConnector(BaseConnector):
    """Pulls files from Google Drive via a service account.

    Setup:
    1. Create a GCP project + enable the Google Drive API
    2. Create a service account and download the JSON key
    3. Share Drive folders/Shared Drives with the service account email
    4. Set APPLEAP_GOOGLE_DRIVE_CREDENTIALS_PATH to the JSON key path
    """

    source_name = "google_drive"

    def __init__(self, credentials_path: str | None = None):
        self._credentials_path = credentials_path or settings.google_drive_credentials_path
        self._service: Any = None  # googleapiclient Resource
        self._credentials: Any = None

    # ── Authentication ─────────────────────────────────────────────

    async def authenticate(self) -> None:
        """Load service account credentials and build the Drive API client."""
        if not self._credentials_path:
            raise ValueError(
                "Google Drive credentials path not configured. "
                "Set APPLEAP_GOOGLE_DRIVE_CREDENTIALS_PATH to the service account JSON key."
            )
        if not os.path.exists(self._credentials_path):
            raise FileNotFoundError(
                f"Service account key not found: {self._credentials_path}"
            )

        def _build_service():
            creds = service_account.Credentials.from_service_account_file(
                self._credentials_path, scopes=SCOPES
            )
            self._credentials = creds
            self._service = build("drive", "v3", credentials=creds)

        await asyncio.to_thread(_build_service)
        logger.info("Google Drive connector authenticated via service account")

    # ── List files ─────────────────────────────────────────────────

    async def list_files(self) -> list[ConnectorFile]:
        """List all files the service account can access.

        Pulls file metadata + permissions for each file.
        Returns ConnectorFile objects (file_path not yet set — download is separate).
        """
        if not self._service:
            raise RuntimeError("Call authenticate() before list_files()")

        raw_files = await self._list_all_files()
        logger.info("Google Drive: found %d files", len(raw_files))

        # Build a folder name lookup for readable paths
        folder_names = await self._list_folder_names()

        connector_files: list[ConnectorFile] = []
        for f in raw_files:
            mime = f.get("mimeType", "")
            name = f.get("name", "unknown")
            file_id = f["id"]

            # Skip Google types we can't export (Forms, Sites, Maps, etc.)
            if mime.startswith("application/vnd.google-apps.") and mime not in EXPORT_MAP:
                logger.debug("Skipping unsupported native type: %s (%s)", name, mime)
                continue

            # For non-native files, check extension
            if not mime.startswith("application/vnd.google-apps."):
                ext = _get_extension(name)
                if ext not in SUPPORTED_EXTENSIONS:
                    logger.debug("Skipping unsupported extension: %s", name)
                    continue

            # Pull permissions for this file
            permissions = await self._get_file_permissions(file_id)

            folder_path = _build_folder_path(f, folder_names)
            owners = f.get("owners", [])
            owner_email = owners[0].get("emailAddress", "") if owners else ""
            last_modified = f.get("modifiedTime", "")

            connector_files.append(
                ConnectorFile(
                    file_path="",  # set during download
                    filename=name,
                    mime_type=mime,
                    source_id=file_id,
                    source=self.source_name,
                    metadata={
                        "drive_file_id": file_id,
                        "mime_type": mime,
                        "owner_email": owner_email,
                        "last_modified": last_modified,
                        "folder_path": folder_path,
                        "web_view_link": f.get("webViewLink", ""),
                    },
                    permissions=permissions,
                )
            )

        logger.info("Google Drive: %d files eligible for ingestion", len(connector_files))
        return connector_files

    # ── Download / Export ──────────────────────────────────────────

    async def download_file(self, connector_file: ConnectorFile) -> ConnectorFile:
        """Download or export a file to a local temp path."""
        mime = connector_file.mime_type
        file_id = connector_file.source_id

        if mime in EXPORT_MAP:
            # Google native doc — export to HTML/CSV/text
            export_mime, export_ext = EXPORT_MAP[mime]
            content = await self._export_file(file_id, export_mime)
            suffix = export_ext
        else:
            # Uploaded file — direct download
            content = await self._download_file(file_id)
            suffix = _get_extension(connector_file.filename) or ".bin"

        # Write to temp file
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        try:
            os.write(fd, content)
        finally:
            os.close(fd)

        connector_file.file_path = tmp_path
        # Update filename to reflect export format if it was a native doc
        if mime in EXPORT_MAP:
            base_name = os.path.splitext(connector_file.filename)[0]
            connector_file.filename = base_name + suffix
            connector_file.mime_type = EXPORT_MAP[mime][0]

        return connector_file

    # ── Private helpers ────────────────────────────────────────────

    async def _list_all_files(self) -> list[dict]:
        """Page through files.list to get all accessible files (non-folders)."""
        all_files: list[dict] = []
        page_token: str | None = None

        while True:
            result = await asyncio.to_thread(
                self._service.files()
                .list(
                    pageSize=LIST_PAGE_SIZE,
                    fields=(
                        "nextPageToken, "
                        "files(id, name, mimeType, modifiedTime, owners, parents, "
                        "webViewLink, size)"
                    ),
                    # Exclude folders and trashed files
                    q="mimeType != 'application/vnd.google-apps.folder' and trashed = false",
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute
            )

            files = result.get("files", [])
            all_files.extend(files)
            page_token = result.get("nextPageToken")
            if not page_token:
                break

        return all_files

    async def _list_folder_names(self) -> dict[str, str]:
        """Build a mapping of folder ID -> folder name for readable paths."""
        folder_map: dict[str, str] = {}
        page_token: str | None = None

        while True:
            result = await asyncio.to_thread(
                self._service.files()
                .list(
                    pageSize=LIST_PAGE_SIZE,
                    fields="nextPageToken, files(id, name)",
                    q="mimeType = 'application/vnd.google-apps.folder' and trashed = false",
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute
            )

            for f in result.get("files", []):
                folder_map[f["id"]] = f["name"]
            page_token = result.get("nextPageToken")
            if not page_token:
                break

        return folder_map

    async def _get_file_permissions(self, file_id: str) -> list[str]:
        """Get the effective permissions for a file.

        Returns a list of email addresses and group emails that have access.
        Includes inherited permissions (from parent folders / Shared Drives).
        """
        try:
            result = await asyncio.to_thread(
                self._service.permissions()
                .list(
                    fileId=file_id,
                    fields="permissions(emailAddress, type, role, displayName)",
                    supportsAllDrives=True,
                )
                .execute
            )

            permissions: list[str] = []
            for perm in result.get("permissions", []):
                perm_type = perm.get("type", "")
                email = perm.get("emailAddress", "")

                if perm_type in ("user", "group") and email:
                    permissions.append(email)
                elif perm_type == "domain":
                    # Domain-wide access — store as "domain:company.com"
                    domain = perm.get("displayName", "") or email
                    permissions.append(f"domain:{domain}")
                elif perm_type == "anyone":
                    permissions.append("anyone")

            return permissions

        except Exception as e:
            # Some files (especially in Shared Drives) may restrict permission listing
            logger.warning("Could not fetch permissions for %s: %s", file_id, e)
            return []

    async def _export_file(self, file_id: str, export_mime: str) -> bytes:
        """Export a Google native doc (Docs/Sheets/Slides) to the given MIME type."""

        def _do_export():
            request = self._service.files().export_media(
                fileId=file_id, mimeType=export_mime
            )
            buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(buffer, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            return buffer.getvalue()

        content = await asyncio.to_thread(_do_export)
        if len(content) > MAX_EXPORT_BYTES:
            logger.warning("Export of %s exceeds 10MB, may be truncated", file_id)
        return content

    async def _download_file(self, file_id: str) -> bytes:
        """Download an uploaded (non-native) file's raw bytes."""

        def _do_download():
            request = self._service.files().get_media(fileId=file_id)
            buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(buffer, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            return buffer.getvalue()

        return await asyncio.to_thread(_do_download)
