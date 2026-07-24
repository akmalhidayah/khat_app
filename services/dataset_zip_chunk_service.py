"""Optional chunked ZIP upload support for very large dataset archives."""

from __future__ import annotations

import os
import uuid
from typing import Dict, Optional

from flask import current_app


class ChunkedZipUploadStore:
    """In-memory chunk registry for resumable large ZIP uploads."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Dict] = {}

    def start_session(self, filename: str, total_chunks: int) -> str:
        session_id = uuid.uuid4().hex
        chunk_dir = os.path.join(
            current_app.config["DATASET_DIR"],
            "processed",
            "zip_chunks",
            session_id,
        )
        os.makedirs(chunk_dir, exist_ok=True)
        self._sessions[session_id] = {
            "filename": filename,
            "total_chunks": total_chunks,
            "received": set(),
            "chunk_dir": chunk_dir,
        }
        return session_id

    def save_chunk(self, session_id: str, chunk_index: int, chunk_data: bytes) -> int:
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError("Sesi upload tidak ditemukan atau sudah kedaluwarsa.")
        chunk_path = os.path.join(session["chunk_dir"], f"chunk_{chunk_index:05d}.part")
        with open(chunk_path, "wb") as handle:
            handle.write(chunk_data)
        session["received"].add(chunk_index)
        return len(session["received"])

    def assemble(self, session_id: str) -> str:
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError("Sesi upload tidak ditemukan atau sudah kedaluwarsa.")
        if len(session["received"]) != session["total_chunks"]:
            raise ValueError("Semua chunk belum diterima.")

        output_dir = os.path.join(current_app.config["DATASET_DIR"], "processed")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"dataset_upload_{session_id}.zip")

        with open(output_path, "wb") as output:
            for index in range(session["total_chunks"]):
                chunk_path = os.path.join(session["chunk_dir"], f"chunk_{index:05d}.part")
                with open(chunk_path, "rb") as chunk:
                    output.write(chunk.read())

        self.cleanup(session_id)
        return output_path

    def cleanup(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if not session:
            return
        chunk_dir = session.get("chunk_dir")
        if chunk_dir and os.path.isdir(chunk_dir):
            for name in os.listdir(chunk_dir):
                try:
                    os.remove(os.path.join(chunk_dir, name))
                except OSError:
                    pass
            try:
                os.rmdir(chunk_dir)
            except OSError:
                pass


_chunk_store: Optional[ChunkedZipUploadStore] = None


def get_chunk_upload_store() -> ChunkedZipUploadStore:
    global _chunk_store
    if _chunk_store is None:
        _chunk_store = ChunkedZipUploadStore()
    return _chunk_store
