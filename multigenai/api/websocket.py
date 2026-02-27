"""WebSocket streaming — Phase 9 stub."""

from __future__ import annotations


class WebSocketStreamer:
    """Streams generation progress updates over WebSocket. Activates in Phase 9."""

    async def stream_progress(self, websocket, job_id: str):
        """[Phase 9] Stream job progress to a WebSocket client."""
        raise NotImplementedError("WebSocket streaming activates in Phase 9.")
