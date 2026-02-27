"""
REST API — Phase 9 stub (FastAPI skeleton).

Phase 9 will implement:
  - Full async REST API endpoints
  - WebSocket streaming
  - Job submission and polling
  - Authentication middleware
"""

from __future__ import annotations


def create_app():
    """
    Create a FastAPI application instance.

    [Phase 9] Will return a fully-configured FastAPI app with
    generation endpoints, job queue integration, and WebSocket support.
    """
    try:
        from fastapi import FastAPI
        app = FastAPI(
            title="MultiGenAI OS API",
            description="REST API for Multimodal Generation Operating System",
            version="0.1.0",
        )

        @app.get("/health")
        async def health():
            return {"status": "ok", "phase": 1}

        @app.get("/capability")
        async def capability():
            from multigenai.core.capability_report import CapabilityReport
            return CapabilityReport().to_dict()

        return app

    except ImportError:
        raise ImportError(
            "FastAPI not installed. Run: pip install fastapi uvicorn\n"
            "Full REST API activates in Phase 9."
        )
