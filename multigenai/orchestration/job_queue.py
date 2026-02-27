"""
JobQueue — In-process job queue.

Phase 1: Wraps Python queue.Queue for synchronous job processing.
Phase 9: Will be replaced by Redis/Celery/Ray for distributed processing.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, Optional


class JobQueue:
    """
    Simple in-process job queue.

    Usage:
        q = JobQueue()
        job_id = q.enqueue(my_fn, prompt="hello")
        result = q.process_next()
    """

    def __init__(self) -> None:
        self._queue: list = []
        self._results: Dict[str, Any] = {}

    def enqueue(self, fn: Callable, **kwargs) -> str:
        """Add a job to the queue. Returns a job ID."""
        job_id = uuid.uuid4().hex[:8]
        self._queue.append({"job_id": job_id, "fn": fn, "kwargs": kwargs})
        return job_id

    def process_next(self) -> Optional[Dict[str, Any]]:
        """Process the next job synchronously. Returns result dict or None."""
        if not self._queue:
            return None
        job = self._queue.pop(0)
        try:
            result = job["fn"](**job["kwargs"])
            self._results[job["job_id"]] = {"status": "done", "result": result}
        except Exception as exc:
            self._results[job["job_id"]] = {"status": "failed", "error": str(exc)}
        return self._results[job["job_id"]]

    def process_all(self) -> Dict[str, Any]:
        """Process all queued jobs. Returns {job_id: result}."""
        results = {}
        while self._queue:
            job = self._queue.pop(0)
            try:
                result = job["fn"](**job["kwargs"])
                results[job["job_id"]] = {"status": "done", "result": result}
            except Exception as exc:
                results[job["job_id"]] = {"status": "failed", "error": str(exc)}
        self._results.update(results)
        return results

    def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Return the result for a completed job, or None."""
        return self._results.get(job_id)

    def pending_count(self) -> int:
        return len(self._queue)
