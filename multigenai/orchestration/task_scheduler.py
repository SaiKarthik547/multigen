"""
TaskScheduler — Phase 9 stub.

Phase 9 will implement VRAM-aware job scheduling
with priority queues and model warm-pooling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List
import threading
import queue


class JobPriority(Enum):
    LOW = 3
    NORMAL = 2
    HIGH = 1


@dataclass
class Job:
    """A scheduled generation job."""
    job_id: str
    fn: Callable
    args: dict = field(default_factory=dict)
    priority: JobPriority = JobPriority.NORMAL
    result: Any = None
    error: Any = None


class TaskScheduler:
    """
    VRAM-aware task scheduler for generation jobs.

    Phase 1: Simple FIFO in-process queue.
    Phase 9: Priority scheduling, VRAM pre-checks, model warm-pooling.
    """

    def __init__(self, max_workers: int = 1) -> None:
        self._q: queue.Queue = queue.Queue()
        self._max_workers = max_workers
        self._jobs: dict = {}

    def submit(self, job: Job) -> str:
        """Submit a job to the queue. Returns job_id."""
        self._q.put(job)
        self._jobs[job.job_id] = job
        return job.job_id

    def run_sync(self) -> List[Job]:
        """Drain the queue synchronously (Phase 1 mode). Returns completed jobs."""
        completed = []
        while not self._q.empty():
            job = self._q.get()
            try:
                job.result = job.fn(**job.args)
            except Exception as exc:
                job.error = exc
            completed.append(job)
        return completed
