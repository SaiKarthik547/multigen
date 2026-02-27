"""
WorldStateEngine — Scene/world continuity tracking.

Tracks the current state of the generated world (time of day, weather,
object positions, etc.) and persists snapshots for cross-scene continuity.

Phase 5 will use this to inject consistent environmental hints into
each frame of video generation.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from multigenai.core.logging.logger import get_logger

LOG = get_logger(__name__)


@dataclass
class WorldState:
    """
    Snapshot of the world at a point in time.

    Attributes:
        scene_id:        Unique scene identifier.
        time_of_day:     "dawn" | "morning" | "noon" | "afternoon" | "dusk" | "night"
        weather:         "clear" | "overcast" | "rain" | "snow" | "fog" | "storm"
        lighting_vector: Dominant light direction [x, y, z] (normalised).
        object_positions: {object_name: {"x": float, "y": float, "z": float}}
        notes:           Free-form annotation for the LLM context.
        timestamp:       ISO8601 UTC timestamp of the snapshot.
    """
    scene_id: str
    time_of_day: str = "noon"
    weather: str = "clear"
    lighting_vector: List[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])
    object_positions: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WorldState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class WorldStateEngine:
    """
    Manages a stack of WorldState snapshots for the current session.

    Persists the history to `<store_dir>/world_state.json` so it survives
    between CLI runs.

    Usage:
        wse = WorldStateEngine()
        wse.update(WorldState(scene_id="s01", time_of_day="dusk", weather="rain"))
        snap = wse.snapshot()   # returns current WorldState
        wse.reset()
    """

    def __init__(self, store_dir: str = "multigen_outputs/.memory") -> None:
        self._root = pathlib.Path(store_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._history_path = self._root / "world_state.json"
        self._history: List[WorldState] = self._load()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, state: WorldState) -> None:
        """Push a new world state snapshot."""
        self._history.append(state)
        self._save()
        LOG.debug(f"WorldState updated: scene={state.scene_id}, weather={state.weather}")

    def snapshot(self) -> Optional[WorldState]:
        """Return the most recent WorldState, or None if history is empty."""
        return self._history[-1] if self._history else None

    def history(self) -> List[WorldState]:
        """Return all historical states in chronological order."""
        return list(self._history)

    def reset(self) -> None:
        """Clear all world state history."""
        self._history = []
        self._save()
        LOG.info("WorldState history reset.")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        data = [s.to_dict() for s in self._history]
        self._history_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self) -> List[WorldState]:
        if not self._history_path.exists():
            return []
        try:
            raw = json.loads(self._history_path.read_text(encoding="utf-8"))
            return [WorldState.from_dict(d) for d in raw]
        except Exception as exc:
            LOG.warning(f"Could not load world state history: {exc}")
            return []
