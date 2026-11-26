"""Gesture catalog, debounced event trigger engine, and async WebSocket event broadcaster."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from fastapi import WebSocket
from pydantic import BaseModel, Field

from gesture_gcn.inference.recognizer import PredictionResult

# Canonical gesture catalog (internal label -> Chinese display metadata)
PRESET_GESTURES: list[dict[str, str]] = [
    {"id": "static_open_palm", "zh": "张开手掌", "kind": "static"},
    {"id": "static_fist", "zh": "握拳", "kind": "static"},
    {"id": "static_ok_pinch", "zh": "拇指食指捏合", "kind": "static"},
    {"id": "static_peace", "zh": "剪刀手", "kind": "static"},
    {"id": "dynamic_swipe_right", "zh": "向右划", "kind": "dynamic"},
    {"id": "dynamic_swipe_left", "zh": "向左划", "kind": "dynamic"},
    {"id": "dynamic_swipe_up", "zh": "向上划", "kind": "dynamic"},
    {"id": "dynamic_swipe_down", "zh": "向下划", "kind": "dynamic"},
    {"id": "dynamic_push", "zh": "前推", "kind": "dynamic"},
    {"id": "dynamic_pull", "zh": "后拉", "kind": "dynamic"},
    {"id": "dynamic_grab_close", "zh": "抓握", "kind": "dynamic"},
    {"id": "dynamic_circle_wave", "zh": "画圈挥动", "kind": "dynamic"},
    {"id": "static_idle", "zh": "无动作", "kind": "static"},
]

IDLE_LABELS: frozenset[str] = frozenset({"static_idle", "idle", "none", "无动作"})

OPPOSITE_GESTURES: dict[str, str] = {
    "dynamic_swipe_right": "dynamic_swipe_left",
    "dynamic_swipe_left": "dynamic_swipe_right",
    "dynamic_swipe_up": "dynamic_swipe_down",
    "dynamic_swipe_down": "dynamic_swipe_up",
    "dynamic_push": "dynamic_pull",
    "dynamic_pull": "dynamic_push",
}


class ControlConfig(BaseModel):
    """Runtime configuration for gesture-to-event triggering (exposed via REST API)."""

    enabled: bool = True
    confidence_threshold: float = Field(default=0.72, ge=0.30, le=0.99)
    hold_frames: int = Field(default=4, ge=1, le=20)
    cooldown_sec: float = Field(default=0.9, ge=0.1, le=5.0)
    opposite_cooldown_sec: float = Field(default=1.6, ge=0.1, le=5.0)
    require_left_lock: bool = False
    lock_gesture: str = "static_open_palm"


@dataclass
class _HandTracker:
    streak_label: str = ""
    streak_count: int = 0
    last_trigger_ts: float = 0.0
    last_trigger_label: str = ""
    still_frames: int = 0
    latched_static_label: str = ""
    latest_label: str = ""
    latest_conf: float = 0.0
    latest_ts: float = 0.0


class GestureEventEngine:
    """Converts noisy per-frame predictions into clean, debounced gesture trigger events."""

    def __init__(self, config: ControlConfig | None = None) -> None:
        self.config = config or ControlConfig()
        self._trackers: dict[str, _HandTracker] = {}

    def update_config(self, new_cfg: ControlConfig) -> ControlConfig:
        self.config = new_cfg
        return self.config

    def reset(self, hand: str | None = None) -> None:
        if hand is None:
            self._trackers.clear()
        else:
            self._trackers.pop(hand, None)

    def _is_left_lock_active(self, now: float) -> bool:
        left = self._trackers.get("Left")
        if left is None:
            return False
        return (
            left.latest_label == self.config.lock_gesture
            and left.latest_conf >= self.config.confidence_threshold
            and (now - left.latest_ts) <= 0.65
        )

    def process(
        self,
        hand: str,
        pred: PredictionResult,
        now: float | None = None,
    ) -> dict[str, Any] | None:
        """Feed one hand's PredictionResult and return a gesture_triggered event dict if fired."""
        ts = time.time() if now is None else now
        tr = self._trackers.setdefault(hand, _HandTracker())
        tr.latest_label = pred.label
        tr.latest_conf = pred.confidence
        tr.latest_ts = ts

        if not self.config.enabled:
            return None

        # Optional two-hand safety lock: Left hand acts as dead-man's switch for Right hand actions
        if self.config.require_left_lock:
            if hand == "Left":
                return None
            if not self._is_left_lock_active(ts):
                tr.streak_label = ""
                tr.streak_count = 0
                return None

        # Brief stationary pause (~0.25s) clears the return-stroke opposite lock so intentional reverse works fast
        if pred.motion_energy < 0.05:
            tr.still_frames += 1
            if tr.still_frames >= 6:
                tr.last_trigger_label = ""
        else:
            tr.still_frames = 0

        # Ignore idle/background poses and release any latched static gesture
        if pred.label in IDLE_LABELS or pred.confidence < self.config.confidence_threshold:
            tr.streak_label = ""
            tr.streak_count = 0
            if pred.label in IDLE_LABELS:
                tr.latched_static_label = ""
            return None

        # Suppress immediate opposite-direction return stroke (e.g. swipe_right -> returning hand left)
        if (
            tr.last_trigger_label
            and pred.label == OPPOSITE_GESTURES.get(tr.last_trigger_label)
            and (ts - tr.last_trigger_ts) < self.config.opposite_cooldown_sec
        ):
            tr.streak_label = ""
            tr.streak_count = 0
            return None

        if (ts - tr.last_trigger_ts) < self.config.cooldown_sec:
            tr.streak_label = ""
            tr.streak_count = 0
            return None

        if pred.label == tr.streak_label:
            tr.streak_count += 1
        else:
            tr.streak_label = pred.label
            tr.streak_count = 1
            if tr.latched_static_label and pred.label != tr.latched_static_label:
                tr.latched_static_label = ""

        if tr.streak_count < self.config.hold_frames:
            return None

        is_dynamic_gesture = pred.is_dynamic or pred.label.startswith("dynamic_")

        # Static gestures latch until hand leaves or switches pose (avoids spamming F5/Esc)
        if not is_dynamic_gesture and tr.latched_static_label == pred.label:
            return None

        tr.last_trigger_ts = ts
        tr.last_trigger_label = pred.label
        tr.still_frames = 0
        tr.streak_count = 0
        if not is_dynamic_gesture:
            tr.latched_static_label = pred.label

        return {
            "event": "gesture_triggered",
            "hand": hand,
            "gesture": pred.label,
            "confidence": round(pred.confidence, 4),
            "is_dynamic": bool(is_dynamic_gesture),
            "motion_energy": pred.motion_energy,
            "timestamp": round(ts, 3),
        }


class EventBroadcaster:
    """Async pub/sub hub broadcasting gesture trigger events to external /ws/events clients."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()

    @property
    def subscriber_count(self) -> int:
        return len(self._clients)

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self._clients.discard(websocket)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        if not self._clients:
            return
        dead: list[WebSocket] = []
        await asyncio.gather(
            *(self._safe_send(ws, payload, dead) for ws in list(self._clients))
        )
        for ws in dead:
            self._clients.discard(ws)

    async def _safe_send(
        self,
        ws: WebSocket,
        payload: dict[str, Any],
        dead: list[WebSocket],
    ) -> None:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
