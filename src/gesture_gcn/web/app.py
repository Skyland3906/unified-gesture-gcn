"""FastAPI + WebSocket backend serving real-time UnifiedGestureGCN inference & live training."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from gesture_gcn.config import ExperimentConfig
from gesture_gcn.data.skeleton import get_skeleton_spec
from gesture_gcn.data.synthetic import GESTURE_LABELS, build_synthetic_dataloaders
from gesture_gcn.data.transforms import SkeletonPreprocessor
from gesture_gcn.engine.trainer import Trainer, resolve_device
from gesture_gcn.inference.buffer import SlidingWindowBuffer
from gesture_gcn.inference.recognizer import GestureRecognizer
from gesture_gcn.inference.recorder import CustomGestureStore
from gesture_gcn.models.network import UnifiedGestureGCN

STATIC_DIR = Path(__file__).resolve().parent / "static"


class RecordSampleRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=48)
    frames: list[list[list[float]]] = Field(..., min_length=4)


class TrainRequest(BaseModel):
    epochs: int = Field(default=18, ge=4, le=100)


class _AppState:
    def __init__(
        self,
        cfg: ExperimentConfig,
        checkpoint_path: Path,
        custom_store_path: Path,
    ) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.checkpoint_path = checkpoint_path
        self.spec = get_skeleton_spec(cfg.data.skeleton_type)
        self.store = CustomGestureStore(
            store_path=custom_store_path,
            seq_len=cfg.data.seq_len,
            num_joints=self.spec.num_joints,
        )
        self.recognizer = self._load_or_bootstrap_recognizer()
        self.infer_lock = threading.Lock()

    def _load_or_bootstrap_recognizer(self) -> GestureRecognizer:
        if self.checkpoint_path.exists():
            recognizer, _ = GestureRecognizer.from_checkpoint(self.checkpoint_path, self.device)
            return recognizer

        # Bootstrap a ready-to-run checkpoint using the synthetic static+dynamic benchmark
        torch.manual_seed(self.cfg.seed)
        train_loader, val_loader, spec, in_channels = build_synthetic_dataloaders(
            self.cfg.data, seed=self.cfg.seed
        )
        model = UnifiedGestureGCN(cfg=self.cfg.model, spec=spec, in_channels=in_channels)
        trainer = Trainer(model=model, cfg=self.cfg.train, device=self.device)
        trainer.fit(train_loader, val_loader)

        preprocessor = SkeletonPreprocessor(
            spec=spec,
            use_bone_fusion=self.cfg.data.use_bone_fusion,
            scale_floor=self.cfg.data.norm_scale_floor,
        )
        recognizer = GestureRecognizer(
            model=model,
            preprocessor=preprocessor,
            labels=list(GESTURE_LABELS),
            device=self.device,
        )
        recognizer.compile_for_inference(self.cfg.data.seq_len)
        recognizer.save_checkpoint(self.checkpoint_path, self.cfg.data)
        return recognizer


def create_app(
    config_path: Path | None = None,
    checkpoint_path: Path | None = None,
    custom_store_path: Path | None = None,
) -> FastAPI:
    project_root = Path(__file__).resolve().parents[3]
    cfg = (
        ExperimentConfig.from_yaml(config_path)
        if config_path is not None
        else ExperimentConfig.from_yaml(project_root / "configs" / "default.yaml")
    )
    ckpt = checkpoint_path or (project_root / "checkpoints" / "unified_gesture_gcn.pt")
    store = custom_store_path or (project_root / "data" / "custom_gestures.json")

    state = _AppState(cfg=cfg, checkpoint_path=ckpt, custom_store_path=store)
    app = FastAPI(title="Unified Gesture GCN Live Studio", version="0.1.0")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "index.html",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        return {
            "experiment_name": state.cfg.experiment_name,
            "device": str(state.device),
            "skeleton_type": state.spec.name,
            "num_joints": state.spec.num_joints,
            "seq_len": state.cfg.data.seq_len,
            "params_m": round(state.recognizer.model.count_parameters() / 1e6, 3),
            "labels": state.recognizer.labels,
            "recorded_counts": state.store.counts(),
        }

    @app.post("/api/samples")
    async def add_sample(req: RecordSampleRequest) -> dict[str, Any]:
        try:
            count = state.store.add_clip(req.label, req.frames)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "label": req.label,
            "count": count,
            "recorded_counts": state.store.counts(),
        }

    @app.delete("/api/samples")
    async def clear_samples() -> dict[str, Any]:
        state.store.clear()
        return {"recorded_counts": {}}

    @app.post("/api/train")
    async def train_custom(req: TrainRequest) -> dict[str, Any]:
        try:
            model, labels, train_acc = await asyncio.to_thread(
                state.store.train_model,
                base_model_cfg=state.cfg.model,
                data_cfg=state.cfg.data,
                spec=state.spec,
                device=state.device,
                epochs=req.epochs,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        preprocessor = SkeletonPreprocessor(
            spec=state.spec,
            use_bone_fusion=state.cfg.data.use_bone_fusion,
            scale_floor=state.cfg.data.norm_scale_floor,
        )
        new_recognizer = GestureRecognizer(
            model=model,
            preprocessor=preprocessor,
            labels=labels,
            device=state.device,
        )
        await asyncio.to_thread(new_recognizer.compile_for_inference, state.cfg.data.seq_len)
        await asyncio.to_thread(new_recognizer.save_checkpoint, state.checkpoint_path, state.cfg.data)
        with state.infer_lock:
            state.recognizer = new_recognizer
        return {
            "labels": labels,
            "train_acc": round(train_acc, 4),
            "params_m": round(model.count_parameters() / 1e6, 3),
        }

    def _run_predict_batch(seqs_by_key: dict[str, torch.Tensor]) -> dict[str, Any]:
        with state.infer_lock:
            return state.recognizer.predict_batch(seqs_by_key)

    @app.websocket("/ws/predict")
    async def ws_predict(websocket: WebSocket) -> None:
        await websocket.accept()
        conn_id = id(websocket)
        hands = ("Left", "Right")
        bufs: dict[str, SlidingWindowBuffer] = {
            hand: SlidingWindowBuffer(
                seq_len=state.cfg.data.seq_len,
                num_joints=state.spec.num_joints,
            )
            for hand in hands
        }

        def hand_key(hand: str) -> str:
            return f"{conn_id}:{hand}"

        def reset_hand(hand: str | None) -> None:
            targets = hands if hand is None else (hand,)
            with state.infer_lock:
                for name in targets:
                    if name in bufs:
                        bufs[name].clear()
                        state.recognizer.reset_smoothing(hand_key(name))

        def serialize_pred(hand: str, pred: Any) -> dict[str, Any]:
            return {
                "hand": hand,
                "label": pred.label,
                "confidence": round(pred.confidence, 4),
                "probabilities": pred.probabilities,
                "motion_energy": pred.motion_energy,
                "is_dynamic": pred.is_dynamic,
                "latency_ms": pred.latency_ms,
                "buffer_fill": len(bufs[hand]) / bufs[hand].seq_len,
            }

        try:
            while True:
                msg = await websocket.receive_json()
                action = msg.get("action")
                if action == "reset":
                    reset_hand(msg.get("hand"))
                    await websocket.send_json({"status": "reset", "hand": msg.get("hand")})
                    continue

                # Preferred batched protocol: {"hands": {"Left": [[...]], "Right": [[...]]}}
                hands_payload = msg.get("hands")
                if isinstance(hands_payload, dict) and hands_payload:
                    seqs_by_key: dict[str, torch.Tensor] = {}
                    key_to_hand: dict[str, str] = {}
                    try:
                        for raw_hand, lms in hands_payload.items():
                            if not lms:
                                continue
                            hand = "Left" if raw_hand == "Left" else "Right"
                            bufs[hand].push(lms)
                            k = hand_key(hand)
                            seqs_by_key[k] = bufs[hand].get_sequence()
                            key_to_hand[k] = hand
                    except ValueError as exc:
                        await websocket.send_json({"error": str(exc)})
                        continue

                    if not seqs_by_key:
                        continue

                    preds = await asyncio.to_thread(_run_predict_batch, seqs_by_key)
                    results = {
                        key_to_hand[k]: serialize_pred(key_to_hand[k], pred)
                        for k, pred in preds.items()
                    }
                    await websocket.send_json({"results": results})
                    continue

                # Backwards-compatible single-hand protocol: {"landmarks": [[...]], "hand": "Right"}
                landmarks = msg.get("landmarks")
                if not landmarks:
                    continue

                hand = "Left" if msg.get("hand") == "Left" else "Right"
                try:
                    bufs[hand].push(landmarks)
                    k = hand_key(hand)
                    preds = await asyncio.to_thread(_run_predict_batch, {k: bufs[hand].get_sequence()})
                except ValueError as exc:
                    await websocket.send_json({"error": str(exc), "hand": hand})
                    continue

                await websocket.send_json(serialize_pred(hand, preds[k]))
        except WebSocketDisconnect:
            return
        finally:
            reset_hand(None)

    return app
