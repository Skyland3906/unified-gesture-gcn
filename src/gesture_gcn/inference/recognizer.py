"""Real-time gesture recognizer with checkpoint I/O, TorchScript acceleration, batched multi-hand inference, and adaptive EMA smoothing."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch

from gesture_gcn.config import DataConfig, ModelConfig
from gesture_gcn.data.skeleton import HandSkeletonSpec, get_skeleton_spec
from gesture_gcn.data.transforms import SkeletonPreprocessor
from gesture_gcn.models.network import UnifiedGestureGCN


@dataclass(frozen=True)
class PredictionResult:
    label: str
    confidence: float
    probabilities: dict[str, float]
    motion_energy: float
    is_dynamic: bool
    latency_ms: float


class GestureRecognizer:
    """Fast inference wrapper around UnifiedGestureGCN with TorchScript & multi-hand batching."""

    def __init__(
        self,
        model: UnifiedGestureGCN,
        preprocessor: SkeletonPreprocessor,
        labels: list[str],
        device: torch.device,
        ema_alpha: float = 0.45,
        ema_fast_alpha: float = 0.72,
        dynamic_threshold: float = 0.08,
        optimize_inference: bool = True,
    ) -> None:
        self.model = model.to(device).eval()
        self.preprocessor = preprocessor
        self.labels = list(labels)
        self.device = device
        self.ema_alpha = ema_alpha
        self.ema_fast_alpha = max(ema_alpha, ema_fast_alpha)
        self.dynamic_threshold = dynamic_threshold
        self._ema_by_hand: dict[str, torch.Tensor] = {}
        self._optimize_inference = optimize_inference
        self._fast_runners: dict[int, torch.jit.ScriptModule] = {}

        # Small 0.6M-param GCNs suffer from OpenMP thread contention when PyTorch spawns many threads
        if device.type == "cpu" and torch.get_num_threads() > 4:
            torch.set_num_threads(4)

    def reset_smoothing(self, hand: str | None = None) -> None:
        if hand is None:
            self._ema_by_hand.clear()
        else:
            self._ema_by_hand.pop(hand, None)

    def compile_for_inference(self, seq_len: int) -> bool:
        """Trace and optimize the model for a given temporal window length (supports dynamic batch size)."""
        if not self._optimize_inference:
            return False
        if seq_len in self._fast_runners:
            return True
        try:
            num_joints = self.preprocessor.spec.num_joints
            in_ch = self.preprocessor.out_channels
            dummy_x = torch.zeros(1, in_ch, seq_len, num_joints, device=self.device)
            dummy_dr = torch.zeros(1, 3, num_joints, device=self.device)
            with torch.inference_mode(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                traced = torch.jit.trace(self.model, (dummy_x, dummy_dr), check_trace=False)
                optimized = torch.jit.optimize_for_inference(traced)
                # Warmup both B=1 and B=2 to verify dynamic batch support
                _ = optimized(dummy_x, dummy_dr)
                _ = optimized(dummy_x.repeat(2, 1, 1, 1), dummy_dr.repeat(2, 1, 1))
            self._fast_runners[seq_len] = optimized
            return True
        except Exception:
            return False

    def _forward(self, x: torch.Tensor, dr: torch.Tensor) -> torch.Tensor:
        seq_len = int(x.shape[2])
        runner = self._fast_runners.get(seq_len)
        if runner is None and self._optimize_inference:
            self.compile_for_inference(seq_len)
            runner = self._fast_runners.get(seq_len)
        if runner is not None:
            return runner(x, dr)
        return self.model(x, dr)

    def _update_ema(self, hand: str, probs: torch.Tensor) -> torch.Tensor:
        prev = self._ema_by_hand.get(hand)
        if prev is None or prev.shape != probs.shape:
            smoothed = probs
        else:
            raw_conf, raw_idx = probs.max(dim=0)
            prev_idx = prev.argmax(dim=0)
            # Adaptive EMA: respond faster when confident or cleanly switching gesture, stay smooth on jitter
            if raw_conf.item() >= 0.72 and (raw_idx.item() != prev_idx.item() or raw_conf.item() >= 0.85):
                alpha = self.ema_fast_alpha
            else:
                alpha = self.ema_alpha
            smoothed = alpha * probs + (1.0 - alpha) * prev
        self._ema_by_hand[hand] = smoothed
        return smoothed

    def _format_result(
        self,
        smoothed: torch.Tensor,
        motion_energy: float,
        latency_ms: float,
    ) -> PredictionResult:
        conf, idx = smoothed.max(dim=0)
        prob_map = {
            self.labels[i]: round(float(smoothed[i].item()), 4)
            for i in range(len(self.labels))
        }
        return PredictionResult(
            label=self.labels[int(idx.item())],
            confidence=float(conf.item()),
            probabilities=prob_map,
            motion_energy=round(motion_energy, 4),
            is_dynamic=motion_energy >= self.dynamic_threshold,
            latency_ms=round(latency_ms, 2),
        )

    @torch.inference_mode()
    def predict(self, raw_seq: torch.Tensor, hand: str = "default") -> PredictionResult:
        """Run inference on a single raw skeleton sequence of shape (3, T, N)."""
        return self.predict_batch({hand: raw_seq})[hand]

    @torch.inference_mode()
    def predict_batch(self, seqs_by_hand: dict[str, torch.Tensor]) -> dict[str, PredictionResult]:
        """Run one batched forward pass for 1 or more hands (e.g. Left + Right -> Batch=2)."""
        if not seqs_by_hand:
            return {}
        start = time.perf_counter()
        hand_names = list(seqs_by_hand.keys())
        feats: list[torch.Tensor] = []
        drs: list[torch.Tensor] = []
        energies: list[float] = []

        for hand in hand_names:
            feat, dr = self.preprocessor(seqs_by_hand[hand])
            feats.append(feat)
            drs.append(dr)
            energies.append(float(dr.mean().item()))

        x = torch.stack(feats, dim=0).to(self.device)
        dr_batch = torch.stack(drs, dim=0).to(self.device)

        logits = self._forward(x, dr_batch)
        probs_batch = torch.softmax(logits, dim=-1).cpu()
        latency_ms = (time.perf_counter() - start) * 1000.0

        results: dict[str, PredictionResult] = {}
        for i, hand in enumerate(hand_names):
            smoothed = self._update_ema(hand, probs_batch[i])
            results[hand] = self._format_result(smoothed, energies[i], latency_ms)
        return results

    def save_checkpoint(self, path: str | Path, data_cfg: DataConfig) -> Path:
        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state": self.model.state_dict(),
            "model_cfg": self.model.cfg.model_dump(),
            "data_cfg": data_cfg.model_dump(),
            "labels": self.labels,
            "in_channels": self.preprocessor.out_channels,
        }
        torch.save(payload, ckpt_path)
        return ckpt_path

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: torch.device,
    ) -> tuple[GestureRecognizer, DataConfig]:
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        model_cfg = ModelConfig.model_validate(payload["model_cfg"])
        data_cfg = DataConfig.model_validate(payload["data_cfg"])
        labels: list[str] = list(payload["labels"])
        in_channels: int = int(payload["in_channels"])

        spec: HandSkeletonSpec = get_skeleton_spec(data_cfg.skeleton_type)
        preprocessor = SkeletonPreprocessor(
            spec=spec,
            use_bone_fusion=data_cfg.use_bone_fusion,
            scale_floor=data_cfg.norm_scale_floor,
        )
        model = UnifiedGestureGCN(cfg=model_cfg, spec=spec, in_channels=in_channels)
        model.load_state_dict(payload["model_state"])
        recognizer = cls(
            model=model,
            preprocessor=preprocessor,
            labels=labels,
            device=device,
        )
        recognizer.compile_for_inference(data_cfg.seq_len)
        return recognizer, data_cfg
