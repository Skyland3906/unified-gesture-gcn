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


OPPOSITE_GESTURES: dict[str, str] = {
    "dynamic_swipe_right": "dynamic_swipe_left",
    "dynamic_swipe_left": "dynamic_swipe_right",
    "dynamic_swipe_up": "dynamic_swipe_down",
    "dynamic_swipe_down": "dynamic_swipe_up",
    "dynamic_push": "dynamic_pull",
    "dynamic_pull": "dynamic_push",
}


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
        use_kinematic_prior: bool = True,
    ) -> None:
        self.model = model.to(device).eval()
        self.preprocessor = preprocessor
        self.labels = list(labels)
        self.device = device
        self.ema_alpha = ema_alpha
        self.ema_fast_alpha = max(ema_alpha, ema_fast_alpha)
        self.dynamic_threshold = dynamic_threshold
        self._ema_by_hand: dict[str, torch.Tensor] = {}
        self._swipe_guard: dict[str, tuple[str, float, int]] = {}  # hand -> (label, ts, still_frames)
        self._optimize_inference = optimize_inference
        self._use_kinematic_prior = use_kinematic_prior
        self._fast_runners: dict[int, torch.jit.ScriptModule] = {}

        # Small 0.6M-param GCNs suffer from OpenMP thread contention when PyTorch spawns many threads
        if device.type == "cpu" and torch.get_num_threads() > 4:
            torch.set_num_threads(4)

    def reset_smoothing(self, hand: str | None = None) -> None:
        if hand is None:
            self._ema_by_hand.clear()
            self._swipe_guard.clear()
        else:
            self._ema_by_hand.pop(hand, None)
            self._swipe_guard.pop(hand, None)

    def compile_for_inference(self, seq_len: int) -> bool:
        """Trace and optimize the model for a given temporal window length (supports dynamic batch size)."""
        if not self._optimize_inference or self.device.type == "mps":
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

    def _apply_kinematic_prior(
        self,
        raw_seq: torch.Tensor,
        probs: torch.Tensor,
        motion_energy: float,
        hand: str | None = None,
        now: float | None = None,
    ) -> torch.Tensor:
        """Enforce axis/direction isolation, suppress return strokes, and block phantom grabs."""
        if not self._use_kinematic_prior or raw_seq.shape[2] != 21 or raw_seq.shape[1] < 4:
            return probs

        ts = time.perf_counter() if now is None else now
        t_len = int(raw_seq.shape[1])
        k = min(14, t_len)
        # Recent ~0.5s wrist displacement (CSS-mirrored webcam: screen-left => dx>0, screen-right => dx<0)
        dx = float((raw_seq[0, -1, 0] - raw_seq[0, -k, 0]).item())
        dy = float((raw_seq[1, -1, 0] - raw_seq[1, -k, 0]).item())
        abs_dx = abs(dx)
        abs_dy = abs(dy)

        # 2D palm scale (wrist 0 -> middle MCP 9)
        palm_vecs = raw_seq[:2, :, 9] - raw_seq[:2, :, 0]  # (2, T)
        palm_scales = torch.linalg.norm(palm_vecs, dim=0).clamp(min=1e-4)  # (T,)
        s_recent = float(palm_scales[-k].item())
        s_end = float(palm_scales[-1].item())
        ds = (s_end - s_recent) / max(s_recent, 0.05)

        # Finger openness: average distance from fingertips (8,12,16,20) to wrist (0) / palm_scale
        tips = raw_seq[:2, :, (8, 12, 16, 20)] - raw_seq[:2, :, 0:1]  # (2, T, 4)
        open_curve = torch.linalg.norm(tips, dim=0).mean(dim=1) / palm_scales  # (T,)
        open_start_full = float(open_curve[: min(3, t_len)].mean().item())
        open_start_recent = float(open_curve[-k].item())
        open_end = float(open_curve[-min(2, t_len) :].mean().item())
        max_close_drop = max(open_start_full - open_end, open_start_recent - open_end)

        # 2D palm-center trajectory geometry for genuine circle-wave validation
        palm_pts = 0.5 * (raw_seq[:2, :, 0] + raw_seq[:2, :, 9])  # (2, T)
        span_x = float((palm_pts[0].max() - palm_pts[0].min()).item())
        span_y = float((palm_pts[1].max() - palm_pts[1].min()).item())
        min_span = min(span_x, span_y)
        max_span = max(span_x, span_y)
        span_ratio = min_span / max(max_span, 1e-4)

        step_diffs = palm_pts[:, 1:] - palm_pts[:, :-1]  # (2, T-1)
        path_len = float(torch.linalg.norm(step_diffs, dim=0).sum().item())
        net_disp = float(torch.linalg.norm(palm_pts[:, -1] - palm_pts[:, 0]).item())
        tortuosity = path_len / max(net_disp, 1e-4)

        center = palm_pts.mean(dim=1, keepdim=True)  # (2, 1)
        rel = palm_pts - center  # (2, T)
        radii = torch.linalg.norm(rel, dim=0)  # (T,)
        cross = rel[0, :-1] * rel[1, 1:] - rel[1, :-1] * rel[0, 1:]
        dot = rel[0, :-1] * rel[0, 1:] + rel[1, :-1] * rel[1, 1:]
        dtheta = torch.atan2(cross, dot)
        valid_steps = (radii[:-1] > 0.015) & (radii[1:] > 0.015)
        if bool(valid_steps.any().item()):
            valid_dtheta = dtheta[valid_steps]
            abs_net_angle = float(abs(valid_dtheta.sum().item()))
            total_abs_angle = float(valid_dtheta.abs().sum().item())
            winding_consistency = abs_net_angle / max(total_abs_angle, 1e-4)
        else:
            abs_net_angle = 0.0
            winding_consistency = 0.0

        is_genuine_circle = (
            min_span >= 0.065
            and span_ratio >= 0.32
            and path_len >= 0.24
            and tortuosity >= 1.65
            and abs_net_angle >= 3.5
            and winding_consistency >= 0.55
        )

        # Check active return-stroke guard for this hand (clears quickly after a brief stationary pause)
        suppressed_opposite: str | None = None
        if hand is not None and hand in self._swipe_guard:
            prev_lbl, prev_ts, still_cnt = self._swipe_guard[hand]
            if (ts - prev_ts) > 1.35:
                self._swipe_guard.pop(hand, None)
            elif motion_energy < 0.05:
                still_cnt += 1
                if still_cnt >= 5:
                    self._swipe_guard.pop(hand, None)
                else:
                    self._swipe_guard[hand] = (prev_lbl, prev_ts, still_cnt)
                    suppressed_opposite = OPPOSITE_GESTURES.get(prev_lbl)
            else:
                self._swipe_guard[hand] = (prev_lbl, prev_ts, 0)
                suppressed_opposite = OPPOSITE_GESTURES.get(prev_lbl)

        weights = torch.ones_like(probs)
        for i, lbl in enumerate(self.labels):
            if lbl == "dynamic_swipe_right":
                # Require genuine screen-right displacement (dx <= -0.08) and horizontal dominance
                if dx > -0.08 or abs_dx < 1.25 * abs_dy:
                    weights[i] *= 0.10
                elif is_genuine_circle:
                    weights[i] *= 0.20
            elif lbl == "dynamic_swipe_left":
                # Require genuine screen-left displacement (dx >= +0.08) and horizontal dominance
                if dx < 0.08 or abs_dx < 1.25 * abs_dy:
                    weights[i] *= 0.10
                elif is_genuine_circle:
                    weights[i] *= 0.20
            elif lbl == "dynamic_swipe_up":
                # Require genuine upward displacement (dy <= -0.08) and vertical dominance
                if dy > -0.08 or abs_dy < 1.25 * abs_dx:
                    weights[i] *= 0.10
                elif is_genuine_circle:
                    weights[i] *= 0.20
            elif lbl == "dynamic_swipe_down":
                # Require genuine downward displacement (dy >= +0.08) and vertical dominance
                if dy < 0.08 or abs_dy < 1.25 * abs_dx:
                    weights[i] *= 0.10
                elif is_genuine_circle:
                    weights[i] *= 0.20
            elif lbl == "dynamic_push":
                if ds < 0.06:
                    weights[i] *= 0.15
            elif lbl == "dynamic_pull":
                if ds > -0.06:
                    weights[i] *= 0.15
            elif lbl == "dynamic_grab_close":
                # Genuine grab must actually close fingers and end relatively closed
                if open_end > 1.50 or max_close_drop < 0.14:
                    weights[i] *= 0.12
            elif lbl == "dynamic_circle_wave":
                # Genuine circle requires both-axis span, loop tortuosity, and >=200 deg consistent sweep
                if not is_genuine_circle:
                    weights[i] *= 0.08
                elif suppressed_opposite is not None and abs_net_angle < 4.5:
                    weights[i] *= 0.20

            # Return-stroke suppression: damp opposite direction immediately after a swipe/push/pull
            if suppressed_opposite is not None and lbl == suppressed_opposite:
                weights[i] *= 0.06

            # Suppress dynamic labels when hand is practically stationary
            if lbl.startswith("dynamic_") and motion_energy < 0.055 and max_close_drop < 0.18:
                weights[i] *= 0.30

        adjusted = probs * weights
        denom = float(adjusted.sum().item())
        return adjusted / denom if denom > 1e-8 else probs

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
        now = time.perf_counter()
        latency_ms = (now - start) * 1000.0

        results: dict[str, PredictionResult] = {}
        for i, hand in enumerate(hand_names):
            refined = self._apply_kinematic_prior(
                seqs_by_hand[hand],
                probs_batch[i],
                energies[i],
                hand=hand,
                now=now,
            )
            smoothed = self._update_ema(hand, refined)
            res = self._format_result(smoothed, energies[i], latency_ms)
            if res.label in OPPOSITE_GESTURES and res.confidence >= 0.58:
                self._swipe_guard[hand] = (res.label, now, 0)
            results[hand] = res
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
        base_ch = 6 if data_cfg.use_bone_fusion else 3
        use_velocity = data_cfg.use_velocity or (in_channels == base_ch + 3)
        preprocessor = SkeletonPreprocessor(
            spec=spec,
            use_bone_fusion=data_cfg.use_bone_fusion,
            scale_floor=data_cfg.norm_scale_floor,
            use_velocity=use_velocity,
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
