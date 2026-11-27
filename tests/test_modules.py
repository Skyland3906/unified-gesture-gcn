"""Unit tests for Plan A model modules and ablation switches."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from gesture_gcn.config import DataConfig, ModelConfig
from gesture_gcn.data.skeleton import get_skeleton_spec
from gesture_gcn.data.synthetic import _base_open_hand, _curl_fingers
from gesture_gcn.data.transforms import SkeletonPreprocessor
from gesture_gcn.inference.recognizer import GestureRecognizer
from gesture_gcn.models.network import UnifiedGestureGCN


@pytest.mark.parametrize(
    "overrides",
    [
        {},  # Full Plan A
        {"use_dre": False},
        {"use_mstcn": False},
        {"use_aru": False, "use_dru": True},
        {"use_aru": True, "use_dru": False},
        {"use_shape_branch": True},
        {"use_shape_branch": True, "shape_grid_size": 12},
    ],
)
def test_model_forward_and_backward_ablations(overrides: dict) -> None:
    cfg = ModelConfig(num_classes=6, channels=[32, 32, 32, 64], num_groups=4, **overrides)
    spec = get_skeleton_spec("mediapipe_21")
    model = UnifiedGestureGCN(cfg=cfg, spec=spec, in_channels=6)

    x = torch.randn(2, 6, 24, 21)
    dr = torch.rand(2, 3, 21)
    logits = model(x, dr)

    assert logits.shape == (2, 6)
    loss = logits.sum()
    loss.backward()

    # Ensure gradients flow cleanly without NaNs
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite grad in {name}"


def test_relative_shape_branch_direction_rotation_and_torchscript(tmp_path: Path) -> None:
    spec = get_skeleton_spec("mediapipe_21")
    pre = SkeletonPreprocessor(spec, use_bone_fusion=True, use_velocity=True)
    cfg = ModelConfig(
        num_classes=6,
        channels=[32, 32, 32, 64],
        num_groups=4,
        use_shape_branch=True,
        shape_grid_size=8,
    )
    model = UnifiedGestureGCN(cfg=cfg, spec=spec, in_channels=pre.out_channels).eval()
    assert model.shape_branch is not None

    base = _base_open_hand(21)
    t_grid = torch.linspace(0.0, 1.0, 32)

    # 1. Detail 1: Swipe right vs swipe left preserve temporal direction in MHI grid & net_vec
    swipe_r = base.unsqueeze(1).repeat(1, 32, 1)
    swipe_r[0:1] = swipe_r[0:1] + (0.9 * t_grid).view(1, 32, 1)
    swipe_l = base.unsqueeze(1).repeat(1, 32, 1)
    swipe_l[0:1] = swipe_l[0:1] - (0.9 * t_grid).view(1, 32, 1)

    # Circle trajectory (CCW) vs rotated 45 deg
    circle = base.unsqueeze(1).repeat(1, 32, 1)
    ang = 2.0 * math.pi * t_grid
    circle[0:1] = circle[0:1] + (0.35 * torch.cos(ang)).view(1, 32, 1)
    circle[1:2] = circle[1:2] + (0.35 * torch.sin(ang)).view(1, 32, 1)

    # Rotate 45 degrees in XY plane to check rotation-invariant descriptors
    theta = math.radians(45.0)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rot_xy = torch.tensor([[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]])
    circle_rot = torch.einsum("ij,jtn->itn", rot_xy, circle)

    # Static open palm vs closed fist vs asymmetric peace (index+middle extended, others curled)
    open_seq = base.unsqueeze(1).repeat(1, 32, 1)
    fist_seq = _curl_fingers(base, [0, 1, 2, 3, 4], curl=0.95).unsqueeze(1).repeat(1, 32, 1)
    peace_seq = _curl_fingers(base, [0, 3, 4], curl=0.95).unsqueeze(1).repeat(1, 32, 1)

    feats = torch.stack(
        [pre(s)[0] for s in (swipe_r, swipe_l, circle, circle_rot, open_seq, fist_seq, peace_seq)],
        dim=0,
    )
    with torch.no_grad():
        grids, geom = model.shape_branch.extract_shape_features(feats)

    assert grids.shape == (7, 4, 8, 8)
    assert geom.shape == (7, model.shape_branch.GEOM_DIM)
    assert torch.isfinite(grids).all() and torch.isfinite(geom).all()

    # Swipe right vs swipe left: opposite signed MHI gradient (left vs right halves) and opposite dx
    mhi_r = grids[0, 3]
    mhi_l = grids[1, 3]
    assert (mhi_r[:, 4:].mean() - mhi_r[:, :4].mean()) > 0.15
    assert (mhi_l[:, 4:].mean() - mhi_l[:, :4].mean()) < -0.15
    assert geom[0, 13] > 0.2 and geom[1, 13] < -0.2

    # Circle has higher radial ring ratio, hollowness, and winding than straight swipe
    assert geom[2, 15] > geom[0, 15] + 0.15  # radial_ring_ratio
    assert geom[2, 17].abs() > 0.70          # |signed_winding| ~ 1.0 turn
    assert geom[2, 20] > geom[0, 20]         # grid_hollowness

    # Detail 2: Rotation-invariant descriptors stay consistent under 45 deg XY rotation
    rot_invariant_indices = [1, 3, 5, 6, 7, 11, 12, 15, 16, 17, 18]
    diff = (geom[2, rot_invariant_indices] - geom[3, rot_invariant_indices]).abs()
    assert diff.max().item() < 0.08, f"Rotation-invariant drift too high: {diff.tolist()}"

    # Static open palm vs fist vs peace: openness & radial dispersion separate poses cleanly
    assert geom[4, 3] > geom[5, 3] + 0.50  # open palm has much larger fingertip openness than fist
    assert geom[6, 5] > geom[4, 5] + 0.20  # asymmetric peace has much higher fingertip radial dispersion than open palm

    # TorchScript trace + dynamic batch (B=1 & B=2) + checkpoint save/load compatibility
    labels = ["open", "fist", "ok", "swipe_r", "grab", "circle"]
    rec = GestureRecognizer(model, pre, labels, torch.device("cpu"), optimize_inference=True)
    assert rec.compile_for_inference(32) is True
    out = rec.predict_batch({"Left": swipe_r, "Right": circle})
    assert set(out.keys()) == {"Left", "Right"}

    ckpt = rec.save_checkpoint(tmp_path / "shape_ckpt.pt", DataConfig(seq_len=32, use_velocity=True))
    loaded, _ = GestureRecognizer.from_checkpoint(ckpt, torch.device("cpu"))
    assert loaded.model.shape_branch is not None

    # SHREC-22 topology compatibility
    spec_22 = get_skeleton_spec("shrec_22")
    model_22 = UnifiedGestureGCN(cfg=cfg, spec=spec_22, in_channels=6)
    logits_22 = model_22(torch.randn(2, 6, 20, 22), torch.rand(2, 3, 22))
    assert logits_22.shape == (2, 6)
