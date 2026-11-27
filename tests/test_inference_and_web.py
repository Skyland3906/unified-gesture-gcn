"""Tests for real-time inference buffer, recognizer checkpoint I/O, and FastAPI endpoints."""

from __future__ import annotations

from pathlib import Path

import torch
from fastapi.testclient import TestClient

from gesture_gcn.config import DataConfig, ModelConfig
from gesture_gcn.data.skeleton import get_skeleton_spec
from gesture_gcn.data.transforms import SkeletonPreprocessor
from gesture_gcn.inference.buffer import SlidingWindowBuffer
from gesture_gcn.inference.events import ControlConfig, GestureEventEngine
from gesture_gcn.inference.recognizer import GestureRecognizer, PredictionResult
from gesture_gcn.models.network import UnifiedGestureGCN
from gesture_gcn.web.app import create_app


def test_sliding_window_buffer_and_checkpoint(tmp_path: Path) -> None:
    buf = SlidingWindowBuffer(seq_len=16, num_joints=21)
    for _ in range(5):
        buf.push(torch.randn(21, 3))
    seq = buf.get_sequence()
    assert seq.shape == (3, 16, 21)

    spec = get_skeleton_spec("mediapipe_21")
    data_cfg = DataConfig(seq_len=16)
    model_cfg = ModelConfig(num_classes=4, channels=[32, 32], num_groups=4)
    pre = SkeletonPreprocessor(spec, use_bone_fusion=True)
    model = UnifiedGestureGCN(cfg=model_cfg, spec=spec, in_channels=pre.out_channels)
    labels = ["open", "fist", "swipe", "grab"]

    rec = GestureRecognizer(model, pre, labels, torch.device("cpu"))
    pred = rec.predict(seq)
    assert pred.label in labels
    assert 0.0 <= pred.confidence <= 1.0

    ckpt = rec.save_checkpoint(tmp_path / "test.pt", data_cfg)
    loaded, loaded_data_cfg = GestureRecognizer.from_checkpoint(ckpt, torch.device("cpu"))
    assert loaded.labels == labels
    assert loaded_data_cfg.seq_len == 16


def test_web_app_status_and_websocket(tmp_path: Path) -> None:
    app = create_app(
        checkpoint_path=tmp_path / "ckpt.pt",
        custom_store_path=tmp_path / "custom.json",
    )
    client = TestClient(app)

    res = client.get("/api/status")
    assert res.status_code == 200
    body = res.json()
    assert body["num_joints"] == 21
    assert len(body["labels"]) == 6
    assert body["use_shape_branch"] is False

    # Frontend handedness alignment: HandLandmarker outputs anatomical Left/Right on unmirrored frames,
    # so swapHands must default to unchecked and legacy @mediapipe/hands fallback must normalize labels.
    index_res = client.get("/")
    assert index_res.status_code == 200
    html = index_res.text
    assert '<input type="checkbox" id="swapHands" />' in html
    assert 'id="swapHands" checked' not in html
    assert 'label: h && h.label === "Left" ? "Right" : "Left"' in html
    assert '<input type="checkbox" id="antiFace" checked />' in html
    assert "blaze_face_short_range.tflite" in html
    assert 'id="useShapeBranch"' in html
    assert "启用相对形状分支检测" in html
    assert 'id="trainProgressWrap"' in html
    assert "/api/train/progress" in html

    with client.websocket_connect("/ws/predict") as ws:
        frame = [[0.5, 0.5, 0.0] for _ in range(21)]
        ws.send_json({"landmarks": frame})
        out = ws.receive_json()
        assert "label" in out
        assert "motion_energy" in out
        assert "latency_ms" in out

        # Batched two-hand inference protocol
        ws.send_json({"hands": {"Left": frame, "Right": frame}})
        batched = ws.receive_json()
        assert "results" in batched
        assert set(batched["results"].keys()) == {"Left", "Right"}
        assert "label" in batched["results"]["Left"]
        assert "label" in batched["results"]["Right"]

    # Gesture catalog & control config APIs
    gestures_res = client.get("/api/gestures")
    assert gestures_res.status_code == 200
    preset_ids = {g["id"] for g in gestures_res.json()["presets"]}
    assert {"static_peace", "dynamic_swipe_up", "dynamic_swipe_down", "dynamic_push", "dynamic_pull"}.issubset(
        preset_ids
    )

    cfg_res = client.post(
        "/api/control/config",
        json={
            "enabled": True,
            "confidence_threshold": 0.30,
            "hold_frames": 1,
            "cooldown_sec": 0.1,
            "require_left_lock": False,
            "lock_gesture": "static_open_palm",
        },
    )
    assert cfg_res.status_code == 200
    assert cfg_res.json()["hold_frames"] == 1

    # End-to-end decoupled event broadcast: /ws/predict -> /ws/events
    with client.websocket_connect("/ws/events") as ev_ws:
        hello = ev_ws.receive_json()
        assert hello["event"] == "connected"

        with client.websocket_connect("/ws/predict") as pred_ws:
            frame = [[0.5, 0.5, 0.0] for _ in range(21)]
            pred_ws.send_json({"hands": {"Right": frame}})
            pred_out = pred_ws.receive_json()
            assert pred_out["subscribers"] >= 1
            assert "triggered_event" in pred_out["results"]["Right"]

        ev = ev_ws.receive_json()
        assert ev["event"] == "gesture_triggered"
        assert ev["hand"] == "Right"

    # End-to-end training with relative shape branch toggle + live progress endpoint
    clip_a = [[[0.5 + 0.01 * i, 0.5, 0.0] for i in range(21)] for _ in range(8)]
    clip_b = [[[0.4 + 0.02 * t, 0.5, 0.0] for _ in range(21)] for t in range(8)]
    assert client.post("/api/samples", json={"label": "static_open_palm", "frames": clip_a}).status_code == 200
    assert client.post("/api/samples", json={"label": "dynamic_swipe_left", "frames": clip_b}).status_code == 200

    train_res = client.post("/api/train", json={"epochs": 4, "use_shape_branch": True})
    assert train_res.status_code == 200
    train_body = train_res.json()
    assert train_body["use_shape_branch"] is True
    assert set(train_body["labels"]) == {"static_open_palm", "dynamic_swipe_left"}

    prog_res = client.get("/api/train/progress")
    assert prog_res.status_code == 200
    prog_body = prog_res.json()
    assert prog_body["running"] is False
    assert prog_body["stage"] == "done"
    assert prog_body["progress"] == 1.0
    assert prog_body["use_shape_branch"] is True

    status_after = client.get("/api/status").json()
    assert status_after["use_shape_branch"] is True


def test_gesture_event_engine_debounce_cooldown_and_lock() -> None:
    engine = GestureEventEngine(
        ControlConfig(
            enabled=True,
            confidence_threshold=0.75,
            hold_frames=2,
            cooldown_sec=1.0,
            require_left_lock=False,
        )
    )
    push = PredictionResult("dynamic_push", 0.90, {"dynamic_push": 0.90}, 0.15, True, 5.0)
    peace = PredictionResult("static_peace", 0.90, {"static_peace": 0.90}, 0.02, False, 5.0)
    idle = PredictionResult("static_idle", 0.95, {"static_idle": 0.95}, 0.01, False, 5.0)

    # Needs 2 consecutive frames to fire
    assert engine.process("Right", push, now=10.0) is None
    ev1 = engine.process("Right", push, now=10.05)
    assert ev1 is not None and ev1["gesture"] == "dynamic_push"

    # Cooldown suppresses rapid re-trigger
    assert engine.process("Right", push, now=10.20) is None
    assert engine.process("Right", push, now=10.25) is None

    # After cooldown, dynamic gesture can fire again
    assert engine.process("Right", push, now=11.10) is None
    ev2 = engine.process("Right", push, now=11.15)
    assert ev2 is not None and ev2["gesture"] == "dynamic_push"

    # Static gesture latches until pose changes or idle
    assert engine.process("Right", peace, now=12.50) is None
    ev3 = engine.process("Right", peace, now=12.55)
    assert ev3 is not None and ev3["gesture"] == "static_peace"
    assert engine.process("Right", peace, now=15.00) is None
    assert engine.process("Right", peace, now=15.05) is None
    assert engine.process("Right", idle, now=15.10) is None
    assert engine.process("Right", peace, now=15.20) is None
    ev4 = engine.process("Right", peace, now=15.25)
    assert ev4 is not None and ev4["gesture"] == "static_peace"

    # Left-hand safety lock blocks Right until Left holds lock_gesture
    engine.update_config(
        ControlConfig(
            enabled=True,
            confidence_threshold=0.75,
            hold_frames=1,
            cooldown_sec=0.1,
            require_left_lock=True,
            lock_gesture="static_open_palm",
        )
    )
    engine.reset()
    palm = PredictionResult("static_open_palm", 0.92, {"static_open_palm": 0.92}, 0.02, False, 5.0)
    assert engine.process("Right", push, now=20.0) is None
    assert engine.process("Left", palm, now=20.1) is None
    ev5 = engine.process("Right", push, now=20.2)
    assert ev5 is not None and ev5["gesture"] == "dynamic_push"


def test_velocity_preprocessor_and_kinematic_prior() -> None:
    spec = get_skeleton_spec("mediapipe_21")
    pre = SkeletonPreprocessor(spec, use_bone_fusion=True, use_velocity=True)
    assert pre.out_channels == 9

    # Build an open-hand clip moving clearly to screen-right (raw x decreasing)
    seq = torch.zeros(3, 32, 21)
    # Wrist at x=0.7 -> 0.3, y=0.5; middle MCP at (wrist.x, wrist.y - 0.15); fingertips far from wrist
    xs = torch.linspace(0.7, 0.3, 32)
    for t in range(32):
        seq[0, t, :] = xs[t]
        seq[1, t, 0] = 0.5
        seq[1, t, 9] = 0.35
        for tip in (8, 12, 16, 20):
            seq[1, t, tip] = 0.15  # open ratio ~ 0.35 / 0.15 = 2.33

    feat, dr = pre(seq)
    assert feat.shape == (9, 32, 21)

    labels = [
        "dynamic_swipe_left",
        "dynamic_swipe_right",
        "dynamic_swipe_up",
        "dynamic_swipe_down",
        "dynamic_grab_close",
        "dynamic_circle_wave",
        "static_open_palm",
    ]
    model_cfg = ModelConfig(num_classes=len(labels), channels=[32, 32], num_groups=4)
    model = UnifiedGestureGCN(cfg=model_cfg, spec=spec, in_channels=9)
    rec = GestureRecognizer(model, pre, labels, torch.device("cpu"))

    uniform = torch.full((len(labels),), 1.0 / len(labels))
    refined = rec._apply_kinematic_prior(seq, uniform, float(dr.mean().item()))
    # Moving screen-right with wide-open fingers must prefer swipe_right over swipe_left, grab_close, and circle_wave
    assert refined[1] > refined[0] * 4.0
    assert refined[1] > refined[4] * 4.0
    assert refined[1] > refined[5] * 4.0

    # Diagonal lift / non-circular transition must suppress dynamic_circle_wave instead of letting it absorb mass
    seq_diag = seq.clone()
    ys_diag = torch.linspace(0.72, 0.35, 32)
    for t in range(32):
        seq_diag[0, t, :] = xs[t]
        seq_diag[1, t, 0] = ys_diag[t]
        seq_diag[1, t, 9] = ys_diag[t] - 0.15
        for tip in (8, 12, 16, 20):
            seq_diag[1, t, tip] = ys_diag[t] - 0.35
    refined_diag = rec._apply_kinematic_prior(seq_diag, uniform, 0.18)
    assert refined_diag[5] < refined_diag[6] * 0.25

    # Genuine full-circle trajectory (360 deg in XY) must NOT penalize dynamic_circle_wave
    import math

    seq_circle = torch.zeros(3, 32, 21)
    angles = torch.linspace(0.0, 2.0 * math.pi, 32)
    for t in range(32):
        cx = 0.5 + 0.14 * math.cos(float(angles[t].item()))
        cy = 0.5 + 0.14 * math.sin(float(angles[t].item()))
        seq_circle[0, t, :] = cx
        seq_circle[1, t, 0] = cy
        seq_circle[1, t, 9] = cy - 0.12
        for tip in (8, 12, 16, 20):
            seq_circle[1, t, tip] = cy - 0.28
    refined_circle = rec._apply_kinematic_prior(seq_circle, uniform, 0.22)
    assert refined_circle[5] > refined_circle[0] * 4.0
    assert refined_circle[5] >= refined_circle[6] * 0.95

    # Return-stroke suppression: immediately after swipe_right, returning left is damped
    rec._swipe_guard["Right"] = ("dynamic_swipe_right", 100.0, 0)
    seq_left = seq.clone()
    seq_left[0] = torch.linspace(0.3, 0.7, 32).view(32, 1)
    refined_ret = rec._apply_kinematic_prior(seq_left, uniform, 0.18, hand="Right", now=100.5)
    assert refined_ret[0] < 0.20

    # Event engine opposite-direction cooldown blocks return-stroke left swipe, clears after brief pause
    ev_engine = GestureEventEngine(
        ControlConfig(
            enabled=True,
            confidence_threshold=0.75,
            hold_frames=1,
            cooldown_sec=0.5,
            opposite_cooldown_sec=1.6,
        )
    )
    sw_r = PredictionResult("dynamic_swipe_right", 0.90, {"dynamic_swipe_right": 0.90}, 0.16, True, 5.0)
    sw_l = PredictionResult("dynamic_swipe_left", 0.90, {"dynamic_swipe_left": 0.90}, 0.16, True, 5.0)
    still = PredictionResult("static_idle", 0.90, {"static_idle": 0.90}, 0.01, False, 5.0)
    assert ev_engine.process("Right", sw_r, now=50.0) is not None
    # At 50.8s (> cooldown_sec 0.5s, < opposite_cooldown_sec 1.6s), opposite return stroke is blocked
    assert ev_engine.process("Right", sw_l, now=50.8) is None
    # After 6 stationary frames (brief pause), intentional reverse swipe works immediately
    for i in range(6):
        ev_engine.process("Right", still, now=50.85 + 0.04 * i)
    ev_rev = ev_engine.process("Right", sw_l, now=51.2)
    assert ev_rev is not None and ev_rev["gesture"] == "dynamic_swipe_left"


def test_mps_device_config_and_cross_platform_controller() -> None:
    import importlib.util

    from gesture_gcn.config import ExperimentConfig
    from gesture_gcn.engine.trainer import resolve_device

    cfg = ExperimentConfig(device="mps")
    assert cfg.device == "mps"
    assert resolve_device("mps").type == "mps"
    assert resolve_device("cpu").type == "cpu"

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "pc_controller.py"
    spec = importlib.util.spec_from_file_location("pc_controller", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mac_exec = mod.create_executor(platform_name="darwin", zoom_steps=2, dry_run=True)
    assert isinstance(mac_exec, mod.MacActionExecutor)
    assert "PageDown" in mac_exec.execute({"action": "ppt_next"})
    assert "Cmd+Return" in mac_exec.execute({"action": "ppt_toggle_show"})
    assert "Esc" in mac_exec.execute({"action": "ppt_toggle_show"})
    assert "Cmd+'+'" in mac_exec.execute({"action": "zoom_in"})
    assert "Cmd+'-'" in mac_exec.execute({"action": "zoom_out"})
    assert "Cmd+0" in mac_exec.execute({"action": "zoom_reset"})
    assert "音量 +" in mac_exec.execute({"action": "volume_up"})
    assert "cmd+alt+p" in mac_exec.execute({"action": "hotkey", "keys": ["cmd", "alt", "p"]})

    win_exec = mod.create_executor(platform_name="win32", zoom_steps=2, dry_run=True)
    assert isinstance(win_exec, mod.Win32ActionExecutor)
    assert "Shift+F5" in win_exec.execute({"action": "ppt_toggle_show"})
    assert "Esc" in win_exec.execute({"action": "ppt_toggle_show"})
    assert "Ctrl+滚轮上" in win_exec.execute({"action": "zoom_in"})
