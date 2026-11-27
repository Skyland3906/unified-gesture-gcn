"""Download MediaPipe static assets locally for zero-CDN Web Studio operation.

Downloads:
  1. @mediapipe/tasks-vision (modern HandLandmarker + WebGL/WASM runtime + hand_landmarker.task)
  2. @mediapipe/hands (legacy fallback runtime)
"""

from __future__ import annotations

import io
import json
import tarfile
import urllib.request
from pathlib import Path

LEGACY_HANDS_URLS: tuple[str, ...] = (
    "https://registry.npmmirror.com/@mediapipe/hands/-/hands-0.4.1675469240.tgz",
    "https://registry.npmjs.org/@mediapipe/hands/-/hands-0.4.1675469240.tgz",
)

TASKS_VISION_META_URLS: tuple[str, ...] = (
    "https://registry.npmmirror.com/@mediapipe/tasks-vision/latest",
    "https://registry.npmjs.org/@mediapipe/tasks-vision/latest",
)

HAND_LANDMARKER_MODEL_URLS: tuple[str, ...] = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
)

FACE_DETECTOR_MODEL_URLS: tuple[str, ...] = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
    "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
)


def _fetch_bytes(urls: tuple[str, ...], desc: str, timeout: int = 90) -> bytes:
    last_err: Exception | None = None
    for url in urls:
        try:
            print(f"Downloading {desc} from {url} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "gesture-gcn-setup/0.2"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:
            last_err = exc
            print(f"  Failed ({exc}), trying next mirror...")
    raise RuntimeError(f"Failed to download {desc}: {last_err}")


def _extract_tgz(raw_bytes: bytes, target_dir: Path, keep_prefixes: tuple[str, ...] | None = None) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw_bytes), mode="r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if len(parts) <= 1:
                continue
            rel_path = Path(*parts[1:])
            rel_posix = rel_path.as_posix()
            if keep_prefixes is not None and not any(
                rel_posix == p or rel_posix.startswith(p) for p in keep_prefixes
            ):
                continue
            dest = target_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            extracted = tf.extractfile(member)
            if extracted is not None:
                dest.write_bytes(extracted.read())


def download_mediapipe_hands(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    if (target_dir / "hands.js").exists() and (target_dir / "hand_landmark_lite.tflite").exists():
        print(f"Legacy MediaPipe Hands assets already present in {target_dir}")
        return

    raw_bytes = _fetch_bytes(LEGACY_HANDS_URLS, "Legacy MediaPipe Hands")
    _extract_tgz(raw_bytes, target_dir)
    print(f"Successfully extracted Legacy MediaPipe Hands assets to {target_dir}")


def download_mediapipe_tasks_vision(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    bundle_ok = (target_dir / "vision_bundle.mjs").exists()
    wasm_ok = (target_dir / "wasm" / "vision_wasm_internal.wasm").exists()
    model_ok = (target_dir / "hand_landmarker.task").exists()

    if not (bundle_ok and wasm_ok):
        meta_raw = _fetch_bytes(TASKS_VISION_META_URLS, "@mediapipe/tasks-vision metadata", timeout=30)
        meta = json.loads(meta_raw.decode("utf-8"))
        tarball_url = str(meta["dist"]["tarball"])
        tgz_bytes = _fetch_bytes((tarball_url,), f"@mediapipe/tasks-vision@{meta.get('version', 'latest')}")
        _extract_tgz(
            tgz_bytes,
            target_dir,
            keep_prefixes=("vision_bundle.mjs", "wasm/", "package.json"),
        )
        print(f"Successfully extracted @mediapipe/tasks-vision runtime to {target_dir}")
    else:
        print(f"@mediapipe/tasks-vision runtime already present in {target_dir}")

    if not model_ok:
        model_bytes = _fetch_bytes(HAND_LANDMARKER_MODEL_URLS, "hand_landmarker.task model", timeout=120)
        (target_dir / "hand_landmarker.task").write_bytes(model_bytes)
        print(f"Saved hand_landmarker.task ({len(model_bytes) / 1e6:.2f} MB) to {target_dir}")
    else:
        print(f"hand_landmarker.task already present in {target_dir}")

    face_model_path = target_dir / "blaze_face_short_range.tflite"
    if not face_model_path.exists():
        face_bytes = _fetch_bytes(FACE_DETECTOR_MODEL_URLS, "blaze_face_short_range.tflite face guard model", timeout=90)
        face_model_path.write_bytes(face_bytes)
        print(f"Saved blaze_face_short_range.tflite ({len(face_bytes) / 1e3:.1f} KB) to {target_dir}")
    else:
        print(f"blaze_face_short_range.tflite already present in {target_dir}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    mp_root = project_root / "src" / "gesture_gcn" / "web" / "static" / "mediapipe"
    download_mediapipe_tasks_vision(mp_root / "tasks")
    download_mediapipe_hands(mp_root / "hands")


if __name__ == "__main__":
    main()
