"""Standalone decoupled Windows/macOS PC controller subscribing to the Gesture Event API (/ws/events).

Runs directly on Windows or macOS Python with zero PyTorch / pyautogui dependencies:
- Windows: native Win32 SendInput via user32.dll
- macOS:   native CoreGraphics CGEventPost via ApplicationServices.framework (+ osascript for audio)

Connects to the local or remote recognition server over WebSocket and executes mapped OS controls
such as PowerPoint / Keynote slide navigation, zoom in/out, and media/volume actions.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Protocol

import websockets
import yaml

# ---------------------------------------------------------------------------
# Windows native SendInput definitions (guarded so macOS/Linux import cleanly)
# ---------------------------------------------------------------------------
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
MOUSEEVENTF_WHEEL = 0x0800
WHEEL_DELTA = 120

EXTENDED_VKS: frozenset[int] = frozenset({0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2D, 0x2E})

VK_MAP: dict[str, int] = {
    "ctrl": 0x11,
    "control": 0x11,
    "shift": 0x10,
    "alt": 0x12,
    "option": 0x12,
    "cmd": 0x5B,
    "command": 0x5B,
    "win": 0x5B,
    "esc": 0x1B,
    "escape": 0x1B,
    "space": 0x20,
    "pageup": 0x21,
    "pagedown": 0x22,
    "end": 0x23,
    "home": 0x24,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    "enter": 0x0D,
    "return": 0x0D,
    "tab": 0x09,
    "plus": 0xBB,
    "equal": 0xBB,
    "minus": 0xBD,
    "f5": 0x74,
    "volume_mute": 0xAD,
    "volume_down": 0xAE,
    "volume_up": 0xAF,
    "media_play_pause": 0xB3,
    **{str(d): 0x30 + d for d in range(10)},
    **{chr(ord("a") + i): 0x41 + i for i in range(26)},
}

if sys.platform == "win32":
    from ctypes import wintypes

    ULONG_PTR = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_uint32

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ULONG_PTR),
        ]

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ULONG_PTR),
        ]

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = [
            ("uMsg", wintypes.DWORD),
            ("wParamL", wintypes.WORD),
            ("wParamH", wintypes.WORD),
        ]

    class _INPUT_UNION(ctypes.Union):
        _fields_ = [
            ("mi", MOUSEINPUT),
            ("ki", KEYBDINPUT),
            ("hi", HARDWAREINPUT),
        ]

    class INPUT(ctypes.Structure):
        _anonymous_ = ("_union",)
        _fields_ = [
            ("type", wintypes.DWORD),
            ("_union", _INPUT_UNION),
        ]


# ---------------------------------------------------------------------------
# macOS native CoreGraphics virtual keycodes & modifier masks (Carbon HIToolbox)
# ---------------------------------------------------------------------------
MAC_CG_HID_EVENT_TAP = 0

MAC_MODIFIER_FLAGS: dict[str, int] = {
    "shift": 1 << 17,     # kCGEventFlagMaskShift
    "ctrl": 1 << 18,      # kCGEventFlagMaskControl
    "control": 1 << 18,
    "alt": 1 << 19,       # kCGEventFlagMaskAlternate
    "option": 1 << 19,
    "cmd": 1 << 20,       # kCGEventFlagMaskCommand
    "command": 1 << 20,
    "meta": 1 << 20,
    "super": 1 << 20,
}

MAC_KEYCODE_MAP: dict[str, int] = {
    "a": 0x00,
    "s": 0x01,
    "d": 0x02,
    "f": 0x03,
    "h": 0x04,
    "g": 0x05,
    "z": 0x06,
    "x": 0x07,
    "c": 0x08,
    "v": 0x09,
    "b": 0x0B,
    "q": 0x0C,
    "w": 0x0D,
    "e": 0x0E,
    "r": 0x0F,
    "y": 0x10,
    "t": 0x11,
    "1": 0x12,
    "2": 0x13,
    "3": 0x14,
    "4": 0x15,
    "6": 0x16,
    "5": 0x17,
    "equal": 0x18,
    "plus": 0x18,
    "9": 0x19,
    "7": 0x1A,
    "minus": 0x1B,
    "8": 0x1C,
    "0": 0x1D,
    "o": 0x1F,
    "u": 0x20,
    "i": 0x22,
    "p": 0x23,
    "enter": 0x24,
    "return": 0x24,
    "l": 0x25,
    "j": 0x26,
    "k": 0x28,
    "n": 0x2D,
    "m": 0x2E,
    "tab": 0x30,
    "space": 0x31,
    "esc": 0x35,
    "escape": 0x35,
    "f5": 0x60,
    "home": 0x73,
    "pageup": 0x74,
    "end": 0x77,
    "pagedown": 0x79,
    "left": 0x7B,
    "right": 0x7C,
    "down": 0x7D,
    "up": 0x7E,
}


class ActionExecutor(Protocol):
    """Common interface for platform-specific OS action executors."""

    def execute(self, rule: dict[str, Any]) -> str: ...


class Win32ActionExecutor:
    """Executes OS-level keyboard & zoom actions natively via Win32 SendInput."""

    def __init__(
        self,
        zoom_method: str = "ctrl_wheel",
        zoom_steps: int = 2,
        dry_run: bool = False,
    ) -> None:
        self.zoom_method = zoom_method
        self.zoom_steps = max(1, int(zoom_steps))
        self.dry_run = dry_run
        self._ppt_showing = False
        self._user32 = (
            ctypes.WinDLL("user32", use_last_error=True) if sys.platform == "win32" else None
        )

    def _send_inputs(self, inputs: list[Any]) -> None:
        if self.dry_run or self._user32 is None or sys.platform != "win32":
            return
        arr = (INPUT * len(inputs))(*inputs)
        self._user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT))

    def _key_event(self, vk: int, key_up: bool = False) -> Any:
        flags = KEYEVENTF_KEYUP if key_up else 0
        if vk in EXTENDED_VKS:
            flags |= KEYEVENTF_EXTENDEDKEY
        return INPUT(
            type=INPUT_KEYBOARD,
            ki=KEYBDINPUT(wVk=vk, wScan=0, dwFlags=flags, time=0, dwExtraInfo=0),
        )

    def press_hotkey(self, *key_names: str) -> None:
        if self.dry_run or self._user32 is None or sys.platform != "win32":
            return
        vks = [VK_MAP[k.lower()] for k in key_names if k.lower() in VK_MAP]
        if not vks:
            return
        events = [self._key_event(vk, key_up=False) for vk in vks]
        events.extend(self._key_event(vk, key_up=True) for vk in reversed(vks))
        self._send_inputs(events)

    def ctrl_wheel(self, notches: int) -> None:
        if self.dry_run or self._user32 is None or sys.platform != "win32":
            return
        ctrl_vk = VK_MAP["ctrl"]
        self._send_inputs([self._key_event(ctrl_vk, key_up=False)])
        time.sleep(0.015)
        try:
            step_delta = WHEEL_DELTA if notches > 0 else -WHEEL_DELTA
            for _ in range(abs(notches)):
                wheel_in = INPUT(
                    type=INPUT_MOUSE,
                    mi=MOUSEINPUT(
                        dx=0,
                        dy=0,
                        mouseData=ctypes.c_uint32(step_delta).value,
                        dwFlags=MOUSEEVENTF_WHEEL,
                        time=0,
                        dwExtraInfo=0,
                    ),
                )
                self._send_inputs([wheel_in])
                time.sleep(0.015)
        finally:
            self._send_inputs([self._key_event(ctrl_vk, key_up=True)])

    def execute(self, rule: dict[str, Any]) -> str:
        action = str(rule.get("action", "")).strip()
        if action == "ppt_next":
            self.press_hotkey("pagedown")
            return "PPT 下一页 (PageDown)"
        if action == "ppt_prev":
            self.press_hotkey("pageup")
            return "PPT 上一页 (PageUp)"
        if action == "ppt_next_arrow":
            self.press_hotkey("right")
            return "PPT 下一页 (Right)"
        if action == "ppt_prev_arrow":
            self.press_hotkey("left")
            return "PPT 上一页 (Left)"
        if action == "ppt_start":
            self._ppt_showing = True
            self.press_hotkey("f5")
            return "开始 PPT 放映 (F5)"
        if action == "ppt_current":
            self._ppt_showing = True
            self.press_hotkey("shift", "f5")
            return "从当前页放映 (Shift+F5)"
        if action == "ppt_stop":
            self._ppt_showing = False
            self.press_hotkey("esc")
            return "退出 PPT 放映 (Esc)"
        if action == "ppt_toggle_show":
            self._ppt_showing = not self._ppt_showing
            if self._ppt_showing:
                self.press_hotkey("shift", "f5")
                return "开始 PPT 放映 (Shift+F5)"
            self.press_hotkey("esc")
            return "退出 PPT 放映 (Esc)"
        if action == "zoom_in":
            if self.zoom_method == "ctrl_plus_minus":
                for _ in range(self.zoom_steps):
                    self.press_hotkey("ctrl", "plus")
                return f"放大 (Ctrl+'+' ×{self.zoom_steps})"
            self.ctrl_wheel(self.zoom_steps)
            return f"放大 (Ctrl+滚轮上 ×{self.zoom_steps})"
        if action == "zoom_out":
            if self.zoom_method == "ctrl_plus_minus":
                for _ in range(self.zoom_steps):
                    self.press_hotkey("ctrl", "minus")
                return f"缩小 (Ctrl+'-' ×{self.zoom_steps})"
            self.ctrl_wheel(-self.zoom_steps)
            return f"缩小 (Ctrl+滚轮下 ×{self.zoom_steps})"
        if action == "zoom_reset":
            self.press_hotkey("ctrl", "0")
            return "重置缩放 (Ctrl+0)"
        if action == "media_play_pause":
            self.press_hotkey("media_play_pause")
            return "播放/暂停媒体"
        if action == "volume_up":
            self.press_hotkey("volume_up")
            return "音量 +"
        if action == "volume_down":
            self.press_hotkey("volume_down")
            return "音量 -"
        if action == "volume_mute":
            self.press_hotkey("volume_mute")
            return "静音切换"
        if action == "key":
            key = str(rule.get("key", ""))
            self.press_hotkey(key)
            return f"按键 ({key})"
        if action == "hotkey":
            keys = [str(k) for k in rule.get("keys", [])]
            self.press_hotkey(*keys)
            return f"组合键 ({'+'.join(keys)})"
        return f"未识别动作: {action}"


class MacActionExecutor:
    """Executes OS-level keyboard, zoom & audio actions natively on macOS via CoreGraphics."""

    def __init__(
        self,
        zoom_method: str = "cmd_plus_minus",
        zoom_steps: int = 2,
        dry_run: bool = False,
    ) -> None:
        self.zoom_method = zoom_method
        self.zoom_steps = max(1, int(zoom_steps))
        self.dry_run = dry_run
        self._ppt_showing = False
        self._app_services: Any = None
        self.accessibility_trusted: bool | None = None

        if sys.platform == "darwin":
            lib = ctypes.CDLL(
                "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
            )
            lib.CGEventCreateKeyboardEvent.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint16,
                ctypes.c_bool,
            ]
            lib.CGEventCreateKeyboardEvent.restype = ctypes.c_void_p
            lib.CGEventSetFlags.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
            lib.CGEventSetFlags.restype = None
            lib.CGEventPost.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
            lib.CGEventPost.restype = None
            lib.CFRelease.argtypes = [ctypes.c_void_p]
            lib.CFRelease.restype = None
            lib.AXIsProcessTrusted.argtypes = []
            lib.AXIsProcessTrusted.restype = ctypes.c_bool
            self._app_services = lib
            self.accessibility_trusted = bool(lib.AXIsProcessTrusted())

    def _post_key(self, keycode: int, flags: int = 0) -> None:
        if self.dry_run or self._app_services is None:
            return
        lib = self._app_services
        ev_down = lib.CGEventCreateKeyboardEvent(None, keycode, True)
        if ev_down:
            if flags:
                lib.CGEventSetFlags(ev_down, flags)
            lib.CGEventPost(MAC_CG_HID_EVENT_TAP, ev_down)
            lib.CFRelease(ev_down)
        time.sleep(0.01)
        ev_up = lib.CGEventCreateKeyboardEvent(None, keycode, False)
        if ev_up:
            if flags:
                lib.CGEventSetFlags(ev_up, flags)
            lib.CGEventPost(MAC_CG_HID_EVENT_TAP, ev_up)
            lib.CFRelease(ev_up)

    def press_hotkey(self, *key_names: str) -> None:
        flags = 0
        keycodes: list[int] = []
        for raw_key in key_names:
            k = raw_key.lower().strip()
            if k in MAC_MODIFIER_FLAGS:
                flags |= MAC_MODIFIER_FLAGS[k]
            elif k in MAC_KEYCODE_MAP:
                keycodes.append(MAC_KEYCODE_MAP[k])
        for kc in keycodes:
            self._post_key(kc, flags)

    def _run_osascript(self, script: str) -> None:
        if self.dry_run or sys.platform != "darwin":
            return
        try:
            subprocess.run(
                ["osascript", "-e", script],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            )
        except Exception:
            pass

    def execute(self, rule: dict[str, Any]) -> str:
        action = str(rule.get("action", "")).strip()
        if action == "ppt_next":
            self.press_hotkey("pagedown")
            return "PPT/Keynote 下一页 (PageDown)"
        if action == "ppt_prev":
            self.press_hotkey("pageup")
            return "PPT/Keynote 上一页 (PageUp)"
        if action == "ppt_next_arrow":
            self.press_hotkey("right")
            return "PPT/Keynote 下一页 (Right)"
        if action == "ppt_prev_arrow":
            self.press_hotkey("left")
            return "PPT/Keynote 上一页 (Left)"
        if action == "ppt_start":
            self._ppt_showing = True
            self.press_hotkey("cmd", "shift", "return")
            return "开始 PPT 放映 (Cmd+Shift+Return)"
        if action == "ppt_current":
            self._ppt_showing = True
            self.press_hotkey("cmd", "return")
            return "从当前页放映 (Cmd+Return)"
        if action == "ppt_stop":
            self._ppt_showing = False
            self.press_hotkey("esc")
            return "退出放映 (Esc)"
        if action == "ppt_toggle_show":
            self._ppt_showing = not self._ppt_showing
            if self._ppt_showing:
                self.press_hotkey("cmd", "return")
                return "开始 PPT 放映 (Cmd+Return)"
            self.press_hotkey("esc")
            return "退出放映 (Esc)"
        if action == "zoom_in":
            for _ in range(self.zoom_steps):
                self.press_hotkey("cmd", "plus")
                time.sleep(0.015)
            return f"放大 (Cmd+'+' ×{self.zoom_steps})"
        if action == "zoom_out":
            for _ in range(self.zoom_steps):
                self.press_hotkey("cmd", "minus")
                time.sleep(0.015)
            return f"缩小 (Cmd+'-' ×{self.zoom_steps})"
        if action == "zoom_reset":
            self.press_hotkey("cmd", "0")
            return "重置缩放 (Cmd+0)"
        if action == "media_play_pause":
            self.press_hotkey("space")
            return "播放/暂停 (Space)"
        if action == "volume_up":
            self._run_osascript(
                "set volume output volume ((output volume of (get volume settings)) + 6)"
            )
            return "音量 +"
        if action == "volume_down":
            self._run_osascript(
                "set volume output volume ((output volume of (get volume settings)) - 6)"
            )
            return "音量 -"
        if action == "volume_mute":
            self._run_osascript(
                "set volume output muted (not (output muted of (get volume settings)))"
            )
            return "静音切换"
        if action == "key":
            key = str(rule.get("key", ""))
            self.press_hotkey(key)
            return f"按键 ({key})"
        if action == "hotkey":
            keys = [str(k) for k in rule.get("keys", [])]
            self.press_hotkey(*keys)
            return f"组合键 ({'+'.join(keys)})"
        return f"未识别动作: {action}"


def create_executor(
    zoom_method: str = "ctrl_wheel",
    zoom_steps: int = 2,
    dry_run: bool = False,
    platform_name: str | None = None,
) -> ActionExecutor:
    """Create the appropriate OS action executor for the current (or requested) platform."""
    plat = platform_name or sys.platform
    if plat == "darwin":
        return MacActionExecutor(zoom_method=zoom_method, zoom_steps=zoom_steps, dry_run=dry_run)
    return Win32ActionExecutor(zoom_method=zoom_method, zoom_steps=zoom_steps, dry_run=dry_run)


def load_controller_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def match_rule(mappings: dict[str, Any], hand: str, gesture: str) -> dict[str, Any] | None:
    rule = mappings.get(gesture)
    if not isinstance(rule, dict):
        return None
    req_hand = str(rule.get("hand", "Any"))
    if req_hand != "Any" and req_hand != hand:
        return None
    return rule


async def run_controller(
    config_path: Path, ws_url_override: str | None, dry_run_override: bool | None
) -> None:
    cfg = load_controller_config(config_path)
    ws_url = ws_url_override or str(cfg.get("ws_url", "ws://127.0.0.1:8000/ws/events"))
    dry_run = cfg.get("dry_run", False) if dry_run_override is None else dry_run_override
    mappings: dict[str, Any] = cfg.get("mappings", {})

    executor = create_executor(
        zoom_method=str(cfg.get("zoom_method", "ctrl_wheel")),
        zoom_steps=int(cfg.get("zoom_steps", 2)),
        dry_run=bool(dry_run),
    )

    platform_label = {"darwin": "macOS (CoreGraphics)", "win32": "Windows (Win32 SendInput)"}.get(
        sys.platform, f"{sys.platform} (仅日志/无本地输入后端)"
    )

    print("============================================================")
    print("  Unified Gesture GCN - 电脑操控客户端 (API 解耦模式)")
    print(f"  运行平台 : {platform_label}")
    print(f"  订阅地址 : {ws_url}")
    print(f"  配置文件 : {config_path}")
    print(f"  运行模式 : {'[模拟测试 Dry-Run，不发送真实按键]' if dry_run else '[实时操控模式]'}")
    if (
        isinstance(executor, MacActionExecutor)
        and executor.accessibility_trusted is False
        and not dry_run
    ):
        print("  [警告] 尚未授予 macOS「辅助功能」权限，模拟按键可能无法生效！")
        print("         请前往：系统设置 -> 隐私与安全性 -> 辅助功能，勾选当前终端/IDE 后重启。")
    print("  已加载规则:")
    for g, r in mappings.items():
        print(f"    - {g:22s} ({r.get('hand', 'Any'):5s}) => {r.get('description', r.get('action'))}")
    print("  按 Ctrl+C 退出。")
    print("============================================================")

    connect_kwargs: dict[str, Any] = {"ping_interval": 20, "ping_timeout": 20}
    # websockets >= 15 defaults to proxy=True (reads system proxy like Clash/v2ray),
    # which breaks local 127.0.0.1 WebSocket connections with "did not receive a valid HTTP response".
    try:
        import inspect

        if "proxy" in inspect.signature(websockets.connect).parameters:
            connect_kwargs["proxy"] = None
    except Exception:
        pass

    while True:
        try:
            async with websockets.connect(ws_url, **connect_kwargs) as ws:
                print(f"[已连接] 正在监听手势事件流 {ws_url} ...")
                async for raw in ws:
                    msg = json.loads(raw)
                    ev_type = msg.get("event")
                    if ev_type == "connected":
                        labels = msg.get("labels", [])
                        print(f"[握手成功] 当前模型已训练类别 ({len(labels)}): {', '.join(labels)}")
                        continue
                    if ev_type == "config_updated":
                        c = msg.get("config", {})
                        print(
                            f"[配置同步] enabled={c.get('enabled')} "
                            f"require_left_lock={c.get('require_left_lock')} "
                            f"cooldown={c.get('cooldown_sec')}s"
                        )
                        continue
                    if ev_type != "gesture_triggered":
                        continue

                    hand = str(msg.get("hand", "Right"))
                    gesture = str(msg.get("gesture", ""))
                    conf = float(msg.get("confidence", 0.0))
                    rule = match_rule(mappings, hand, gesture)
                    if rule is None:
                        print(f"[忽略] {hand} · {gesture} ({conf * 100:.0f}%) — 未配置映射动作")
                        continue

                    action_desc = executor.execute(rule)
                    prefix = "[模拟执行]" if dry_run else "[已执行]"
                    print(f"{prefix} {hand} · {gesture} ({conf * 100:.0f}%) => {action_desc}")
        except (OSError, websockets.WebSocketException) as exc:
            print(f"[连接断开] {exc}，2 秒后自动重连...")
            await asyncio.sleep(2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decoupled Windows/macOS PC Controller via Gesture Event API."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "pc_control.yaml",
        help="Path to pc_control.yaml",
    )
    parser.add_argument("--ws-url", type=str, default=None, help="Override WebSocket event URL.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only log matched actions without sending real keyboard/mouse inputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(
            run_controller(
                config_path=args.config,
                ws_url_override=args.ws_url,
                dry_run_override=True if args.dry_run else None,
            )
        )
    except KeyboardInterrupt:
        print("\n已停止电脑操控客户端。")


if __name__ == "__main__":
    main()
