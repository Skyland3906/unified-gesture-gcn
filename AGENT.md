# AGENT.md — AI Agent 自动化部署、验收与运维指南

本文档面向负责部署、调试或二次开发本项目的 **AI Agent** 与工程师，提供在 **macOS / Windows / Linux / Docker** 上开箱即用的完整操作规范与排错清单。

---

## 1. 项目概览

本项目（`gesture-gcn`）是一个面向 **静态 + 动态统一骨架手势识别** 的轻量级 PyTorch 框架（~0.6M 参数），包含三大解耦模块：

1. **核心模型与训练引擎**（`src/gesture_gcn/`）：融合 DA-GCN（ARU/DRU/CTF）、DSTSA-GCN（分组静态拓扑 + MS-TCN）与 DRE（动态范围嵌入）及 Joint+Bone+Velocity 特征预处理。
2. **实时摄像头 Web Studio**（`scripts/run_web.py` + `src/gesture_gcn/web/`）：FastAPI + WebSocket 后端，前端通过浏览器原生摄像头 + 本地化新版 `@mediapipe/tasks-vision` `HandLandmarker`（GPU 加速，自动回退旧版 `@mediapipe/hands`）与暗部自适应提亮增强提取 21 点 3D 手部骨架，支持双手实时推理、在线录制自定义手势并一键微调热更新模型，以及运动学先验与回手抑制。
3. **跨平台手势控电脑客户端**（`scripts/pc_controller.py` + `configs/pc_control.yaml`）：零 PyTorch 依赖，订阅 `ws://127.0.0.1:8000/ws/events` 事件流，原生调用操作系统输入接口控制 PPT / Keynote 翻页、缩放、放映开关与系统音量：
   - **Windows**：原生 Win32 `user32.SendInput`
   - **macOS**：原生 CoreGraphics `CGEventPost`（`ApplicationServices.framework`，兼容 Intel & Apple Silicon ARM64）+ `osascript`

---

## 2. 环境前置要求

- **Python 版本**：强烈推荐 **Python 3.10 – 3.12**（PyTorch 官方 wheel 支持最稳定；若宿主机默认是 Python 3.13/3.14 且安装 `torch` 失败，请优先用 `conda` / `uv` / `pyenv` 创建 Python 3.11 环境）。
- **算力后端**（`configs/default.yaml` 默认 `device: "auto"`）：
  - NVIDIA GPU -> 自动启用 `cuda`
  - Apple Silicon（M1 / M2 / M3 / M4）-> 自动启用 `mps`
  - 无 GPU -> 自动回退 `cpu`（模型仅 ~0.6M 参数，CPU 实时推理与在线训练同样流畅）
- **网络需求**：首次部署需下载 PyTorch、Python 依赖及本地 MediaPipe Hands 静态资源（约 24MB，已内置国内 `npmmirror` 与 `npmjs` 双镜像回退）；下载完成后可完全离线运行。
- **默认端口**：`8000`（本地）或 `$PORT` / `7860`（Docker）。

---

## 3. 分平台一键部署流程

Agent 在项目根目录执行对应平台的步骤：

### 3.1 macOS（Apple Silicon / Intel）

```bash
# 1. 创建并激活虚拟环境（推荐 Python 3.11 / 3.12）
python3 -m venv .venv
source .venv/bin/activate

# 2. 安装 PyTorch（注意：macOS 直接装官方 PyPI 版，不要加 --index-url .../whl/cpu）
pip install --upgrade pip
pip install torch

# 3. 安装本项目及 Web、控制器、测试依赖
pip install -e ".[web,controller,dev]"

# 4. 下载本地 MediaPipe Hands 静态资源（避免前端依赖外网 CDN）
python scripts/download_assets.py
```

### 3.2 Windows（PowerShell 原生部署）

```powershell
# 1. 创建并激活虚拟环境（推荐 Python 3.11 / 3.12）
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. 安装 PyTorch（CPU 版示例；若有 NVIDIA GPU 请按官网命令安装对应 CUDA 版）
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. 安装本项目及全部依赖
pip install -e ".[web,controller,dev]"

# 4. 下载本地 MediaPipe Hands 静态资源
python scripts/download_assets.py
```

> **Windows + WSL2 混合模式（可选）**：若 PyTorch 虚拟环境装在 WSL2 的 `.venv/bin/python` 中，[start_web.bat](start_web.bat) 会自动检测并拉起 WSL 后端，而手势控电脑客户端（`scripts/pc_controller.py`）只需在 Windows 原生 Python 安装 `pip install websockets pyyaml` 即可跨 WSL 订阅 `ws://127.0.0.1:8000/ws/events` 控制 Windows 桌面。

### 3.3 Linux / Docker

```bash
# 本地 Linux
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # 有 GPU 则装 CUDA 版
pip install -e ".[web,controller,dev]"
python scripts/download_assets.py

# 或直接构建并运行 Docker 容器（监听 7860 端口）
docker build -t unified-gesture-gcn .
docker run -p 7860:7860 unified-gesture-gcn
```

---

## 4. Agent 自动化验收清单（Verification Checklist）

完成安装后，Agent **必须**按顺序执行以下验收检查：

1. **运行全量单元测试**：
   ```bash
   pytest -v
   ```
   - **预期**：全部测试通过（当前共 14 项），无 `NaN` 梯度，涵盖骨架变换、消融前反向传播、端到端训练收敛、Web/WebSocket 接口、事件防抖/冷却/回手抑制、MPS 设备配置及 Windows/macOS 控制器逻辑。

2. **运行合成基准 Demo（可选验证训练与延迟）**：
   ```bash
   python scripts/run_demo.py --config configs/default.yaml
   ```
   - **预期**：正常打印模型参数量（约 `0.5M – 0.8M`）、12 个 Epoch 内验证准确率达到 `95% – 100%`，并输出单样本推理延迟（ms / FPS）。

3. **验证本地 MediaPipe 资源就位**：
   确认以下文件存在：
   - `src/gesture_gcn/web/static/mediapipe/tasks/vision_bundle.mjs`
   - `src/gesture_gcn/web/static/mediapipe/tasks/wasm/vision_wasm_internal.wasm`
   - `src/gesture_gcn/web/static/mediapipe/tasks/hand_landmarker.task`
   - `src/gesture_gcn/web/static/mediapipe/tasks/blaze_face_short_range.tflite`（防人脸误检）
   - `src/gesture_gcn/web/static/mediapipe/hands/hands.js`（旧版回退）
   - `src/gesture_gcn/web/static/mediapipe/hands/hand_landmark_lite.tflite`（旧版回退）

4. **验证 Web 服务健康接口**：
   启动服务后请求 `GET http://127.0.0.1:8000/api/status`：
   - **预期**：HTTP `200`，JSON 包含 `device`、`num_joints: 21`、`labels`、`control` 字段。
   - **注意**：首次启动若 `checkpoints/unified_gesture_gcn.pt` 不存在，后端会自动在约 10–30 秒内用合成基准训练并保存初始权重，请耐心等待后再判定超时。

5. **验证手势控电脑客户端连通性（Dry-Run）**：
   ```bash
   python scripts/pc_controller.py --dry-run
   ```
   - **预期**：打印 `[已连接]` 与 `[握手成功] 当前模型已训练类别 (...)`，无连接错误。

---

## 5. 启动服务指南

| 平台 | 启动 Web Studio（端口 8000） | 启动手势控电脑客户端 |
| :--- | :--- | :--- |
| **macOS / Linux** | `bash start_web.sh`（或 `python scripts/run_web.py --host 127.0.0.1 --port 8000`） | `bash start_pc_controller.sh` |
| **Windows** | 双击 `start_web.bat`（或 `python scripts/run_web.py --host 127.0.0.1 --port 8000`） | 双击 `start_pc_controller.bat` |

启动后在浏览器访问 **`http://localhost:8000`**（推荐使用 Chrome / Edge）。

---

## 6. 核心目录与接口速查

### 6.1 关键文件

| 路径 | 职责 |
| :--- | :--- |
| `configs/default.yaml` | 模型结构、序列长度、速度特征、设备（`auto`/`cpu`/`cuda`/`mps`）与训练超参 |
| `configs/pc_control.yaml` | 手势 -> 操作系统动作映射（翻页、缩放、放映开关、音量、自定义组合键） |
| `scripts/download_assets.py` | 下载本地 MediaPipe Hands 静态资源至 `src/gesture_gcn/web/static/mediapipe/hands/` |
| `scripts/run_web.py` | 启动 FastAPI + WebSocket 实时手势识别 Web Studio |
| `scripts/pc_controller.py` | 跨平台（Windows/macOS）手势控电脑客户端 |
| `src/gesture_gcn/web/app.py` | Web 路由、在线采样训练与 WebSocket 广播服务 |
| `src/gesture_gcn/inference/recognizer.py` | 模型推理、TorchScript 加速、自适应 EMA 与运动学先验/回手抑制 |
| `src/gesture_gcn/inference/events.py` | 预设手势词表、连续帧防抖、冷却、反向冷却、左手安全锁与事件广播器 |

> **默认预训练权重**：`checkpoints/unified_gesture_gcn.pt` 已纳入仓库以便跨平台开箱即用；其余临时文件与个人录制数据（`data/`、`src/gesture_gcn/web/static/mediapipe/`、`.venv/`）已写入 `.gitignore`，不要提交到 Git。

### 6.2 HTTP / WebSocket API

- `GET /api/status`：查询运行设备、模型参数量、当前生效标签与控制状态。
- `GET /api/gestures`：查询预设手势目录、当前模型类别与已录制样本数。
- `GET / POST /api/control/config`：读取或热更新事件触发阈值（置信度、保持帧数、冷却时间、左手安全锁）。
- `POST /api/samples` & `DELETE /api/samples[/{label}]`：录制或清理真实手势片段。
- `POST /api/train`：使用已录制样本在线微调并热更新推理权重。
- `WS /ws/predict`：前端发送双手骨架坐标 `{"hands": {"Left": [...], "Right": [...]}}`，实时返回预测结果。
- `WS /ws/events`：外部控制器订阅接口，推送 `connected`、`config_updated`、`gesture_triggered` JSON 事件。

---

## 7. 常见踩坑与 Agent 排错指南

1. **macOS 下手势触发日志正常，但 PPT / 浏览器无响应**
   - **原因**：macOS 拦截未授权进程的 `CGEventPost` 模拟按键。
   - **处理**：提示用户打开 **系统设置 -> 隐私与安全性 -> 辅助功能**，勾选当前运行的终端应用（Terminal / iTerm2 / VS Code / Cursor 等）后**完全重启终端**再运行 `bash start_pc_controller.sh`。
2. **macOS 下若使用 Keynote 放映**
   - 默认 `ppt_toggle_show` 发送 `Cmd+Return`（PowerPoint for Mac 快捷键）。如需控制 Keynote 开始放映，可在 `configs/pc_control.yaml` 中将动作改为 `action: "hotkey"`、`keys: ["cmd", "alt", "p"]`。
3. **浏览器报错无法打开摄像头（`navigator.mediaDevices is undefined`）**
   - **原因**：浏览器仅允许在安全上下文调用摄像头。
   - **处理**：必须通过 `http://localhost:8000` 或 `http://127.0.0.1:8000` 访问；若需跨机器访问，必须配置 HTTPS（或部署到 Hugging Face Spaces / Render）。
4. **开启 Clash / v2ray 等系统代理后，`pc_controller.py` 连接本地 WebSocket 失败**
   - `scripts/pc_controller.py` 已针对 `websockets >= 15` 自动传入 `proxy=None` 绕过本地代理；若用户自行写客户端，请确保跳过 `127.0.0.1` 代理。
5. **Apple Silicon 上若出现 MPS 算子警告或推理波动**
   - 模型非常轻量，若在特定 PyTorch 版本下希望切换回纯 CPU，只需将 `configs/default.yaml` 中的 `device: "auto"` 改为 `device: "cpu"` 即可。
6. **在 Windows 编辑 `.sh` 脚本后传到 Mac 报 `\r: command not found`**
   - 确保 `start_web.sh` 与 `start_pc_controller.sh` 使用 Unix `LF` 换行符（仓库中已保持 `LF`）。

---

## 8. 开发协作与 Git 提交规范

- **每次完成功能改动或 Bug 修复并验收通过后，必须使用中文编写清晰的 `git commit` 提交信息并提交到本地仓库**（不要把未验收或调试临时文件混入提交）。
- 推荐提交格式：`<类型>: <中文改动核心说明>`，例如：
  - `feat: 升级 HandLandmarker 并增加暗部自适应提亮增强`
  - `fix: 修复单手画圈时左右手身份跳变问题`
  - `perf: 合并双手 Batch=2 推理并启用 TorchScript 加速`
- 如改动涉及多个文件或关键参数调整，建议在 commit 正文中补充要点列表，方便后续回溯；仅在用户明确要求时再执行 `git push`。
