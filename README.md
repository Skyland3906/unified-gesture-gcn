# Unified Static & Dynamic Hand Gesture GCN (Plan A)

面向**静态 + 动态统一骨架手势识别**的轻量级 PyTorch 工程框架与实时摄像头 Web Studio，融合三篇前沿论文的核心设计：

1. **DA-GCN (2025)**：4 层轻量骨干 + **ARU（注意力拓扑）** + **DRU（差分拓扑）** + 通道/时间双流融合（CTF）。
2. **DSTSA-GCN (2025)**：**分组可学习静态拓扑**（消除局部偏置，强化静态指尖远距离交互）+ **MS-TCN**（多尺度膨胀卷积 + Shortcut，兼容静/动态不同速度）。
3. **Doodle to Detect / SKETCH (NeurIPS)**：安全窗口归一化 + **DRE（Dynamic Range Embedding，动态范围嵌入）**，显式保留窗口运动幅度以区分静态与动态手势，并结合 **Joint + Bone Early Fusion**。

---

## 目录结构

```text
├── .github/workflows/ci.yml          # GitHub Actions 自动化测试
├── Dockerfile                        # 一键云端部署（兼容 Hugging Face Spaces / Render）
├── configs/default.yaml              # 声明式实验配置（支持一键消融实验）
├── pyproject.toml                    # 现代 Python 依赖与 pytest/ruff 配置
├── scripts/
│   ├── download_assets.py            # 下载 MediaPipe Hands 本地静态资源（零外网 CDN 依赖）
│   ├── run_demo.py                   # CLI 合成静+动态基准训练与推理测速
│   └── run_web.py                    # 启动实时摄像头手势识别 & 在线录制训练 Web Studio
├── src/gesture_gcn/
│   ├── config.py                     # Pydantic 强类型配置校验
│   ├── data/                         # 骨架拓扑注册表、安全窗口归一化、DRE 提取、合成基准
│   ├── models/                       # ARU/DRU/Grouped-TAM、DualStreamDAGC、DRE、MS-TCN、UnifiedGestureGCN
│   ├── engine/                       # 训练、评估与延迟测速引擎
│   ├── inference/                    # 滑动窗口缓存、Checkpoint 推理器、真实样本录制与在线训练
│   └── web/                          # FastAPI + WebSocket 实时推理与交互式前端
└── tests/                            # 单元测试、消融梯度检查与端到端集成测试
```

---

## 快速开始（本地运行：Windows / macOS / Linux）

推荐使用 Python 3.10–3.12：

```bash
# 1. 创建虚拟环境并安装 PyTorch
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# macOS（Intel & Apple Silicon M1–M4，自动支持 MPS 加速）：
pip install torch
# Windows / Linux（CPU 版示例；如有 NVIDIA GPU 可装对应 CUDA 版）：
# pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. 安装本项目及 Web/测试依赖
pip install -e ".[web,dev]"

# 3. 下载本地 MediaPipe Hands 静态资源（约 24MB，避免国内访问 CDN 超时）
python scripts/download_assets.py

# 4. 运行全部自动化测试
pytest -v

# 5. 启动实时摄像头手势识别 Web Studio
python scripts/run_web.py --host 127.0.0.1 --port 8000
# 或一键启动：
#   macOS / Linux: bash start_web.sh
#   Windows:       双击 start_web.bat
```

### 手势操控电脑客户端（Windows / macOS 跨平台）

在 Web Studio 运行后，可在新终端启动独立的手势控电脑客户端（默认支持 PPT/Keynote 翻页、缩放、放映开关等，规则见 `configs/pc_control.yaml`）：

```bash
# macOS / Linux:
bash start_pc_controller.sh
# Windows:
# 双击 start_pc_controller.bat 或运行 python scripts/pc_controller.py
```

> **macOS 权限提示**：首次运行手势控电脑客户端时，需要前往 **系统设置 -> 隐私与安全性 -> 辅助功能**，为当前使用的终端（如 Terminal / iTerm2 / VS Code）开启权限后重启终端，CoreGraphics 模拟按键才会生效。可先用 `python scripts/pc_controller.py --dry-run` 验证事件连通性。

浏览器打开 **`http://localhost:8000`** 即可：
- 实时 21 点 3D 手部骨架追踪 + `UnifiedGestureGCN` 推理 + DRE 动/静态能量指示；
- 自动跳过虚拟摄像头，并支持下拉框实时切换物理摄像头；
- 网页端直接录制自己的真实静/动态手势样本并一键训练、热更新模型权重。

---

## 云端部署（Docker / Hugging Face Spaces / Render）

项目根目录已内置 [Dockerfile](Dockerfile)，支持一键构建并自带 HTTPS（满足浏览器调用摄像头的安全上下文要求）：

### 本地 Docker 运行
```bash
docker build -t unified-gesture-gcn .
docker run -p 7860:7860 unified-gesture-gcn
# 浏览器打开 http://localhost:7860
```

### 部署到 Hugging Face Spaces（免费公网 HTTPS）
1. 在 [Hugging Face Spaces](https://huggingface.co/new-space) 新建 Space，SDK 选择 **Docker (Blank)**。
2. 将本 GitHub 仓库关联或推送到该 Space，平台会自动按 `Dockerfile` 构建并监听 `7860` 端口。

### 部署到 Render
1. 在 [Render](https://render.com/) 选择 **New Web Service** -> 连接本 GitHub 仓库。
2. Runtime 自动识别为 **Docker**，`Dockerfile` 已自动兼容 Render 注入的 `$PORT` 环境变量。

---

## 论文消融与二次开发

* **消融实验**：复制修改 `configs/default.yaml` 中的 `use_aru`、`use_dru`、`use_dre`、`use_mstcn`、`use_bone_fusion`、`num_groups`，无需改动模型代码。
* **接入真实数据集（SHREC'17 / DHG-14/28 等）**：在 `src/gesture_gcn/data/` 新增 `Dataset` 读取 `(3, T, N)` 序列并传入 `SkeletonPreprocessor(spec)` 即可直接复用训练与推理管线。
