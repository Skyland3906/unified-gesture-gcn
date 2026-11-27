# 项目功能更新日志（CHANGELOG）

本文件按时间倒序（最新在前）记录 `unified-gesture-gcn` 项目的每次功能迭代、Bug 修复、性能优化与文档变更。

> **维护规范**：每次执行 `git commit` 以及上传（`git push`）到 GitHub 前，必须将本次改动的时间与核心内容写入本文件并一并提交（详见 `AGENT.md` 第 8 节）。

---

## 2026-11-27

### 2026-11-27 20:11 | `feat`: 默认启用相对形状分支并更新 12 类真实手势预训练权重
- 将 [configs/default.yaml](configs/default.yaml) 中的 `use_shape_branch` 调整为默认 `true`（保留 `ModelConfig` 默认回退 `false` 以确保历史旧权重向后兼容），同步更新 [tests/test_inference_and_web.py](tests/test_inference_and_web.py) 对应默认状态断言。
- 基于 12 类真实录制手势样本启用 `RelativeShapeBranch` 重新训练（22 轮，训练集准确率 `100.00%`，参数量 `0.332M`）并更新默认权重 [checkpoints/unified_gesture_gcn.pt](checkpoints/unified_gesture_gcn.pt)（约 `1.34 MB`），启动 Web Studio 后默认即开启「启用相对形状分支检测」。

### 2026-11-27 20:01 | `feat`: 新增手势区域相对形状分支、网页端开关与实时训练可视进度条
- 新增 [src/gesture_gcn/models/shape.py](src/gesture_gcn/models/shape.py) 实现 `RelativeShapeBranch`：基于窗口归一化骨架提取 4 通道局部 `8×8` 高斯软光栅网格（手部轮廓占据、指尖轮廓、扫掠轨迹密度、带时间先后的 MHI 运动历史网格）与 21 维相对几何描述子（兼顾轴对齐宽高比/有向位移与旋转不变 2D PCA 细长度、刚性掌骨尺度归一化指尖展开度与分散度、环形空心度及有向绕转角），通过样本自适应残差门控融合进 `UnifiedGestureGCN`，完整支持 TorchScript 动态 Batch 编译与旧权重向后兼容。
- 在 [src/gesture_gcn/config.py](src/gesture_gcn/config.py) 与 [configs/default.yaml](configs/default.yaml) 中新增 `use_shape_branch`（默认 `false`）和 `shape_grid_size` 配置开关，并在 [src/gesture_gcn/models/network.py](src/gesture_gcn/models/network.py) 接入分支装配。
- 在 [src/gesture_gcn/engine/trainer.py](src/gesture_gcn/engine/trainer.py)、[src/gesture_gcn/inference/recorder.py](src/gesture_gcn/inference/recorder.py) 与 [src/gesture_gcn/web/app.py](src/gesture_gcn/web/app.py) 中新增 Epoch 级进度回调、`GET /api/train/progress` 实时训练进度接口，以及 `POST /api/train` 的 `use_shape_branch` 动态训练参数与 `/api/status` 状态透出。
- 在 Web Studio 前端 [src/gesture_gcn/web/static/index.html](src/gesture_gcn/web/static/index.html) 新增「启用相对形状分支检测」开关、状态提示与实时可视训练进度条（展示阶段、当前轮次、准确率、Loss、耗时与流光动画）。
- 在 [tests/test_modules.py](tests/test_modules.py) 与 [tests/test_inference_and_web.py](tests/test_inference_and_web.py) 补充时间方向性、45° 旋转不变性、SHREC-22、TorchScript 编译与 Web 训练开关/进度接口端到端测试（全量 17 项通过）。

### 2026-11-27 18:54 | `chore`: 更新 12 类真实手势预训练权重 checkpoints/unified_gesture_gcn.pt
- 更新默认模型权重 [checkpoints/unified_gesture_gcn.pt](checkpoints/unified_gesture_gcn.pt)（约 1.2 MB，9 通道 Joint+Bone+Velocity、32 帧窗口），包含在 macOS 环境下新训练的 12 类完整静态+动态手势（`static_open_palm`、`static_fist`、`static_ok_pinch`、`static_peace`、`dynamic_swipe_left/right/up/down`、`dynamic_push`、`dynamic_pull`、`dynamic_grab_close`、`dynamic_circle_wave`）。

### 2026-11-27 18:43 | `docs`: 新增项目功能更新日志并在 AGENT.md 中补充日志维护规范 (`6bcf443`)
- 新增根目录 [CHANGELOG.md](CHANGELOG.md)，完整回溯并记录项目自首个版本以来的全部功能演进与修复历史。
- 更新 [AGENT.md](AGENT.md) 第 6.1 节与第 8 节，明确要求每次 `git commit` 以及上传（`git push`）到 GitHub 时，必须同步将改动时间与具体内容写入 `CHANGELOG.md` 并一并提交。

### 2026-11-27 18:38 | `fix`: 为画圈挥动增加 2D 轨迹绕转几何先验以消除过渡动作误判 (`16c04ab`)
- 在 `GestureRecognizer._apply_kinematic_prior` 中基于掌心 2D 轨迹计算 X/Y 双向跨度、纵横比、路径曲折度（路程/净位移）以及绕轨迹质心的累计有向转角与绕转一致性。
- 仅当满足真实画圈几何特征（双向跨度 `>= 0.065`、纵横比 `>= 0.32`、总路程 `>= 0.24`、曲折度 `>= 1.65`、累计净转角 `>= 200°` 且绕转一致性 `>= 0.55`）时才放行 `dynamic_circle_wave`，否则乘以 `0.08` 抑制，并在挥手回手保护期内额外收紧。
- 检测到真实完整画圈时反向抑制四向直线滑动（`×0.20`），防止画圈过程误触发翻页。
- 补充斜向抬手抑制与完整 360° 画圈放行的单元测试。

### 2026-11-27 18:18 | `fix`: 接入 BlazeFace 人脸区域过滤并加固几何/轨迹防人脸误检机制 (`4bcfbb3`)
- 集成轻量级 MediaPipe `FaceDetector`（`blaze_face_short_range.tflite`，约 230 KB，每 8 帧 CPU 节流检测），前端新增默认开启的「防人脸误检」开关，拦截凭空生成在人脸/下巴/颈部区域的误检，同时允许脸外已确认的真手平滑经过脸部前方。
- 提高 `HandLandmarker` 初始检测与存在置信度阈值（`0.78/0.74`，高精度档 `0.82/0.78`），抑制肤色区域误触发。
- 修复单手离开后大半径轨迹（`0.58`）导致人脸误检劫持放宽阈值的问题，引入紧邻半径（`0.22`）与画幅宽高比校正的掌骨/指节弧几何校验。
- 同步更新 `scripts/download_assets.py`、`start_web.sh`、`AGENT.md` 与单元测试。

### 2026-11-27 18:04 | `fix`: 统一新旧 MediaPipe 左右手输出语义并修复默认左右手颠倒问题 (`38bc408`)
- 新版 `@mediapipe/tasks-vision` `HandLandmarker` 内部已将标签映射反转为 `0=Right / 1=Left`，在未镜像原始帧上直接输出真实解剖学左右手，取消前端 `swapHands` 默认勾选。
- 在旧版 `@mediapipe/hands` 回退回调中统一翻转 `multiHandedness` 标签，确保新旧引擎在 macOS 与 Windows 上左右手语义完全一致。
- 迁移 `localStorage` 键名至 `swapHandsV2` 并清理旧键，避免旧缓存残留导致左右依旧颠倒。
- 补充前端左右手默认配置与回退归一化单元测试。

### 2026-11-27 17:36 | `feat`: 将默认训练权重 unified_gesture_gcn.pt 纳入仓库以便跨平台开箱即用 (`5cf546f`)
- 更新 `.gitignore` 放行 `checkpoints/unified_gesture_gcn.pt`（约 1.15 MB），其余临时权重与 `data/` 仍保持忽略。
- 提交已训练好的 12 类真实手势模型权重，解决迁移到 macOS / 新设备时因缺少权重回退合成模型导致识别异常的问题。
- 同步更新 `AGENT.md` 说明默认权重跟踪规则。

### 2026-11-27 00:15 | `feat`: 升级 HandLandmarker 并增加暗光自适应提亮增强与中文提交规范 (`eb557ce`)
- 升级 `scripts/download_assets.py`，支持下载新版 `@mediapipe/tasks-vision` 运行时与 `hand_landmarker.task` 至本地离线目录。
- 前端检测引擎优先启用新版 `HandLandmarker`（WebGL GPU 加速，支持 CPU 与旧版 `@mediapipe/hands` 自动回退），提升手背/侧手/遮挡精度。
- 在离屏推理画布加入自适应手部区域测光与暗部/逆光提亮对比度增强开关，不改变可见预览画面。
- 后端 `app.py` 显式注册 `.mjs` 与 `.wasm` 的 MIME 类型以兼容 ES 模块加载。
- 在 `AGENT.md` 中补充每次改动验收通过后用中文 `git commit` 的协作规范。

### 2026-11-27 00:06 | `feat`: 新增跨平台手势控电脑扩展、macOS 适配与 AGENT 部署指南 (`943eb03`)
- 新增解耦手势事件引擎（`/ws/events`）、连续帧防抖、冷却、左手张掌安全锁与挥手回手抑制。
- 适配 Windows（Win32 `SendInput`）与 macOS（CoreGraphics `CGEventPost` + `osascript`）双平台电脑操控客户端（`scripts/pc_controller.py` + `configs/pc_control.yaml`）。
- 新增 Apple Silicon MPS 设备支持及 macOS/Linux 一键启动脚本（`start_web.sh`、`start_pc_controller.sh`）。
- 新增 `AGENT.md` 自动化部署验收指南并补充单元测试。

---

## 2026-11-25

### 2026-11-25 23:42 | `perf`: 双手识别加速（TorchScript、Batch=2 合并推理与自适应 26 FPS 循环） (`0c874f5`)
- 将模型中的 `einsum` 替换为直接矩阵乘法并优化静态拓扑广播（零精度损失）。
- 推理器启用 TorchScript `trace + optimize_for_inference`、`inference_mode`、CPU 4 线程限制、双手合并 `predict_batch (B=2)` 与自适应 EMA 平滑（双手推理延迟由 `19.93ms` 降至 `8.44ms`，提速 `2.36x`）。
- 后端 `/ws/predict` 新增双手打包协议并保持单手协议向后兼容；前端合并双手单帧请求，目标帧率提升至 ~26 FPS。

### 2026-11-25 23:19 | `feat`: 新增双手独立识别面板与全中文界面 (`38d8d93`)
- 后端 `/ws/predict` 为左右手分别维护独立的滑动窗口缓冲与 EMA 平滑状态，共享同一模型权重。
- 前端新增左右手并排实时识别窗口、中文化手势名称映射（`LABEL_ZH`）、分色骨架渲染与镜像文字修正、高精度/交换左右开关，以及 Windows 一键启动脚本 `start_web.bat`。

---

## 2026-11-23

### 2026-11-23 20:59 | `feat`: 首个版本发布——统一静+动态手势识别 GCN 框架与实时 Web Studio (`ab1ed60`)
- 实现融合 DA-GCN（ARU/DRU/CTF）、DSTSA-GCN（分组静态拓扑 + MS-TCN）与 DRE（动态范围嵌入）的 `UnifiedGestureGCN` 核心网络与训练/评估引擎。
- 提供 FastAPI + WebSocket 实时摄像头识别与在线录制训练 Web Studio、本地 MediaPipe 静态资源下载脚本、合成基准 Demo、Docker 一键部署及全套自动化测试。
