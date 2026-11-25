# Anima Prompt 闭环演化工作台

基于「**ComfyUI + 本地 WD14 识图 + 5 档评分 (`-2..+2`) + RL 8 槽位归一化权重 & 组合相性演化**」的本地交互式生图调优工作台。

> **给接手改造的同学 & AI Coding Agent**：
> 如果你打算用 **Claude Code / Cursor / Codex / Antigravity** 等 Agent 把本项目改造成适配你自己的电脑、显卡、SDXL/Flux 工作流或 API，**请先让 Agent 阅读仓库根目录的 [`AGENTS.md`](./AGENTS.md)** —— 里面完整梳理了本项目的特殊混合架构（SSH 端口转发 + 远端 ComfyUI + 本地 ONNX 模型）、核心约束、API 契约与 5 套直接可执行的改造剧本。

---

## 核心特性

1. **开箱即用的预置 Tag 词库（`seed_library.json`）**：
   - 仓库内置了经过实际迭代积累的 **746 个 8 槽位分类标签**（`camera / pose / expression / lighting / outfit / scene / appearance / discovered`）与 **2,309 组组合相性先验**。
   - 首次启动 `server.py` 会自动灌入本地 `data/loop.db`，开箱即可直接 Roll 词并浏览完整的相性星图；如果想**保留这 746 个已分类标签但从零训练你自己的审美权重**，只需在「2. 词池与权重」点击 **权重清零** 即可。
2. **打分工作台（4 种生图策略）**：
   - `自动 Roll`：按 8 槽位链式抽取并结合组合相性（Synergy）避开冲突搭配；
   - `A/B 对照`：锁定骨架词（可选同 Seed）精测 1~2 个变量词的纯差异归因；
   - `不 Roll（自选 Prompt）` 与 `自选 + 补全 Roll`。
3. **全生图页快速 LoRA 装配舱**：
   - 在「1. 打分工作台」与「5. 自由生图」均内嵌 **LORA 挂载舱**，支持一键启停胶囊、权重推子调节、串联顺序调整与远端 LoRA 下拉挂载，改动自动持久化到 `config.json`，无需切页即可生效。
4. **相性星图（Astrolabe Ring / Nebula）**：
   - NASA 白色拟物极简风格的「天文星盘」视图，清晰呈现词间正向搭档、避雷冲突与黄金三星协同簇（`top_triads`），支持切换「星云」布局及一键组词送回生图台。
5. **外部好图导入**：
   - 支持拖拽 / `Cmd+V` 贴图，经本地 WD14 反推标签并手动筛选注入词池。

---

## 为什么这套流程比较特殊？（架构速览）

本项目默认采用 **轻量本机 + 远端 GPU** 的拆分部署模式（也完全支持单机直接运行）：

- **本机运行**：`server.py`（纯 Python 标准库 Web 服务，端口 `7866`）、SQLite（`data/loop.db`）以及 **本地 CPU 版 WD14 ONNX 模型**（`tagger.py`），负责 UI 交互、RL 权重演化与图片自动反推打标。
- **远端 GPU 运行**：ComfyUI 负责扩散生图。本机通过 SSH 端口转发将远端 `8188` 映射到本机 `127.0.0.1:8188`。
  - **如果你本机自带独显（Windows / Linux 本地跑 ComfyUI）**：无需 SSH 转发，直接本机开 ComfyUI 监听 `8188`（或在配置中修改 `comfy_url`）即可。

---

## 快速开始

### 1. 安装依赖（Python 3.10+）

```bash
pip install -r requirements.txt
```

*(仅需 `Pillow`、`numpy`、`onnxruntime` 三个依赖，无前端打包步骤。)*

### 2. 可选：放置本地 WD14 ONNX 识图模型

如需启用自动识图与新词挖掘，从 HuggingFace（`SmilingWolf/wd-swinv2-tagger-v3`）下载以下两个文件：
- `model.onnx`
- `selected_tags.csv`

放到以下任一目录即可自动识别：
- `models/wd-swinv2-tagger-v3/`（仓库根目录内）
- `../models/wd-swinv2-tagger-v3/`（仓库上层目录）
- 或通过环境变量 `WD14_MODEL_DIR` 指定路径。

> 暂时不想下载模型？可在「3. 配置与历史」关闭 **启用 WD14 识图**（纯人工打分依然完整可用），或打开 **Mock 离线模式** 直接体验全套界面与星图交互。

### 3. 连接 ComfyUI（二选一）

- **方式 A（本机有显卡）**：直接启动你本机的 ComfyUI（默认 `http://127.0.0.1:8188`）。
- **方式 B（远端服务器通过 SSH 端口转发）**：
  ```bash
  ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -L 127.0.0.1:8188:127.0.0.1:8188 <user>@<your-comfyui-host>
  ```

### 4. 启动工作台

```bash
python server.py
```

macOS 也可直接双击 `启动闭环工作台.command`。首次启动会自动：
1. 从 `config.example.json` 复制生成本地 `config.json`；
2. 从 `seed_library.json` 初始化 `data/loop.db` 预置词库与星图先验；
3. 自动打开 `http://127.0.0.1:7866/`。

*(个人 `config.json` 与 `data/` 目录均在 `.gitignore` 中，不会被提交到 Git。若日后想把你新积累的词库打包分享给别人，运行 `python export_seed_library.py` 即可脱敏更新 `seed_library.json`。)*

---

## 目录结构

- `AGENTS.md`：给 AI Agent 与二次开发者的架构说明与 5 套改造剧本
- `server.py`：本地 Web 服务与 API（默认 `127.0.0.1:7866`，占用自动顺延至 `7875`）
- `web/index.html`：单文件前端界面（打分台 / 词池 / 配置历史 / 外部导入 / 自由生图 / 相性星图）
- `rl_engine.py`：8 槽位链式采样、权重归因、A/B 对照更新、组合相性与星图算法
- `tag_classifier.py`：8 槽位规则分类器与扩词包
- `storage.py`：SQLite (`data/loop.db`) 持久化层
- `comfy_client.py`：ComfyUI API 客户端（含动态多 LoRA 串联）与 `MockComfyClient`
- `tagger.py`：本地 WD14 ONNX 封装
- `seed_library.json`：开箱即用的预置词库与组合相性先验
- `export_seed_library.py`：从本地 `data/loop.db` 导出脱敏 `seed_library.json`
- `workflows/anima_base_skp.json`：示例 Anima ComfyUI API Workflow
- `config.example.json`：默认配置模板
