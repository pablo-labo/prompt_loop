# Anima Prompt 闭环演化工作台

基于「**ComfyUI (Anima) + 本地 WD14 识图 + 5 档评分 (`-2..+2`) + RL 8 槽位归一化权重 & 组合相性演化**」的本地交互式生图调优工作台。

## 核心特性

1. **打分工作台（4 种生图策略）**：
   - `自动 Roll`：按 8 个语义槽位（`camera / pose / expression / lighting / outfit / scene / appearance / discovered`）链式抽取组合并结合组合相性（Synergy）避开冲突搭配；
   - `A/B 对照`：锁定骨架词（可选同 Seed）进行 1~2 个变量词的纯净对比归因；
   - `不 Roll（自选 Prompt）` 与 `自选 + 补全 Roll`：自由手选词并保留 RL 归因能力。
2. **快速 LoRA 装配舱（同步全局生图页）**：
   - 在「1. 打分工作台」与「5. 自由生图」顶部/操作区直接提供 LoRA 一键启停胶囊、权重推子调强、串联顺序调整与远端 LoRA 下拉挂载，改动自动同步至 `config.json`，无需切页即可生图生效。
3. **8 槽位词池管理与扩词包**：
   - 内置语义分类器与可复用的扩词包，支持 5 档评分后的期望残差归因、Bayesian 平滑、UCB 探索、连续冷却与观察池（`staging`）自动晋升。
4. **相性星图（Astrolabe Ring / Nebula）**：
   - 默认以 NASA 白色精密仪器风格的「天文星盘」展示词间正/负向组合相性与黄金三星协同簇（`top_triads`），可切换力导向「星云」布局并将选定组合一键送回打分台或自由生图。
5. **外部好图导入**：
   - 支持拖拽 / `Cmd + V` 剪贴板贴图，通过 WD14 反推并手动筛选标签写入词池权重。

## 目录结构

- `server.py`：本地 Web 服务与 API（默认监听 `http://127.0.0.1:7866`，端口被占用时自动顺延至 `7875`）
- `web/index.html`：单文件前端工作台（打分台 / 词池 / 配置历史 / 外部导入 / 自由生图 / 相性星图）
- `rl_engine.py`：8 槽位链式采样、权重回归归因、A/B 对照更新、组合相性计算与星图后端算法
- `tag_classifier.py`：8 槽位语义规则分类器与扩展词包定义
- `storage.py`：SQLite (`data/loop.db`) 持久化层
- `comfy_client.py`：ComfyUI HTTP API 客户端及离线测试用 `MockComfyClient`
- `tagger.py`：本地 WD14 ONNX 图像反推封装
- `workflows/anima_base_skp.json`：示例 Anima ComfyUI API Workflow 模板
- `config.example.json`：默认配置模板（首次启动自动复制生成本地 `config.json`）

## 快速开始

### 1. 安装依赖

推荐使用独立 Python 虚拟环境（Python 3.10+）：

```bash
pip install -r requirements.txt
```

### 2. 可选：准备本地 WD14 ONNX 识图模型

若需启用真实 WD14 反推识图与新涌现词挖掘，请从 HuggingFace（如 `SmilingWolf/wd-swinv2-tagger-v3`）下载以下两个文件：
- `model.onnx`
- `selected_tags.csv`

放置于以下任意位置即可自动识别：
- `models/wd-swinv2-tagger-v3/`（仓库根目录内）
- `../models/wd-swinv2-tagger-v3/`（仓库上层目录）
- 或通过环境变量 `WD14_MODEL_DIR` 指定路径。

> 暂不配置模型时也可先在网页「3. 配置与历史」中关闭「启用 WD14 识图」，或在 **Mock 离线模式** 下直接体验界面与生图闭环流程。

### 3. 启动服务

```bash
python server.py
```

macOS 用户也可在终端赋予执行权限后直接双击 `启动闭环工作台.command` 启动，服务就绪后自动打开 `http://127.0.0.1:7866/`。首次运行会自动从 `config.example.json` 初始化你的本地 `config.json` 与 `data/loop.db`（两者均已加入 `.gitignore`，不会提交个人数据）。

### 4. 连接远端 ComfyUI（如通过 SSH 端口转发）

如 ComfyUI 运行在远端机器（默认端口 `8188`），先建立本地端口映射后再启动工作台：

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -L 127.0.0.1:8188:127.0.0.1:8188 <user>@<your-comfyui-host>
```

然后在网页「3. 配置与历史」或生图页顶部的「LoRA 挂载舱」点击 **刷新远端** 即可拉取可用 LoRA 并开始生图。
