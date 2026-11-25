# AGENTS.md — 开发者与 AI Coding Agent 接手改造指南

> **给接手本项目的 AI Agent（Claude Code / Cursor / Codex / Antigravity 等）**：
> 本项目最初为「**本机 Mac (无独显) + 远端独显 ComfyUI (经 SSH 端口转发) + 本地 CPU ONNX WD14 识图**」的混合闭环工作流设计。当新用户要求你“改成适配我自己的环境/工作流/模型”时，请先通读本文档以快速定位架构边界与改造点，避免误伤核心 RL 逻辑。

---

## 1. 系统拓扑与特殊流程背景

```
┌───────────────────────────────────────────────────────────────────────────┐
│ 本机 (macOS / Windows / Linux)                                            │
│                                                                           │
│  浏览器 (web/index.html)  ──HTTP JSON──▶  server.py (:7866, stdlib HTTP)  │
│                                              │                            │
│                    ┌─────────────────────────┼──────────────────────┐     │
│                    ▼                         ▼                      ▼     │
│             rl_engine.py               storage.py              tagger.py  │
│       (8槽位采样/打分归因/星图)      (data/loop.db SQLite)   (本地 CPU ONNX)│
│                    │                         ▲             wd-swinv2-v3   │
│                    ▼                         │                            │
│             comfy_client.py ───────▶ seed_library.json (首次启动自动灌库)   │
└────────────────────┬──────────────────────────────────────────────────────┘
                     │ HTTP -> http://127.0.0.1:8188
                     │ (默认假设通过 SSH -L 8188:127.0.0.1:8188 转发到远端独显)
                     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ 生图算力端 (远端 ComfyUI 服务器 / 或本机自带显卡的 ComfyUI)                  │
│   - 加载 workflows/anima_base_skp.json                                    │
│   - 动态串联 0..N 个 LoraLoaderModelOnly 节点                              │
└───────────────────────────────────────────────────────────────────────────┘
```

### 为什么这套流程比较特殊？
1. **算力与交互分离（SSH 端口转发）**：原作者用轻薄本运行前端、SQLite 与轻量 CPU 推理，把重型生图放在远端 GPU 机器的 ComfyUI 上，通过 `ssh -N -L 127.0.0.1:8188:127.0.0.1:8188 <user>@<host>` 映射到本机 `127.0.0.1:8188`。**如果新用户本身就是单机独显（Windows/Linux），不需要任何 SSH 转发，直接本机开 ComfyUI 监听 `8188` 即可。**
2. **本地 CPU ONNX WD14 识图模型（闭环反推）**：每张生成图从 ComfyUI 拉回本机后，会立刻经 `tagger.py` 用本地 `wd-swinv2-tagger-v3` (`model.onnx` + `selected_tags.csv`) 反推画面实际出现了哪些 Tag，用于：
   - **执行率归因**：Prompt 写了某词但画面没画出来时降权归因误差；
   - **差集挖掘（涌现新词）**：发现画面里出现、但 Prompt 没写的高分词，自动送入观察池 `staging`。
3. **开箱即用种子词库（`seed_library.json`）**：仓库不提交个人历史图片与二进制 `data/loop.db`，而是将已积累分类好的 8 槽位词库与组合相性先验导出为 `seed_library.json`。首次启动若 `data/loop.db` 为空会自动灌入；用户也可在网页「2. 词池与权重」点 **权重清零** 保留全部已分类词条、从零训练自己的审美权重。

---

## 2. 改造前必须知道的核心设计约束（Invariants）

1. **Prompt 刻意不生成 `(tag:1.2)` 括号权重**：
   - 原工作流底座 **Anima** 使用 Qwen LLM 文本编码器（`qwen_3_06b_base`），传统 SD1.5/SDXL 的 `(tag:weight)` 语法会破坏自然语义，因此 `rl_engine.py` 通过「**采样概率 + 槽位内按 `z_score` 降序排列**」体现权重。
   - **如果你帮用户改造成 SDXL / Illustrious / NoobAI / Pony（CLIP 编码器）**：这套位置排序依然完全兼容；若用户明确希望输出 `(tag:1.15)` 加权语法，只需在 `rl_engine.py` 组装 `prompt_chunks` 时按 `z_score` 映射即可。
2. **动态 LoRA 串联机制（`comfy_client.py -> _load_and_patch_workflow`）**：
   - 每次生图时，代码读取 `workflows/anima_base_skp.json`，找到 `config.json -> workflow_nodes.lora`（默认节点 `"10"`），先将其从图中移除，再把 `config.json -> loras` 中所有 `enabled=true && strength != 0` 的 LoRA 按顺序创建为 `LoraLoaderModelOnly` 节点链（`10 -> 1001 -> 1002 -> ...`），最后接到 `sampler.inputs.model`。
   - **注意**：如果用户换用 `CheckpointLoaderSimple`（同时输出 `MODEL` 和 `CLIP`，且希望 LoRA 同时作用于 CLIP），需要把 `comfy_client.py` 里的 `LoraLoaderModelOnly` 改成 `LoraLoader` 并同步串联 `clip` 引用。
3. **配置与数据分离**：
   - `config.json` 与 `data/` 都在 `.gitignore` 中。首次运行 `server.py` 时，`_ensure_config_exists()` 会自动从 `config.example.json` 复制生成 `config.json`。
   - 切勿把用户的 `data/loop.db`、`data/images/` 或含私密 IP/路径的 `config.json` 提交到 Git；若需分享新积累的 Tag 词库，运行 `python export_seed_library.py` 更新 `seed_library.json`。
4. **纯标准库后端 + 单文件前端**：
   - 后端仅依赖 `Pillow`、`numpy`、`onnxruntime`（无 Flask/FastAPI/React/Vite 构建步骤），修改 `web/index.html` 后浏览器 `Cmd/Ctrl + Shift + R` 即可生效。

---

## 3. 代码地图与核心接口

| 文件 | 职责与关键符号 |
| --- | --- |
| `server.py` | 本地 HTTP 服务（默认 `:7866`），`PromptLoopApp` 协调配置、异步生图线程与路由 |
| `comfy_client.py` | `ComfyClient`（`check_health`, `list_loras`, `_load_and_patch_workflow`, `generate_one`）与离线 `MockComfyClient` |
| `tagger.py` | `LocalWDTagger`（`is_available`, `tag_image_bytes`），按 `WD14_MODEL_DIR` -> `./models/wd-swinv2-tagger-v3` -> `../models/wd-swinv2-tagger-v3` 查找模型 |
| `rl_engine.py` | `RLEngine`：8 槽链式采样、A/B 对照探测、`apply_batch_ratings`、`rebuild_pair_stats`、`get_synergy_graph`（星盘骨架与三星群） |
| `tag_classifier.py` | `SLOT_ORDER`（8 槽位）、`CLUSTER_DEFS`、`classify_tag()` 规则分类器、`CURATED_EXPANSION_PACKS` |
| `storage.py` | `LoopStorage`（SQLite WAL 模式）：`batches`, `generations`, `tag_stats`, `cluster_stats`, `tag_pair_stats`, `meta_state`, `external_imports`, `manual_generations` |
| `web/index.html` | 单文件 UI（NASA 白色拟物极简风 + Tab 1/5 快速 LoRA 装配舱 + Tab 6 天文星盘 Canvas `StarMapApp`） |
| `seed_library.json` | 开箱即用的 8 槽位词库与组合相性先验快照（首次启动自动导入） |
| `export_seed_library.py` | 从本地 `data/loop.db` 只读导出脱敏 `seed_library.json` 的工具脚本 |
| `workflows/anima_base_skp.json` | ComfyUI API 格式工作流模板（需通过 ComfyUI 菜单 `Export (API)` 导出） |

### 主要 HTTP API
- `GET /api/status`：健康检查（ComfyUI、WD14、激活 LoRA、轮次与词池统计）
- `GET /api/config` / `POST /api/config`：读取与深度合并保存配置（支持只传 `{"loras": [...]}` 局部更新）
- `GET /api/loras`：拉取 ComfyUI 可用 LoRA 列表
- `POST /api/batches/generate` / `GET /api/batches/{id}` / `POST /api/batches/{id}/rate`：打分台生图与 RL 更新
- `POST /api/manual/generate` / `POST /api/manual/roll` / `GET /api/manual/history`：自由生图
- `GET /api/tags` / `POST /api/tags/action`：词池查询、改槽位、重分类、扩词包导入、`reset_weights`
- `POST /api/external/preview` / `POST /api/external/apply`：外部好图 WD14 导入
- `GET /api/synergy/graph` / `POST /api/synergy/rebuild`：相性星图数据与重算

---

## 4. 常见改造剧本（Recipes for Agents）

### 剧本 A：单机本地显卡直接跑 ComfyUI（去掉 SSH 转发）
1. 启动本地 ComfyUI（默认 `http://127.0.0.1:8188`），无需任何 SSH 命令。
2. 如端口非 `8188`（或局域网另一台机器），直接修改 `config.json` 里的 `comfy_url`（或在网页「3. 配置与历史」修改）。

### 剧本 B：更换为其他 ComfyUI 模型工作流（SDXL / Illustrious / NoobAI / Flux）
1. 在用户的 ComfyUI 界面里开启 Dev Mode，使用 **Export (API)** 导出 JSON，保存到 `workflows/custom.json`。
2. 对照导出的 JSON 节点 ID，修改 `config.json`：
   - `workflow_path`: `"workflows/custom.json"`
   - `workflow_nodes`: 填入对应节点 ID（`unet` 填 `UNETLoader` 或 `CheckpointLoaderSimple`，`positive`/`negative` 填 `CLIPTextEncode`，`latent` 填 `EmptyLatentImage`，`sampler` 填 `KSampler`，`save` 填 `SaveImage`，`lora` 填初始 LoRA 节点或留空）。
3. 若新工作流需要 LoRA 同时修改 CLIP，修改 `comfy_client.py` 中 `_load_and_patch_workflow` 的 `LoraLoaderModelOnly` 为 `LoraLoader` 并串联 `clip` 输出。

### 剧本 C：把生图后端从 ComfyUI 换成 SD-WebUI (A1111/Forge) 或云端 API
- 只需在 `comfy_client.py` 新增一个客户端类，实现与 `ComfyClient` 相同的 3 个方法，并在 `server.py -> PromptLoopApp.make_client()` 返回它即可：
  - `check_health() -> {"ok": bool, "url": str, ...}`
  - `list_loras() -> {"ok": bool, "loras": list[str]}`
  - `generate_one(positive, negative, seed, gen_cfg, loras=None) -> {"image_bytes": bytes, "comfy_prompt_id": str, "elapsed_ms": float}`

### 剧本 D：没有本地 WD14 模型时的三种处理方式
1. **推荐（最快）**：从 HuggingFace `SmilingWolf/wd-swinv2-tagger-v3` 下载 `model.onnx` 与 `selected_tags.csv`，放入 `models/wd-swinv2-tagger-v3/`（CPU 推理单张仅需 ~0.3s）。
2. **极简（零模型）**：在 `config.json` 或网页将 `enable_wd14` 设为 `false`，工作台自动降级为纯人工 5 档打分 RL + 组合相性演化（仅停用自动新词挖掘与外部贴图反推）。
3. **远端/视觉大模型替代**：保留 `LocalWDTagger.tag_image_bytes(image_bytes)` 的返回结构 `{"tags": [{"tag": str, "raw_tag": str, "score": float, "category": "general"}], "ratings": [...], "elapsed_ms": float}`，将其内部实现替换为调用远端 ComfyUI WD14 节点或 OpenAI/Gemini Vision API 即可无缝接入整个 RL 闭环。

### 剧本 E：保留预置 8 槽词库，但想从零训练自己的审美偏好
- 首次启动后已自动载入 `seed_library.json` 中的全部已分槽标签与先验权重。
- 若希望**保留所有标签和 8 槽分类，但清空原作者的喜好权重与星图先验**：在网页「2. 词池与权重」点击 **权重清零**（`POST /api/tags/action {"action": "reset_weights"}`），即可从干净的 `0` 权重重新开始演化。
- 若想修改标签归类规则或增减扩词包：直接编辑 `tag_classifier.py` 并在网页点击 **一键智能重分类**。
