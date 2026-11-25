#!/bin/bash
cd "$(dirname "$0")"
echo "正在启动 Anima Prompt 闭环演化工作台..."

if [ -x "$HOME/miniforge3/envs/wd-tagger/bin/python" ]; then
  PY_BIN="$HOME/miniforge3/envs/wd-tagger/bin/python"
elif [ -x "$HOME/miniconda3/envs/wd-tagger/bin/python" ]; then
  PY_BIN="$HOME/miniconda3/envs/wd-tagger/bin/python"
elif [ -x ".venv/bin/python" ]; then
  PY_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY_BIN="$(command -v python3)"
else
  PY_BIN="python"
fi

"$PY_BIN" server.py
