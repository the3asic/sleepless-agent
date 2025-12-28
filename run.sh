#!/bin/bash
# 加载环境变量并运行 sleepless-agent

cd "$(dirname "$0")"
source .venv/bin/activate

# 从 .env 导出环境变量
set -a
source .env
set +a

# 运行命令
sle "$@"
