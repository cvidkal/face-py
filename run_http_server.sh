#!/bin/bash
# face-py 启动 wrapper.
#
# 用法:
#   ./run_http_server.sh                    # ENV 由 shell / systemd 提供
#   ./run_http_server.sh --env dev          # source deploy/env/dev.env.example
#
# 跟 training_analyzer / cloth 等 sibling 的脚本布局对齐.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ "${1:-}" == "--env" ]]; then
    env_name="${2:-}"
    env_file="${SCRIPT_DIR}/deploy/env/${env_name}.env.example"
    if [[ -z "${env_name}" || ! -f "${env_file}" ]]; then
        echo "usage: $0 [--env dev|prod] [serve.py args...]" >&2
        echo "  ${env_file} not found" >&2
        exit 2
    fi
    set -a
    # shellcheck disable=SC1090
    source "${env_file}"
    set +a
    shift 2
fi

exec python3 -u "${SCRIPT_DIR}/service/serve.py" "$@"
