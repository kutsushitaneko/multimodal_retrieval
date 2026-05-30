#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

APP_SCRIPT="multimodal_retriever.py"
LOG_FILE="output.log"
PID_FILE="multimodal_retriever.pid"
STOP_TIMEOUT_SEC=10

usage() {
    cat <<'EOF'
Usage: ./run.sh [start|stop|restart]

  start    アプリをバックグラウンド起動（デフォルト）
  stop     起動中のアプリを停止
  restart  停止してから起動

標準出力・標準エラーは output.log に出力します。
EOF
}

is_pid_alive() {
    local pid="$1"
    kill -0 "$pid" 2>/dev/null
}

read_pid_file() {
    if [[ ! -f "$PID_FILE" ]]; then
        return 1
    fi
    local pid
    pid="$(tr -d '[:space:]' < "$PID_FILE")"
    if [[ -z "$pid" || ! "$pid" =~ ^[0-9]+$ ]]; then
        rm -f "$PID_FILE"
        return 1
    fi
    echo "$pid"
}

find_running_pids() {
    pgrep -f "${APP_SCRIPT}" 2>/dev/null || true
}

collect_pids_to_stop() {
    local -a pids=()
    local pid

    if pid="$(read_pid_file 2>/dev/null || true)"; then
        if is_pid_alive "$pid"; then
            pids+=("$pid")
        else
            rm -f "$PID_FILE"
        fi
    fi

    while IFS= read -r pid; do
        [[ -z "$pid" ]] && continue
        local seen=0
        for existing in "${pids[@]:-}"; do
            if [[ "$existing" == "$pid" ]]; then
                seen=1
                break
            fi
        done
        if [[ "$seen" -eq 0 ]]; then
            pids+=("$pid")
        fi
    done < <(find_running_pids)

    if ((${#pids[@]} > 0)); then
        printf '%s\n' "${pids[@]}"
    fi
}

stop_app() {
    local -a pids=()
    while IFS= read -r pid; do
        [[ -n "$pid" ]] && pids+=("$pid")
    done < <(collect_pids_to_stop || true)

    if ((${#pids[@]} == 0)); then
        echo "停止対象のプロセスは見つかりませんでした。"
        rm -f "$PID_FILE"
        return 0
    fi

    echo "停止中: PID ${pids[*]}"
    kill "${pids[@]}" 2>/dev/null || true

    local elapsed=0
    while ((elapsed < STOP_TIMEOUT_SEC)); do
        local still_running=0
        for pid in "${pids[@]}"; do
            if is_pid_alive "$pid"; then
                still_running=1
                break
            fi
        done
        if ((still_running == 0)); then
            break
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    for pid in "${pids[@]}"; do
        if is_pid_alive "$pid"; then
            echo "SIGTERM で終了しなかったため SIGKILL します: PID $pid"
            kill -9 "$pid" 2>/dev/null || true
        fi
    done

    rm -f "$PID_FILE"
    echo "停止しました。"
}

start_app() {
    local -a running_pids=()
    while IFS= read -r pid; do
        [[ -n "$pid" ]] && running_pids+=("$pid")
    done < <(collect_pids_to_stop || true)

    if ((${#running_pids[@]} > 0)); then
        echo "既に起動中です: PID ${running_pids[*]}"
        echo "再起動する場合は ./run.sh restart を実行してください。"
        exit 1
    fi

    echo "起動中: uv run ${APP_SCRIPT}"
    PYTHONUNBUFFERED=1 nohup uv run "$APP_SCRIPT" > "$LOG_FILE" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_FILE"

    sleep 1
    if ! is_pid_alive "$pid"; then
        rm -f "$PID_FILE"
        echo "起動に失敗しました。${LOG_FILE} を確認してください。"
        exit 1
    fi

    echo "起動しました: PID $pid"
    echo "ログ: ${SCRIPT_DIR}/${LOG_FILE}"
}

restart_app() {
    stop_app
    start_app
}

main() {
    local command="${1:-start}"

    case "$command" in
        start)
            start_app
            ;;
        stop)
            stop_app
            ;;
        restart)
            restart_app
            ;;
        -h|--help|help)
            usage
            ;;
        *)
            echo "不明なコマンド: $command" >&2
            usage >&2
            exit 1
            ;;
    esac
}

main "$@"
