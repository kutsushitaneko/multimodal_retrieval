#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

APP_SCRIPT="multimodal_retriever.py"
# プロセス検索に使う部分一致パターン（コマンドラインに含まれる文字列）
APP_MATCH="multimodal_retriever"
LOG_FILE="output.log"
PID_FILE="multimodal_retriever.pid"
STOP_TIMEOUT_SEC=10

# 実行プラットフォームを判定する。
# Windows の Git Bash / MSYS2 / Cygwin には pgrep / pkill が無いため、
# プロセスの検索・生存確認・停止を OS ネイティブの手段（PowerShell）に切り替える。
case "$(uname -s 2>/dev/null)" in
    MINGW*|MSYS*|CYGWIN*) IS_WINDOWS=1 ;;
    *) IS_WINDOWS=0 ;;
esac

usage() {
    cat <<'EOF'
Usage: ./run.sh [start|stop|restart]

  start    アプリをバックグラウンド起動（デフォルト）
  stop     起動中のアプリを停止
  restart  停止してから起動

標準出力・標準エラーは output.log に出力します。
Linux と Windows(Git Bash 等) の両方で動作します。
EOF
}

# --- プラットフォーム抽象化 ----------------------------------------------------

# APP_MATCH を含むアプリプロセスの PID を 1 行ずつ出力する（OS ネイティブの実 PID）。
find_running_pids() {
    if ((IS_WINDOWS)); then
        # python.exe / uv.exe のうち、コマンドラインに APP_MATCH を含むものだけを抽出する。
        # PowerShell 自身は Name フィルタで除外されるため自己マッチしない。
        local ps_cmd='Get-CimInstance Win32_Process | Where-Object { $_.Name -match "python|uv" -and $_.CommandLine -match "'"$APP_MATCH"'" } | ForEach-Object { $_.ProcessId }'
        powershell -NoProfile -Command "$ps_cmd" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]+$' || true
    elif command -v pgrep >/dev/null 2>&1; then
        pgrep -f "$APP_MATCH" 2>/dev/null || true
    else
        # pgrep が無い Unix 環境向けフォールバック（[m] で grep 自身の自己マッチを回避）。
        ps -e -o pid= -o args= 2>/dev/null | grep "[${APP_MATCH:0:1}]${APP_MATCH:1}" | awk '{print $1}' || true
    fi
}

is_pid_alive() {
    local pid="$1"
    if ((IS_WINDOWS)); then
        powershell -NoProfile -Command "if (Get-Process -Id $pid -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }" >/dev/null 2>&1
    else
        kill -0 "$pid" 2>/dev/null
    fi
}

# 穏当な停止要求（Unix: SIGTERM / Windows: Stop-Process）。
terminate_pid() {
    local pid="$1"
    if ((IS_WINDOWS)); then
        powershell -NoProfile -Command "Stop-Process -Id $pid -ErrorAction SilentlyContinue" >/dev/null 2>&1 || true
    else
        kill "$pid" 2>/dev/null || true
    fi
}

# 強制終了（Unix: SIGKILL / Windows: Stop-Process -Force）。
force_pid() {
    local pid="$1"
    if ((IS_WINDOWS)); then
        powershell -NoProfile -Command "Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue" >/dev/null 2>&1 || true
    else
        kill -9 "$pid" 2>/dev/null || true
    fi
}

# ------------------------------------------------------------------------------

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

collect_pids_to_stop() {
    local -a pids=()
    local pid

    if pid="$(read_pid_file 2>/dev/null || true)"; then
        if [[ -n "$pid" ]] && is_pid_alive "$pid"; then
            pids+=("$pid")
        elif [[ -z "$(find_running_pids)" ]]; then
            # PID ファイルのプロセスは既に居らず、他にも該当プロセスが無ければ掃除する
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
    for pid in "${pids[@]}"; do
        terminate_pid "$pid"
    done

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
            echo "通常終了しなかったため強制終了します: PID $pid"
            force_pid "$pid"
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
    local launcher_pid=$!

    # 実プロセスが立ち上がるまで少し待つ（uv が python を起動するため Windows では時間差がある）。
    local started_pids=""
    local waited=0
    while ((waited < 5)); do
        sleep 1
        waited=$((waited + 1))
        started_pids="$(find_running_pids)"
        [[ -n "$started_pids" ]] && break
        # Unix では nohup 直下の PID も確認する（pgrep が遅延するケースの保険）
        if ((IS_WINDOWS == 0)) && is_pid_alive "$launcher_pid"; then
            started_pids="$launcher_pid"
            break
        fi
    done

    if [[ -z "$started_pids" ]]; then
        rm -f "$PID_FILE"
        echo "起動に失敗しました。${LOG_FILE} を確認してください。"
        exit 1
    fi

    # 代表 PID を 1 つだけ PID ファイルに記録する（停止時は find_running_pids が全プロセスを拾う）。
    echo "$started_pids" | head -n 1 > "$PID_FILE"
    echo "起動しました: PID $(echo "$started_pids" | tr '\n' ' ')"
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
