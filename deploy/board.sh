#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
models="${MODEL_DIR:-$PWD/models}"
label=org.cix.local-harness.v1
runtime="${CONTAINER_CLI:-docker}"
vlm=cix-local-vlm
vla=cix-local-vla

owns() {
  [[ "$("$runtime" inspect --format '{{index .Config.Labels "org.cix.local-harness.v1"}}' "$1" 2>/dev/null || true)" == "yes" ]]
}

case "${1:-help}" in
  prepare|prepare-vla)
    [[ "$(uname -m)" == aarch64 ]] || { echo "Run prepare on TY1100 ARM64" >&2; exit 1; }
    ixsmi
    while read -r digest file url; do
      if [[ $1 == prepare-vla && $file != deploy/wheels/* ]]; then continue; fi
      if [[ ! -f "$file" ]]; then
        mkdir -p "$(dirname "$file")"
        curl -fL --retry 3 -C - -o "$file.part" "$url"
        echo "$digest  $file.part" | sha256sum -c -
        mv "$file.part" "$file"
      else
        echo "$digest  $file" | sha256sum -c -
      fi
    done < deploy/runtime.lock
    if [[ $1 == prepare-vla ]]; then
      python3 deploy/fetch_models.py --dir "$models" --only openvla
    else
      python3 deploy/fetch_models.py --dir "$models"
      "$runtime" build -f deploy/Dockerfile.vlm -t cix-harness/vlm:b12121 .
    fi
    "$runtime" build -f deploy/Dockerfile.vla -t cix-harness/openvla:1 .
    ;;
  start|start-vla)
    [[ -f "$models/openvla/model.safetensors.index.json" ]] || { echo "Missing OpenVLA in $models" >&2; exit 1; }
    # A fixed container name prevents multiple model instances from exhausting memory.
    created=()
    cleanup() {
      for name in "${created[@]}"; do
        "$runtime" logs --tail 24 "$name" >&2 || true
        "$runtime" rm -f "$name" >/dev/null || true
      done
    }
    trap cleanup ERR
    trap 'cleanup; exit 130' INT TERM
    check=check-vla
    if [[ $1 == start ]]; then
      [[ -f "$models/qwen/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf" && -f "$models/qwen/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf" ]] || { echo "Missing staged Qwen in $models" >&2; exit 1; }
      "$runtime" run -d --name "$vlm" --label "$label=yes" --network host \
      -v "$models/qwen:/weights:ro" cix-harness/vlm:b12121 \
      -m /weights/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf --mmproj /weights/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf \
      --alias cix-vlm --host 127.0.0.1 --port 9100 --ctx-size 4096 --parallel 1 --n-gpu-layers 0
      created+=("$vlm")
      check=check
    fi
    "$runtime" run -d --name "$vla" --label "$label=yes" --network host --privileged --shm-size=2g \
      -v "$models/openvla:/models/openvla:ro" cix-harness/openvla:1
    created+=("$vla")
    deadline=$((SECONDS + 360))
    while (( SECONDS < deadline )); do
      for name in "${created[@]}"; do
        [[ "$("$runtime" inspect --format '{{.State.Running}}' "$name")" == true ]] || { echo "$name exited" >&2; false; }
      done
      if python3 -m cix_harness "$check" 2>/dev/null; then
        echo "Warming OpenVLA with the example image..."
        python3 -m cix_harness predict --image examples/baxter-d455.jpg --subtask 'pick up the red bar'
        exit 0
      fi
      sleep 2
    done
    echo "Models did not become ready; see startup logs" >&2
    false
    ;;
  check) python3 -m cix_harness check ;;
  stop)
    for name in "$vla" "$vlm"; do
      if owns "$name"; then "$runtime" rm -f "$name"; fi
    done
    ;;
  *) echo "Usage: bash deploy/board.sh {prepare|prepare-vla|start|start-vla|check|stop}" ;;
esac
