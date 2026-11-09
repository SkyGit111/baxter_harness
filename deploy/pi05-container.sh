#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root="$PWD"
runtime="${CONTAINER_CLI:-nerdctl}"
name=cix-harness-pi05
image="${PI05_IMAGE:-harbor.iluvatar.com.cn:10443/saas/mr-bi150-4.4.0-aarch64-ubuntu20.04-py3.10-poc-llm-infer:v1.2.4-202605280001-ty1100-4.4.0}"
label=org.cix.local-harness.pi05
owns() {
  [[ "$("$runtime" inspect --format '{{index .Config.Labels "org.cix.local-harness.pi05"}}' "$name" 2>/dev/null || true)" == yes ]]
}
case "${1:-status}" in
  start)
    test -f models/pi05-libero/verified-revision.txt
    test -f vendor/lerobot/src/lerobot/policies/pi05/modeling_pi05.py
    test -f vendor/transformers-pi/src/transformers/models/siglip/check.py
    mkdir -p runs/deploy
    if [[ ! -f config/pi05-baxter.local.json ]]; then
      cp config/pi05-baxter.example.json config/pi05-baxter.local.json
    fi
    if "$runtime" inspect "$name" >/dev/null 2>&1; then
      owns || { echo 'Existing container does not belong to this project' >&2; exit 1; }
      if [[ "$("$runtime" inspect --format '{{.State.Running}}' "$name")" != true ]]; then
        "$runtime" start "$name"
      fi
    else
      "$runtime" run -d --network host --privileged --name "$name" --label "$label=yes" \
        -v "$root:/opt/harness" --entrypoint /bin/bash "$image" -lc 'exec sleep infinity'
    fi
    if python3 -m cix_harness check-vla >/dev/null 2>&1; then
      echo 'A policy service already occupies port 9101; preserving it.'
      exit 0
    fi
    "$runtime" exec -d "$name" bash -lc '
      source /opt/harness/deploy/pi05-env.sh
      cd /opt/harness
      exec flock -n runs/pi05-service.lock python3 -u -m cix_harness.pi05 \
        --model models/pi05-libero --profile config/pi05-baxter.local.json >runs/deploy/pi05-service.log 2>&1'
    echo 'Pi05 startup requested; inspect runs/deploy/pi05-service.log and run check-vla.'
    ;;
  status)
    tail -25 runs/deploy/pi05-service.log
    python3 -m cix_harness check-vla
    ;;
  stop)
    owns || { echo 'Container ownership not verified' >&2; exit 1; }
    "$runtime" rm -f "$name"
    ;;
  *) echo 'Usage: sudo bash deploy/pi05-container.sh {start|status|stop}' >&2; exit 2 ;;
esac
