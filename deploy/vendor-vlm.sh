#!/usr/bin/env bash
# Validated on the lab's COREX 4.4.0 ARM64 vendor image, using existing AWQ weights.
set -euo pipefail
runtime="${CONTAINER_CLI:-nerdctl}"
name=cix-harness-qwen35-awq
label=org.cix.local-harness.vendor-vlm
image="${VLM_IMAGE:-harbor.iluvatar.com.cn:10443/saas/mr-bi150-4.4.0-aarch64-ubuntu20.04-py3.10-poc-llm-infer:v1.2.4-202605280001-ty1100-4.4.0}"
models="${VLM_MODEL_DIR:-/home/cix/models/Qwen3.5-9B-AWQ}"
case "${1:-status}" in
  start)
    test -f "$models/config.json"
    "$runtime" image inspect "$image" >/dev/null
    # A name collision fails without replacing another running service.
    "$runtime" run -d --network host --name "$name" --label "$label=yes" --privileged \
      -v "$models:/model:ro" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
      -e LD_PRELOAD=/usr/local/lib/python3.10/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0 \
      --entrypoint /bin/bash "$image" -lc \
      'exec python3 -m vllm.entrypoints.openai.api_server --model /model --served-model-name qwen35-9b --host 127.0.0.1 --port 9110 --gpu-memory-utilization 0.48 --max-model-len 2048 --max-num-seqs 1 --dtype half --enforce-eager --default-chat-template-kwargs '\''{"enable_thinking":false}'\'''
    ;;
  status)
    "$runtime" logs --tail 30 "$name"
    curl --noproxy '*' -fsS --max-time 5 http://127.0.0.1:9110/v1/models
    ;;
  stop)
    # Also accept the exact initial lab container, created before this launcher.
    owned="$("$runtime" inspect --format '{{index .Config.Labels "org.cix.local-harness.vendor-vlm"}}' "$name")"
    id="$("$runtime" inspect --format '{{.Id}}' "$name")"
    [[ "$owned" == yes || "$id" == 6e86f4eb25b8ecf2c070377466baaa2b025ce5085bb7887b00d5e91e80b3f4ab ]] || {
      echo 'Container ownership is unverified; refusing to remove it.' >&2; exit 1;
    }
    "$runtime" rm -f "$name"
    ;;
  *) echo 'Usage: sudo bash deploy/vendor-vlm.sh {start|status|stop}' >&2; exit 2 ;;
esac
