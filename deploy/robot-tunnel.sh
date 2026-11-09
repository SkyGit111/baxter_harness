#!/usr/bin/env bash
set -euo pipefail
key="${BAXTER_SSH_KEY:-$HOME/.ssh/cix_harness_baxter_ed25519}"
known="${BAXTER_KNOWN_HOSTS:-$HOME/.ssh/cix_harness_known_hosts}"
test -f "$key"
test -f "$known"
exec ssh -NT -o BatchMode=yes -o StrictHostKeyChecking=yes \
  -o "UserKnownHostsFile=$known" -o IdentitiesOnly=yes -i "$key" \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:9102:127.0.0.1:9102 "${BAXTER_SSH_TARGET:-cothink@192.168.1.101}"
