#!/usr/bin/env bash
# Source inside the vendor container. Retain the COREX Torch installation.
export PYTHONPATH=/opt/harness/vendor/pi05-python:/opt/harness/vendor/transformers-pi/src:/opt/harness/vendor/lerobot/src:/opt/harness:/usr/local/corex/lib64/python3/dist-packages
export LD_PRELOAD=/usr/local/lib/python3.10/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
