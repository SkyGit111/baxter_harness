#!/usr/bin/env bash
# Run on the ROS control PC; this entry point never enables motor commands.
set -eo pipefail
cd "$(dirname "$0")/.."
: "${ROS_MASTER_URI:?Set ROS_MASTER_URI to the Baxter ROS master URL}"
: "${ROS_IP:?Set ROS_IP to this control PC address on the robot LAN}"
unset ROS_HOSTNAME
source /opt/ros/noetic/setup.bash
source "${BAXTER_WORKSPACE:-$HOME/catkin_ws}/devel/setup.bash"
/usr/bin/python3 - <<'PY'
import json
from pathlib import Path
import pyrealsense2 as rs
path = Path('config/baxter.local.json')
if not path.exists():
    devices = [d for d in rs.context().query_devices() if 'D455' in d.get_info(rs.camera_info.name)]
    if len(devices) != 1:
        raise SystemExit('Connect one D455, or explicitly configure camera_serial in config/baxter.local.json')
    config = json.loads(Path('config/baxter.example.json').read_text())
    config['camera_serial'] = devices[0].get_info(rs.camera_info.serial_number)
    path.write_text(json.dumps(config, indent=2) + '\n')
PY
exec /usr/bin/python3 -u -m cix_harness.baxter --config config/baxter.local.json
