"""Record a manually verified view-clearing position. Reads telemetry; never actuates."""
import argparse
import copy
import json
import math
from pathlib import Path
from .primitives import tcp_from_eef, validate_profile
from .rpc import HttpRpcClient


def add_view(cfg,obs):
    result=copy.deepcopy(cfg)
    state=obs['state'];p=result['primitives']
    if state['arm']!=cfg['arm']:raise ValueError('Observed arm differs from profile arm')
    position=tcp_from_eef(state['eef_m'],state['eef_xyzw'],p)
    views=p.setdefault('observation_tcp_m',[])
    if any(math.dist(position,v)<.02 for v in views):raise ValueError('Observation position already recorded nearby')
    views.append(position);validate_profile(p)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,help='Local primitive profile on this computer')
    parser.add_argument('--robot-port',type=int,default=9102)
    args=parser.parse_args();path=Path(args.config)
    cfg=json.loads(path.read_text(encoding='utf-8'))
    obs=HttpRpcClient(args.robot_port).call('env.observe')
    updated=add_view(cfg,obs)
    # Validate completely before replacing the file. No calibration/arming flags change.
    pending=path.with_name(path.name+'.pending')
    pending.write_text(json.dumps(updated,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    pending.replace(path)
    print(json.dumps({'recorded_tcp_m':updated['primitives']['observation_tcp_m'][-1],
                      'applied':False,'restart_bridge_to_load_profile':True}))


if __name__=='__main__':main()
