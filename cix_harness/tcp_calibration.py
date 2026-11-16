"""Read-only pivot calibration: keep the same tool tip at one fixed point while rotating."""
import argparse
import json
import math
from pathlib import Path
from .contract import vector
from .pi05_contract import quaternion,rotate
from .rpc import HttpRpcClient


def solve(matrix,rhs):
    rows=[list(r)+[b] for r,b in zip(matrix,rhs)]
    for col in range(len(rhs)):
        pivot=max(range(col,len(rhs)),key=lambda i:abs(rows[i][col]))
        rows[col],rows[pivot]=rows[pivot],rows[col]
        scale=rows[col][col]
        if abs(scale)<1e-6:raise ValueError('Insufficient orientation diversity for TCP calibration')
        rows[col]=[v/scale for v in rows[col]]
        for i in range(len(rhs)):
            if i!=col:
                factor=rows[i][col];rows[i]=[a-factor*b for a,b in zip(rows[i],rows[col])]
    return [r[-1] for r in rows]


def fit(samples):
    if not isinstance(samples,list) or len(samples)<4:raise ValueError('At least four fixed-tip orientations are required')
    arms={s.get('arm') for s in samples}
    if len(arms)!=1 or not arms<={'left','right'}:raise ValueError('Samples must belong to one arm')
    rows=[];rhs=[];poses=[]
    for sample in samples:
        p=vector(sample['eef_m'],3,'EEF');q=quaternion(sample['eef_xyzw']);poses.append((p,q))
        columns=[rotate(q,[int(j==k) for j in range(3)]) for k in range(3)]
        for j in range(3):
            rows.append([columns[k][j] for k in range(3)]+[-int(j==k) for k in range(3)])
            rhs.append(-p[j])
    normal=[[sum(r[i]*r[j] for r in rows) for j in range(6)] for i in range(6)]
    target=[sum(r[i]*b for r,b in zip(rows,rhs)) for i in range(6)]
    result=solve(normal,target);offset=result[:3];point=result[3:]
    residual=[math.dist([a+b for a,b in zip(p,rotate(q,offset))],point) for p,q in poses]
    rms=math.sqrt(sum(e*e for e in residual)/len(residual))
    if math.sqrt(sum(v*v for v in offset))>.4:raise ValueError('Estimated tool offset is implausible')
    if rms>.005 or max(residual)>.01:raise ValueError('Fixed-tip residual too large; repeat contact measurements')
    return {'arm':next(iter(arms)),'tcp_offset_tool_m':offset,'fixed_point_base_m':point,
            'samples':len(samples),'rms_m':rms,'max_error_m':max(residual),
            'method':'fixed_tip_pivot','commissioned':False,
            'note':'Requires actual same-point contact and independent field check; no automatic motion authorization.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['capture','fit']);p.add_argument('--samples',type=Path,required=True)
    p.add_argument('--arm',choices=['left','right']);p.add_argument('--robot-port',type=int,default=9102)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    rows=json.loads(a.samples.read_text(encoding='utf-8')) if a.samples.exists() else []
    if a.command=='capture':
        if not a.arm:p.error('capture requires --arm')
        rpc=HttpRpcClient(a.robot_port);h=rpc.call('healthz')
        if h.get('armed') is not False:raise RuntimeError('Record calibration only while bridge is read-only')
        obs=rpc.call('dual.observe',(a.arm,)) if h.get('dual_protocol') else rpc.call('env.observe')
        state=obs['state']
        if state['arm']!=a.arm:raise ValueError('Requested arm differs from observed arm')
        if rows and rows[0]['arm']!=a.arm:raise ValueError('Do not mix arm samples')
        rows.append({k:state[k] for k in ('arm','eef_m','eef_xyzw')})
        a.samples.parent.mkdir(parents=True,exist_ok=True)
        a.samples.write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'samples':len(rows),'applied':False}))
    else:
        result=fit(rows)
        if a.output:
            a.output.parent.mkdir(parents=True,exist_ok=True)
            a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()
