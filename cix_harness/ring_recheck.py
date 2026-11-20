"""Recheck a material-bound pick once without asking the planner to select the same ring again."""
import math

from .contract import vector
from .ring_peg import selected


def _bound_ring(receipt,obs,decision):
    if (decision.get('skill')!='pick' or receipt.get('status')!='rejected' or receipt.get('applied') is not False
            or receipt.get('recoverable') is not True or receipt.get('failure_code')!='ring_geometry'
            or receipt.get('obs_id')!=obs.get('id') or obs.get('state',{}).get('gripping') is not False):return None
    legacy=receipt.get('message')=='No fingertip contact depth above the local tabletop'
    if receipt.get('geometry_independent_of_uv') is not True and not legacy:return None
    try:ring=selected(obs['ring_scene'],'rings',decision['uv'],('center_uv_px','rim_grasp_uv_px'),receipt.get('selected_ring_id'))
    except (KeyError,TypeError,ValueError):return None
    return ring if ring.get('location')=='loose' and ring.get('peg_id') is None else None


def _fresh_match(old,before,now):
    scene=now.get('ring_scene',{});state=now.get('state',{})
    if (not now.get('id') or now['id']==before['id'] or scene.get('obs_id')!=now['id'] or
            state.get('arm')!=before['state'].get('arm') or state.get('gripping') is not False):return None
    same=[r for r in scene.get('rings',[]) if r.get('id')==old.get('id') and r.get('color')==old.get('color')
          and r.get('location')=='loose' and r.get('peg_id') is None]
    if len(same)!=1:return None
    new=same[0]
    try:
        a=vector(old['center_base_m'],3,'prior ring centre');b=vector(new['center_base_m'],3,'ring centre')
        radii=vector([old['outer_radius_m'],new['outer_radius_m']],2,'ring radii');radius=min(radii)
        vector(new['center_uv_px'],2,'ring landmark')
    except (KeyError,TypeError,ValueError):return None
    # These thresholds only decide whether the same semantic selection is still valid.
    # The bridge always rebuilds the entire pose and all motion checks on the fresh frame.
    if (radius<=0 or math.dist(a[:2],b[:2])>min(.006,.35*radius) or abs(a[2]-b[2])>.008 or
            abs(old['outer_radius_m']-new['outer_radius_m'])>.15*radius):return None
    return new


def prepare(robot,log,obs,decision,options,observe,remaining,allow_recovery):
    def request(frame,choice):
        return robot.call('primitive.prepare',(frame['id'],choice['skill'],choice.get('uv')),timeout_s=60,**options)
    result=request(obs,decision);ring=_bound_ring(result,obs,decision)
    if ring is None:return result,obs,decision,0
    if remaining<=0 or (allow_recovery and not allow_recovery('ring_geometry')):
        return dict(result,recoverable=False),obs,decision,0
    fresh=observe();new=_fresh_match(ring,obs,fresh)
    if new is None:
        log('ring_selection_changed',previous_obs_id=obs['id'],obs_id=fresh.get('id'),ring_id=ring['id'])
        return result,obs,decision,0
    w,h=fresh['ring_scene']['resolution'];uv=[new['center_uv_px'][0]/(w-1),new['center_uv_px'][1]/(h-1)]
    choice=dict(decision,uv=uv)
    log('ring_pick_geometry_rejected',**result)
    log('recovery',kind='recheck_same_ring',cause='ring_geometry',previous_obs_id=obs['id'],obs_id=fresh['id'],ring_id=new['id'])
    log('ring_pick_relocalized',obs_id=fresh['id'],previous_obs_id=obs['id'],ring_id=new['id'],uv=uv,
        center_xy_change_m=math.dist(ring['center_base_m'][:2],new['center_base_m'][:2]),
        radius_change_m=abs(new['outer_radius_m']-ring['outer_radius_m']),previous_table_upper_z_m=ring.get('local_table_upper_z_m'),
        current_table_upper_z_m=new.get('local_table_upper_z_m'))
    repeated=request(fresh,choice)
    if _bound_ring(repeated,fresh,choice) is not None:
        repeated=dict(repeated,recoverable=False,failure_code='ring_pick_geometry_unchanged',
                      original_failure_code='ring_geometry',previous_message=result.get('message'),previous_obs_id=obs['id'])
    return repeated,fresh,choice,1
