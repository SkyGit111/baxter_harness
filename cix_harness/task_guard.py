"""Bounds derived from an operator-observed fingertip/table contact pose."""
import math
from .contract import vector
from .pi05_contract import quaternion, rotate


def validate_guard(guard):
    if guard.get('source') != 'operator_confirmed_fingertips_at_table':
        raise ValueError('A measured fingertip/table reference is required')
    vector(guard['reference_eef_m'],3,'contact position')
    quaternion(guard['reference_eef_xyzw'])
    for key,lo,hi in [('clearance_m',.002,.05),('radius_m',.01,.35),
                      ('tool_extent_bound_m',.05,.30)]:
        value=guard[key]
        if type(value) not in (int,float) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError('Invalid contact guard '+key)


def contact_bounds(guard, xyzw):
    validate_guard(guard)
    reference=guard['reference_eef_m']
    current_axis=rotate(quaternion(xyzw),[0,0,1])
    reference_axis=rotate(quaternion(guard['reference_eef_xyzw']),[0,0,1])
    tilt=math.acos(max(-1.,min(1.,sum(a*b for a,b in zip(current_axis,reference_axis)))))
    # For every tool point r with ||r|| <= extent, its vertical displacement
    # is bounded by extent * ||R_current.T*z - R_reference.T*z||. Comparing
    # only tool Z axes misses local yaw of an already tilted, asymmetric tool.
    current_q=quaternion(xyzw)
    reference_q=quaternion(guard['reference_eef_xyzw'])
    current_vertical=rotate([-current_q[0],-current_q[1],-current_q[2],current_q[3]],[0,0,1])
    reference_vertical=rotate([-reference_q[0],-reference_q[1],-reference_q[2],reference_q[3]],[0,0,1])
    margin=guard['tool_extent_bound_m']*math.dist(current_vertical,reference_vertical)
    floor=reference[2]+guard['clearance_m']+margin
    return {'minimum_eef_z_m':floor,'tool_tilt_rad':tilt,'orientation_margin_m':margin}


def check_contact_pose(guard, xyz, xyzw, *, enforce_planning_margin=True):
    bounds=contact_bounds(guard,xyzw)
    position=vector(xyz,3,'endpoint')
    reference=guard['reference_eef_m']
    # This radius bounds use of the local table reference, not free-space
    # travel with the whole estimated tool well above that surface.
    if (math.hypot(position[0]-reference[0],position[1]-reference[1]) > guard['radius_m'] and
            position[2] < bounds['minimum_eef_z_m']+.10):
        raise ValueError('Outside measured local table region')
    floor=bounds['minimum_eef_z_m']
    if not enforce_planning_margin:
        floor-=guard['clearance_m']
    bounds['enforced_eef_z_m']=floor
    if position[2] < floor:
        raise ValueError('Fingertip/table clearance violated: endpoint z %.4f < %.4f'%(position[2],floor))
    return bounds
