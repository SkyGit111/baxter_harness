"""Ground a loose annulus and its same-colour peg using one measured RGB-D scene."""
import math

from .contract import vector
from .pi05_contract import rotate
from .primitives import number,tcp_from_eef


class RingGeometryError(ValueError):
    pass


class RingAlignmentNeeded(RingGeometryError):
    def __init__(self,dx,dy,allowance,frame):
        self.dx,self.dy,self.allowance,self.frame=dx,dy,allowance,frame
        super().__init__('Observed hole is offset from peg by {:.1f} mm, allowance {:.1f} mm'.format(
            1000*math.hypot(dx,dy),1000*allowance))


def validate_settings(settings):
    for name,low,high in [('ring_thickness_m',.001,.10),('horizontal_uncertainty_m',.0001,.10),
                          ('vertical_uncertainty_m',.0001,.10),('jaw_open_half_span_m',.002,.15),
                          ('finger_half_width_m',.0005,.05),('fingertip_below_tcp_m',0,.15),
                          ('release_gap_m',.001,.03),('max_drop_m',.001,.30)]:
        number(settings[name],low,high,name)
    axis=vector(settings['jaw_axis_tool'],3,'finger opening axis')
    if abs(math.sqrt(sum(x*x for x in axis))-1)>.001:raise ValueError('Expected unit finger opening axis')
    if not isinstance(settings.get('perception'),dict):raise ValueError('Missing ring perception configuration')
    if settings.get('source_peg_id') is not None:raise ValueError('Only loose tabletop rings may be picked')
    number(settings.get('servo_step_m',.008),.001,.025,'ring servo step')
    if 'relative_xy_uncertainty_m' in settings:
        number(settings['relative_xy_uncertainty_m'],.0001,.1,'simultaneous hole/peg alignment uncertainty')
    if type(settings.get('servo_iterations',3)) is not int or not 0<=settings.get('servo_iterations',3)<=6:
        raise ValueError('Invalid ring servo iteration limit')


def measured(item,key,n=None):
    try:
        if n:return vector(item[key],n,key)
        return number(item[key],-2,2,key)
    except (ValueError,TypeError,KeyError) as exc:raise RingGeometryError('Missing measured '+key) from exc


def indexed(scene,kind,ident):
    matches=[item for item in scene.get(kind,[]) if item.get('id')==ident]
    if not ident or len(matches)!=1:raise RingGeometryError('Missing or ambiguous '+kind+' identity')
    return matches[0]


def unit_up(item,name):
    if item.get('normal_resolved') is False:raise RingGeometryError('Surface normal is unresolved')
    axis=measured(item,name,3);length=math.sqrt(sum(x*x for x in axis))
    if abs(length-1)>.01 or axis[2]<=0:raise RingGeometryError('Unverified upward '+name)
    return [v/length for v in axis]


def selected(scene,kind,uv,points,ident=None):
    """A pixel selects identity. It never supplies depth from the hole or the peg top as a contact."""
    width,height=scene['resolution'];u,v=vector(uv,2,'normalized ring target')
    if not 0<=u<=1 or not 0<=v<=1:raise RingGeometryError('Target outside image')
    p=(u*(width-1),v*(height-1));picked=[]
    for row in scene.get(kind,[]):
        if ident and row.get('id')!=ident:continue
        panel=[row.get(k) for k in points if row.get(k) is not None]
        if any(math.dist(p,vector(q,2,'image landmark'))<=row.get('selection_radius_px',5.) for q in panel):
            picked.append(row)
    if len(picked)!=1:raise RingGeometryError('Select one observed '+kind+' landmark in the current image')
    return picked[0]


def dimension(ring):
    small=measured(ring,'inner_radius_m');large=measured(ring,'outer_radius_m')
    if not .001<small<large<.20:raise RingGeometryError('Unresolved ring inner and outer radii')
    return small,large


def fit_error(item,name):
    # Reported per-frame contour/plane fit error adds to the commissioned pose uncertainty.
    if name not in item:return 0.
    value=measured(item,name)
    if value<0:raise RingGeometryError('Invalid fit error '+name)
    return value


def color(row):
    value=row.get('color')
    if not isinstance(value,str) or value.lower() in ('','unknown'):raise RingGeometryError('Unverified colour')
    return value


def pick_geometry(profile,scene,ring,orientation):
    s=profile['ring_peg'];center=measured(ring,'center_base_m',3);rim=measured(ring,'rim_grasp_base_m',3)
    inner,outer=dimension(ring);margin=s['horizontal_uncertainty_m']+max(fit_error(ring,'inner_boundary_p90_m'),fit_error(ring,'outer_boundary_p90_m'))
    vertical=s['vertical_uncertainty_m']+fit_error(ring,'depth_plane_p90_m')
    radial=math.dist(center[:2],rim[:2]);normal=unit_up(ring,'normal_base')
    if not inner*normal[2]<radial<outer:raise RingGeometryError('Measured upper rim landmark is outside the annulus')
    axis=rotate(orientation,s['jaw_axis_tool']);span=s['jaw_open_half_span_m'];finger=s['finger_half_width_m']
    tips=[[rim[0]+sign*span*axis[0],rim[1]+sign*span*axis[1]] for sign in (-1,1)]
    near,far=sorted(math.dist(v,center[:2]) for v in tips)
    # A narrow hole can still permit a diametrical grip on the two outside walls.
    # Its midpoint is the hole centre; Z still comes from the measured upper rim.
    mode=s.get('grasp_mode','auto')
    if mode not in ('auto','outer','rim'):raise ValueError('Unsupported ring grasp mode')
    outer_fits=outer*normal[2]>margin and math.hypot(axis[0],axis[1])*span-finger-margin>outer
    if mode in ('auto','outer') and outer_fits:
        grasp_xy=center[:2];mode='outer';near=far=span*math.hypot(*axis[:2])
    elif (mode in ('auto','rim') and inner+margin<radial<outer*normal[2]-margin and
          near+finger+margin<inner*normal[2] and far-finger-margin>outer):
        grasp_xy=rim[:2];mode='rim'
    else:raise RingGeometryError('Open fingers cannot straddle the outer walls or one rim wall')
    inset=profile['grasp_inset_m'];thickness=s['ring_thickness_m']
    if not vertical<inset or inset+s['fingertip_below_tcp_m']+abs(axis[2])*span+vertical>=thickness*normal[2]:
        raise RingGeometryError('Finger contact depth cannot isolate this rim above its support')
    contact=[*grasp_xy,(center[2] if mode=='outer' else rim[2])-inset]
    return contact,{'ring_id':ring['id'],'color':color(ring),'inner_radius_m':inner,'outer_radius_m':outer,
                    'rim_radius_m':radial,'grasp_mode':mode,'open_finger_center_radii_m':[near,far]}


def release_geometry(profile,scene,state,ring,peg):
    """Allow gravity to thread the ring. The held annulus and fingers stay over the peg tip."""
    s=profile['ring_peg'];inner,outer=dimension(ring);tip=measured(peg,'tip_base_m',3)
    rod=measured(peg,'radius_m');axis=unit_up(peg,'axis_base');normal=unit_up(ring,'normal_base')
    support=measured(peg,'support_z_m');center=measured(ring,'center_base_m',3)
    if rod<=0 or support>=tip[2]:raise RingGeometryError('Unverified free peg above the current ring stack')
    tcp=tcp_from_eef(state['eef_m'],state['eef_xyzw'],profile)
    vertical=s['vertical_uncertainty_m']+fit_error(ring,'depth_plane_p90_m')+fit_error(peg,'depth_plane_p90_m')
    margin=s.get('relative_xy_uncertainty_m',s['horizontal_uncertainty_m'])+fit_error(ring,'inner_boundary_p90_m')+fit_error(peg,'outer_boundary_p90_m');thick=s['ring_thickness_m']
    slope=math.hypot(axis[0],axis[1])/axis[2]; tilt=math.hypot(normal[0],normal[1]);down=outer*tilt+thick
    center_z_above_tcp=center[2]-tcp[2];tip_gap=max(s['release_gap_m']+vertical,
        s['fingertip_below_tcp_m']+center_z_above_tcp-down+vertical+s['release_gap_m'])
    drop=tip[2]+tip_gap-support
    if drop+vertical>s['max_drop_m']:raise RingGeometryError('Release exceeds measured allowed fall to stack')
    # After the rod enters the bore, it guides the fall. Only entry and passage
    # through the ring thickness require unsupported alignment at its tip.
    free_xy=inner*normal[2]-rod/axis[2]-margin-(tip_gap+thick)*slope-thick*tilt
    if free_xy<=0:raise RingGeometryError('Ring bore cannot clear the peg over the drop with measured uncertainty')
    contact=[tip[i]+tcp[i]-center[i] for i in (0,1)]+[tip[2]+tip_gap+down-center_z_above_tcp]
    evidence={'ring_id':ring['id'],'peg_id':peg['id'],'color':color(ring),'inner_radius_m':inner,
        'outer_radius_m':outer,'peg_radius_m':rod,'support_z_m':support,'ring_bottom_z_m':center[2]-down,
        'thickness_m':thick,'vertical_uncertainty_m':vertical,
        'peg_tip_base_m':tip,'ring_center_base_m':center,'free_xy_m':free_xy,'drop_m':drop,
        'stack_count_before':peg.get('stack_count'),'finger_bottom_z_m':tcp[2]-s['fingertip_below_tcp_m']}
    return contact,evidence


def resolve(profile,scene,state,skill,uv=None,*,release_peg_id=None):
    s=profile['ring_peg']
    if skill=='pick':
        ring=selected(scene,'rings',uv,('center_uv_px','rim_grasp_uv_px'),s.get('source_ring_id'))
        if ring.get('location')!='loose' or 'peg_id' not in ring or ring['peg_id'] is not None:
            raise RingGeometryError('Only confirmed loose tabletop rings may be picked')
        if state.get('gripping') is not False:raise RingGeometryError('Empty gripper required')
        matches=[p for p in scene.get('pegs',[]) if p.get('color')==color(ring)]
        if len(matches)!=1:raise RingGeometryError('Matching colour peg is missing or ambiguous')
        if s.get('destination_peg_id') not in (None,matches[0]['id']):raise RingGeometryError('Ring colour does not match configured peg')
        return ring,None
    if skill!='place' or state.get('gripping') is not True:raise RingGeometryError('Held ring and place required')
    ring=indexed(scene,'rings',scene.get('held_ring_id'))
    if ring.get('location')!='held' or s.get('source_ring_id') not in (None,ring['id']):raise RingGeometryError('Held ring identity unverified')
    peg=(indexed(scene,'pegs',release_peg_id) if release_peg_id is not None else
         selected(scene,'pegs',uv,('tip_uv_px',),s.get('destination_peg_id')))
    if color(ring)!=color(peg):raise RingGeometryError('Held ring and receiving peg colours differ')
    return ring,peg


def verify_before_open(profile,scene,state,prior):
    ring,peg=resolve(profile,scene,state,'place',release_peg_id=prior['peg_id'])
    if color(ring)!=prior['color'] or ring['id']!=prior['ring_id']:raise RingGeometryError('Held ring changed during transport')
    _,e=release_geometry(profile,scene,state,ring,peg);s=profile['ring_peg']
    gap=e['ring_bottom_z_m']-e['peg_tip_base_m'][2];error_xy=math.dist(e['ring_center_base_m'][:2],e['peg_tip_base_m'][:2])
    if gap<=e['vertical_uncertainty_m']:
        raise RingGeometryError('Held ring is too low for free lateral alignment above peg')
    if e['finger_bottom_z_m']<=e['peg_tip_base_m'][2]+e['vertical_uncertainty_m']:
        raise RingGeometryError('Fingertips are below the allowed peg release window')
    if e['ring_bottom_z_m']-e['support_z_m']+e['vertical_uncertainty_m']>s['max_drop_m']:
        raise RingGeometryError('Measured fall exceeds allowed drop')
    if error_xy>=e['free_xy_m']:
        raise RingAlignmentNeeded(*(e['peg_tip_base_m'][i]-e['ring_center_base_m'][i] for i in (0,1)),
                                  e['free_xy_m'],scene.get('frame_number'))
    return dict(e,release_xy_error_m=error_xy,release_clearance_m=gap)


def correction_suffix(profile,state,offset,remaining):
    """Take one bounded horizontal step. Replan release and vertical retreat from its endpoint."""
    tcp=tcp_from_eef(state['eef_m'],state['eef_xyzw'],profile);error=math.hypot(offset.dx,offset.dy)
    bound=profile['ring_peg'].get('servo_step_m',.008);gain=min(.8,bound/error)
    target=[tcp[0]+gain*offset.dx,tcp[1]+gain*offset.dy,tcp[2]]
    stage={'kind':'move','tcp_m':target,'xyzw':state['eef_xyzw'],'motion_phase':'placement'}
    suffix=[stage]
    for existing in remaining:
        if existing['kind']=='move':
            existing=dict(existing,tcp_m=[*target[:2],existing['tcp_m'][2]],xyzw=state['eef_xyzw'])
        suffix.append(existing)
    return suffix


def verified(action,scene,state):
    """The colour marker present before a drop cannot verify that a second ring was added."""
    if not action or scene.get('obs_id')==action['obs_id']:return None
    if action['skill']=='pick':
        if state.get('gripping') is not True:return None
        rings=[r for r in scene.get('rings',[]) if r.get('id')==scene.get('held_ring_id') and r.get('location')=='held']
        if len(rings)==1 and rings[0].get('color')==action['color']:return 'pick'
    if action['skill']=='place' and state.get('gripping') is False:
        matches=[p for p in scene.get('pegs',[]) if p.get('id')==action['peg_id'] and p.get('color')==action['color']]
        if len(matches)!=1:return None
        peg=matches[0];before=action.get('stack_count_before');after=peg.get('stack_count')
        if type(before) is int and before>=1 and type(after) is int and after==before+1:return 'place'
        if after is None or before is None:
            # A measured support rise must exclude unchanged height and two added rings.
            thick=action.get('thickness_m');err=action.get('vertical_uncertainty_m');top=peg.get('support_z_m')
            if all(type(x) in (int,float) for x in (top,thick,err)):
                rise=top-action['support_z_m']
                if 2*err<rise<2*thick-2*err and abs(rise-thick)<=2*err:return 'place'
    return None
