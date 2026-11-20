"""Ground a loose annulus and its same-colour peg using one measured RGB-D scene."""
import math

from .contract import vector
from .pi05_contract import rotate
from .primitives import number,tcp_from_eef


class RingGeometryError(ValueError):
    pass


class RingViewConflict(RingGeometryError):
    pass


class RingAlignmentNeeded(RingGeometryError):
    def __init__(self,dx,dy,allowance,frame):
        self.dx,self.dy,self.allowance,self.frame=dx,dy,allowance,frame
        super().__init__('Observed hole is offset from peg by {:.1f} mm, allowance {:.1f} mm'.format(
            1000*math.hypot(dx,dy),1000*allowance))


def validate_settings(settings):
    if 'transport' in settings:
        from .ring_transport import validate
        validate(settings['transport'])
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
    number(settings.get('tcp_table_error_m',0.),0.,.05,'residual TCP-to-local-table error')
    if 'local_pick_support' in settings:
        s=settings['local_pick_support']
        if not isinstance(s,dict) or s.get('ring_id')!=settings.get('source_ring_id') or not s.get('ring_id'):
            raise ValueError('Fingertip survey must identify the configured source ring')
        frames=s.get('frame_ids')
        if not isinstance(frames,list) or any(type(n) is not int or n<0 for n in frames) or len(set(frames))<4:
            raise ValueError('Fingertip survey requires at least four distinct frames')
        if abs(math.hypot(*vector(s['jaw_axis_base_xy'],2,'surveyed jaw axis'))-1.)>.001:
            raise ValueError('Expected a unit surveyed jaw axis')
        number(s['swept_table_upper_z_m'],-2.,2.,'surveyed swept tabletop upper height')
        nominal=number(s['ring_top_z_m'],-2.,2.,'surveyed ring top')
        number(s['ring_top_upper_z_m'],nominal,nominal+.011,'empirical upper ring-top height')
    if type(settings.get('servo_iterations',3)) is not int or not 0<=settings.get('servo_iterations',3)<=6:
        raise ValueError('Invalid ring servo iteration limit')
    if 'wrist_alignment' in settings:
        wrist=settings['wrist_alignment']
        for k in ('resolution','K','D','eef_T_optical','ring_hsv_ranges','peg_tip_hsv_ranges'):
            if k not in wrist:raise ValueError('Missing wrist alignment '+k)
        number(wrist['height_uncertainty_m'],.0001,.1,'wrist plane height uncertainty')
        number(wrist['relative_xy_uncertainty_m'],.0001,.1,'wrist alignment uncertainty')
        number(wrist['cross_camera_tolerance_m'],.001,.04,'cross-camera displacement agreement')
        number(wrist.get('max_frame_age_s',1.25),.1,2.,'maximum stationary wrist frame age')
        number(wrist.get('wait_for_frame_s',1.35),.1,2.,'wrist capture wait')


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


def surveyed_contact(settings, ring, axis, finger_depth, desired):
    """Use a paired surface/ring survey only while its original footprint is observable."""
    survey=settings['local_pick_support']
    if (survey['ring_id']!=ring['id'] or ring.get('local_support_reference_verified') is not True or
            not ring.get('local_support_reference_evidence',{}).get('verified')):
        raise RingGeometryError('Local fingertip support survey needs a new unobstructed observation')
    evidence=ring['local_support_reference_evidence']
    if list(evidence.get('survey_frame_ids',[]))!=list(survey['frame_ids']) or len(set(survey['frame_ids']))<4:
        raise RingGeometryError('Fingertip surface and observation come from different surveys')
    direction=vector(survey['jaw_axis_base_xy'],2,'surveyed finger direction')
    if 'jaw_axis_base_xy' not in evidence:raise RingGeometryError('Missing observed finger sweep direction')
    observed=vector(evidence['jaw_axis_base_xy'],2,'observed finger sweep direction')
    if math.hypot(*observed)<.99 or abs(sum(a*b for a,b in zip(observed,direction)))<.999:
        raise RingGeometryError('Observed finger sweep does not match the surface survey')
    offset=survey.get('grasp_offset_base_xy_m',[0.,0.])
    if math.dist(offset,evidence.get('grasp_offset_base_xy_m',[0.,0.]))>.0001:
        raise RingGeometryError('Observed grasp offset does not match the surveyed finger sweep')
    length=math.hypot(*direction)*math.hypot(*axis[:2])
    if length<.95 or abs(sum(a*b for a,b in zip(axis,direction)))<length*math.cos(math.radians(2)):
        raise RingGeometryError('Candidate finger direction leaves the surveyed sweep')
    shift=number(evidence['current_surface_shift_m'],-.003,.003,'live surface shift')
    # Both the tabletop and its target share the local camera-to-base error.
    # A fresh higher tabletop raises the minimum. A low reading never lowers it.
    upper=measured(survey,'swept_table_upper_z_m')+max(0.,shift)
    nominal_top=measured(survey,'ring_top_z_m')+shift
    plausible_top=measured(survey,'ring_top_upper_z_m')+shift
    if not nominal_top<=plausible_top<=nominal_top+.011:
        raise RingGeometryError('Invalid measured ring-top range')
    tracking=settings.get('tcp_table_error_m',0.)
    low=upper+finger_depth+tracking+.0005
    contact=max(nominal_top-desired,low)
    # The floor covers the lower fingertip even with the measured downward
    # tracking error. Assess a possible grasp at the *higher* fingertip. A
    # nominal TCP above the ring can still give a clean, verifiable side grip.
    highest_tip_at_lowest_endpoint=contact-float(settings['fingertip_below_tcp_m'])+abs(axis[2])*settings['jaw_open_half_span_m']-tracking
    if highest_tip_at_lowest_endpoint>=plausible_top:raise RingGeometryError('No fingertip contact depth above the local tabletop')
    return contact,{'mode':'reobserved_multiframe_support','table_upper_z_m':upper,'grasp_offset_base_xy_m':offset,
        'lowest_allowed_tcp_z_m':low,'surveyed_ring_top_z_m':nominal_top,'highest_plausible_ring_top_z_m':plausible_top,
        'highest_fingertip_at_lowest_endpoint_m':highest_tip_at_lowest_endpoint,
        'live_surface_shift_m':shift,'live_optical_inset_m':ring['center_base_m'][2]-contact,
        'surveyed_nominal_inset_m':nominal_top-contact,'empty_grasp_possible':contact>=nominal_top}


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
        survey=s.get('local_pick_support',{})
        if 'grasp_offset_base_xy_m' in survey:
            offset=vector(survey['grasp_offset_base_xy_m'],2,'surveyed tangential grip displacement')
            along=sum(v*a for v,a in zip(offset,axis))/math.hypot(*axis[:2])
            # Both opposing pads must contact the projected outer wall even with
            # a displaced centre. Only the tangential offset preserves open width.
            if abs(along)>.0002 or math.hypot(*offset)>=outer*normal[2] or max(0.,math.hypot(*offset)+margin-finger)>=outer*normal[2]:
                raise RingGeometryError('Offset finger sweep cannot grip opposing ring walls')
            grasp_xy=[a+b for a,b in zip(grasp_xy,offset)]
            near,far=sorted(math.hypot(offset[0]+sign*span*axis[0],offset[1]+sign*span*axis[1]) for sign in (-1,1))
    elif (mode in ('auto','rim') and inner+margin<radial<outer*normal[2]-margin and
          near+finger+margin<inner*normal[2] and far-finger-margin>outer):
        grasp_xy=rim[:2];mode='rim'
    else:raise RingGeometryError('Open fingers cannot straddle the outer walls or one rim wall')
    desired=profile['grasp_inset_m'];thickness=s['ring_thickness_m']
    # Keep the lowest open fingertip above the support. Failure to guarantee
    # a bite at the high end of the pose error is assessed by the grasp sensor.
    top=(center[2] if mode=='outer' else rim[2]);finger_depth=s['fingertip_below_tcp_m']+abs(axis[2])*span
    survey_evidence=None
    if s.get('local_pick_support') is not None:
        if mode!='outer':raise RingGeometryError('Surveyed finger sweep requires an outer-wall grasp')
        contact_z,survey_evidence=surveyed_contact(s,ring,axis,finger_depth,desired)
        inset=survey_evidence['surveyed_nominal_inset_m']
    elif ring.get('local_table_upper_z_m') is not None:
        upper=measured(ring,'local_table_upper_z_m');mid=measured(ring,'local_table_z_m')
        if mid>upper or type(ring.get('local_table_depth_count')) is not int or ring['local_table_depth_count']<=0:
            raise RingGeometryError('Invalid measured local tabletop interval')
        available=top-upper-finger_depth-s.get('tcp_table_error_m',0.)
    else:available=thickness*normal[2]-finger_depth-vertical
    if survey_evidence is None:
        inset=min(desired,available-.0005)
        if inset<=0:raise RingGeometryError('No fingertip contact depth above the local tabletop')
        contact_z=top-inset
    contact=[*grasp_xy,contact_z]
    return contact,{'ring_id':ring['id'],'color':color(ring),'inner_radius_m':inner,'outer_radius_m':outer,
                    'rim_radius_m':radial,'grasp_mode':mode,'open_finger_center_radii_m':[near,far],
                    'requested_grasp_inset_m':desired,'effective_grasp_inset_m':inset,'vertical_uncertainty_m':vertical,
                    'local_table_upper_z_m':survey_evidence['table_upper_z_m'] if survey_evidence else ring.get('local_table_upper_z_m'),
                    'support_survey':survey_evidence,'tcp_table_error_m':s.get('tcp_table_error_m',0.)}


def release_geometry(profile,scene,state,ring,peg):
    """Allow gravity to thread the ring. The held annulus and fingers stay over the peg tip."""
    s=profile['ring_peg'];inner,outer=dimension(ring);tip=measured(peg,'tip_base_m',3)
    rod=measured(peg,'radius_m');axis=unit_up(peg,'axis_base');normal=unit_up(ring,'normal_base')
    support=measured(peg,'support_z_m');center=measured(ring,'center_base_m',3)
    if rod<=0 or support>=tip[2]:raise RingGeometryError('Unverified free peg above the current ring stack')
    tcp=tcp_from_eef(state['eef_m'],state['eef_xyzw'],profile)
    vertical=s['vertical_uncertainty_m']+fit_error(ring,'depth_plane_p90_m')+fit_error(peg,'depth_plane_p90_m')
    wrist=scene.get('wrist_alignment');thick=s['ring_thickness_m']
    if wrist is not None:
        inner=measured(wrist,'inner_radius_m');rod=measured(wrist,'peg_radius_m');optical_error=measured(wrist,'uncertainty_m')
        margin=optical_error+measured(wrist,'inner_radius_error_m')+measured(wrist,'peg_radius_error_m')
    else:
        optical_error=s.get('relative_xy_uncertainty_m',s['horizontal_uncertainty_m'])
        margin=optical_error+fit_error(ring,'inner_boundary_p90_m')+fit_error(peg,'outer_boundary_p90_m')
    slope=math.hypot(axis[0],axis[1])/axis[2]; tilt=math.hypot(normal[0],normal[1]);down=outer*tilt+thick
    center_z_above_tcp=center[2]-tcp[2];tip_gap=max(s['release_gap_m']+vertical,
        s['fingertip_below_tcp_m']+center_z_above_tcp-down+vertical+s['release_gap_m'])
    drop=tip[2]+tip_gap-support
    if drop+vertical>s['max_drop_m']:raise RingGeometryError('Release exceeds measured allowed fall to stack')
    # After the rod enters the bore, it guides the fall. Only entry and passage
    # through the ring thickness require unsupported alignment at its tip.
    free_xy=inner*normal[2]-rod/axis[2]-margin-(tip_gap+thick)*slope-thick*tilt
    matched=s.get('matching_parts_verified') is True
    # A verified toy set permits a gravity-only attempt from above the peg. A
    # small clearance below image resolution affects success, not tool contact.
    # A resolved gross size/tilt mismatch still means the observed object is wrong.
    if free_xy<=0 and not (matched and rod/axis[2]-inner*normal[2]<=margin):
        raise RingGeometryError('Ring bore cannot clear the peg at entry with measured uncertainty')
    offset=vector(wrist['offset_xy_m'],2,'wrist hole-to-tip offset') if wrist else [tip[i]-center[i] for i in (0,1)]
    contact=[tcp[i]+offset[i] for i in (0,1)]+[tip[2]+tip_gap+down-center_z_above_tcp]
    evidence={'ring_id':ring['id'],'peg_id':peg['id'],'color':color(ring),'inner_radius_m':inner,
        'outer_radius_m':outer,'peg_radius_m':rod,'support_z_m':support,'ring_bottom_z_m':center[2]-down,
        'thickness_m':thick,'vertical_uncertainty_m':vertical,
        'peg_tip_base_m':tip,'ring_center_base_m':center,'free_xy_m':free_xy,'drop_m':drop,
        'alignment_sensor':'wrist' if wrist else 'D455','hole_to_tip_xy_m':offset,'matching_parts_verified':matched,
        'centering_goal_m':min(.001,inner*normal[2]*.25,max(.00035,optical_error*.5)),
        'attempt_max_offset_m':min(.002,inner*normal[2]*.45),
        'stack_count_before':peg.get('stack_count'),'stack_depth_delta_before_m':peg.get('stack_depth_delta_m'),
        'stack_delta_from_frame':peg.get('stack_delta_from_frame'),'finger_bottom_z_m':tcp[2]-s['fingertip_below_tcp_m']}
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


def verify_before_open(profile,scene,state,prior,*,servo_budget_spent=False):
    ring,peg=resolve(profile,scene,state,'place',release_peg_id=prior['peg_id'])
    if color(ring)!=prior['color'] or ring['id']!=prior['ring_id']:raise RingGeometryError('Held ring changed during transport')
    _,e=release_geometry(profile,scene,state,ring,peg);s=profile['ring_peg']
    gap=e['ring_bottom_z_m']-e['peg_tip_base_m'][2];error_xy=math.hypot(*e['hole_to_tip_xy_m'])
    if gap<=e['vertical_uncertainty_m']:
        raise RingGeometryError('Held ring is too low for free lateral alignment above peg')
    if e['finger_bottom_z_m']<=e['peg_tip_base_m'][2]+e['vertical_uncertainty_m']:
        raise RingGeometryError('Fingertips are below the allowed peg release window')
    if e['ring_bottom_z_m']-e['support_z_m']+e['vertical_uncertainty_m']>s['max_drop_m']:
        raise RingGeometryError('Measured fall exceeds allowed drop')
    clear=e['free_xy_m']>0 and error_xy<e['free_xy_m']
    allowance=e['free_xy_m']
    if e['matching_parts_verified']:
        allowance=max(allowance,e['attempt_max_offset_m'] if servo_budget_spent else e['centering_goal_m'])
    if not clear and error_xy>allowance:
        raise RingAlignmentNeeded(*e['hole_to_tip_xy_m'],allowance,scene.get('frame_number'))
    return dict(e,release_xy_error_m=error_xy,release_clearance_m=gap,
                release_basis='resolved_clearance' if clear else 'centered_matching_parts_gravity_attempt')


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
    if (action['skill']=='move_to' and scene.get('held_transport',{}).get('at_hover') is True and
        scene['held_transport'].get('peg_id')==action.get('peg_id') and scene.get('held_ring_id')==action.get('ring_id') and
        state.get('gripping') is True):return 'move_to'
    if action['skill']=='pick':
        if state.get('gripping') is not True:return None
        rings=[r for r in scene.get('rings',[]) if r.get('id')==scene.get('held_ring_id') and r.get('location')=='held']
        if len(rings)==1 and rings[0].get('color')==action['color'] and rings[0]['id']==action['ring_id']:return 'pick'
    if action['skill']=='place' and state.get('gripping') is False:
        if any(r.get('id')==action['ring_id'] and r.get('location')=='loose' for r in scene.get('rings',[])):return None
        matches=[p for p in scene.get('pegs',[]) if p.get('id')==action['peg_id'] and p.get('color')==action['color']]
        if len(matches)!=1:return None
        peg=matches[0];before=action.get('stack_count_before');after=peg.get('stack_count')
        if type(before) is int and before>=1 and type(after) is int and after==before+1:return 'place'
        start_delta=action.get('stack_depth_delta_before_m');end_delta=peg.get('stack_depth_delta_m')
        if (peg.get('stack_rise_supported') is True and action.get('stack_delta_from_frame') is not None and
                action['stack_delta_from_frame']==peg.get('stack_delta_from_frame') and
                type(start_delta) in (float,int) and type(end_delta) in (float,int)):
            difference=end_delta-start_delta;thick=action['thickness_m']
            if .55*thick<=difference<=1.45*thick:return 'place'
        if after is None or before is None:
            # A measured support rise must exclude unchanged height and two added rings.
            thick=action.get('thickness_m');err=action.get('vertical_uncertainty_m')
            top=peg.get('current_support_z_m',peg.get('support_z_m') if peg.get('current_tip_depth_measured') is not False else None)
            if all(type(x) in (int,float) for x in (top,thick,err)):
                rise=top-action['support_z_m']
                if 2*err<rise<2*thick-2*err and abs(rise-thick)<=2*err:return 'place'
    return None


def missed_ring(action,scene,state):
    if not action or action.get('skill')!='place' or scene.get('obs_id')==action['obs_id'] or state.get('gripping') is not False:return None
    loose=[r for r in scene.get('rings',[]) if r.get('id')==action['ring_id'] and r.get('color')==action['color']
           and r.get('location')=='loose' and 'peg_id' in r and r['peg_id'] is None]
    if len(loose)==1 and scene.get('verified_skill')!='place':
        return {'ring_id':action['ring_id'],'color':action['color'],'peg_id':action['peg_id'],
                'center_uv_px':loose[0]['center_uv_px'],'obs_id':scene['obs_id']}
    return None
