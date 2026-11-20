"""Offline candidate profiles from a measured tabletop; never commission hardware."""
import argparse
import copy
import json
import math
from pathlib import Path

from .contract import vector
from .pi05_contract import quaternion
from .primitives import table_height, validate_profile, workspace_polygon
from .trajectory import motion_limits


def build_candidate_profiles(template,geometry,tcp_fits=None,*,transit_speed=None,contact_speed=None,
                             grasp_symmetry=None):
    """Return independent left/right primitive configs with all enable flags false.

    Geometry is an offline survey artifact, not proof of calibration or collision
    safety. The right axial TCP is a rough prior contact estimate; its full XYZ
    transform, orientation and current tool geometry still require validation.
    """
    plane=geometry['plane']
    polygon=[vector(v,3,'survey boundary vertex')[:2] for v in geometry['interpreted_boundary_base_m']]
    q=quaternion(geometry['camera_to_base_xyzw'])
    translation=vector(geometry['camera_to_base_translation_m'],3,'camera translation')
    tcp_fits={} if tcp_fits is None else tcp_fits
    if not isinstance(tcp_fits,dict) or any(arm not in ('left','right') for arm in tcp_fits):
        raise ValueError('TCP fits must be keyed by left/right arm')
    if grasp_symmetry not in (None, 'square', 'fixed'):
        raise ValueError('Choose square or fixed grasp symmetry')
    output={}
    for arm in ('left','right'):
        cfg=copy.deepcopy(template);p=cfg['primitives']
        cfg.update(arm=arm,commissioned=False,commissioning_scope='supervised_primitive_task')
        # Apply to the new, uncommissioned profiles only. Reusing a measured
        # template must not silently increase any existing stage speed.
        original_speed,_ = motion_limits(cfg)
        phases = cfg.setdefault('joint_speed_by_phase_rad_s', {})
        limit = original_speed if transit_speed is None else transit_speed
        if not math.isfinite(limit) or not .01 <= limit <= .12:
            raise ValueError('Transit speed must be in [.01, .12] rad/s')
        if transit_speed is not None:
            for phase in ('orientation','transit','clearance','retreat'):
                phases[phase] = transit_speed
        cfg['joint_speed_rad_s'] = limit
        if contact_speed is None:
            contact_speed_for_arm = min(phases.get('contact', .03), original_speed, limit)
        else:
            contact_speed_for_arm = contact_speed
        phases['contact'] = contact_speed_for_arm
        # Pick uses this phase only after a verified straight departure leg.
        # Carry any explicit loaded-motion cap across from older templates.
        phases['lift_clear'] = min(limit, phases.get('lift_clear', phases.get('lift', phases.get('transit', limit))))
        phases['lift'] = min(phases.get('lift', contact_speed_for_arm), contact_speed_for_arm)
        motion_limits(cfg)
        planning = cfg.setdefault('planning', {})
        planning['trajectory_mode'] = 'continuous'
        square_yaws = [0., math.pi / 2, -math.pi / 2, math.pi]
        if grasp_symmetry is None:
            planning.setdefault('grasp_yaw_offsets_rad', square_yaws)
        else:
            planning['grasp_yaw_offsets_rad'] = square_yaws if grasp_symmetry == 'square' else [0.]
        p['approach_path'] = 'direct_hover'
        p['contact_band_m'] = min(.04, p['approach_m'])
        p.update(calibrated=False,table_plane={'normal':list(plane['normal']),'offset':plane['offset']},
                 table_polygon_base_xy=copy.deepcopy(polygon),camera_to_base_xyzw=list(q),
                 camera_to_base_translation_m=list(translation),observation_tcp_m=[])
        if arm=='right':
            p['tcp_offset_tool_m']=[0.,0.,.0155]
            p['tool_orientation_xyzw']=[1.,0.,0.,0.]
        fit=tcp_fits.get(arm)
        if fit is not None:
            if not isinstance(fit,dict) or fit.get('arm')!=arm or fit.get('method')!='fixed_tip_pivot':
                raise ValueError('TCP fit arm/method mismatch')
            tcp=vector(fit['tcp_offset_tool_m'],3,'TCP fit offset')
            if math.sqrt(sum(v*v for v in tcp))>.4:raise ValueError('TCP fit exceeds plausible tool size')
            if type(fit.get('samples')) is not int or fit['samples']<4:raise ValueError('Insufficient TCP fit samples')
            for key,limit in (('rms_m',.005),('max_error_m',.01)):
                value=fit.get(key)
                if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=limit:
                    raise ValueError('TCP fit residual exceeds accepted limit')
            p['tcp_offset_tool_m']=tcp
        # XY is the surveyed footprint, not a rectangular assertion of reach.
        # The polygon gate remains authoritative inside this enclosing box.
        low_xy=[min(v[i] for v in polygon) for i in (0,1)]
        high_xy=[max(v[i] for v in polygon) for i in (0,1)]
        low_z=min(table_height(x,y,p) for x,y in polygon)+p['table_clearance_m']
        surface_max=max(table_height(x,y,p) for x,y in polygon)
        high_z=surface_max+.35
        p['tcp_workspace_min_m']=low_xy+[low_z]
        p['tcp_workspace_max_m']=high_xy+[high_z]
        validate_polygon=workspace_polygon(p)
        center=[sum(v[i] for v in validate_polygon)/len(validate_polygon) for i in (0,1)]
        edge=max(validate_polygon,key=lambda v:v[1]) if arm=='left' else min(validate_polygon,key=lambda v:v[1])
        candidate_xy=[.75*a+.25*b for a,b in zip(center,edge)]
        p['transit_z_m']=surface_max+p['table_clearance_m']+p['approach_m']+p['held_bottom_below_tcp_m']
        suggested_home=candidate_xy+[max(p['transit_z_m'],surface_max+.25)]
        p['home_tcp_m']=None
        p.pop('legacy_anchor_reference',None)
        radius=sum(v*v for v in p['tcp_offset_tool_m'])**.5
        cfg['workspace_min_m']=[v-radius for v in p['tcp_workspace_min_m']]
        cfg['workspace_max_m']=[v+radius for v in p['tcp_workspace_max_m']]
        cfg['site_candidate']={
            'status':'offline_candidate_not_commissioned','arm':arm,
            'table_boundary_verified':False,'table_plane_verified':False,'tcp_verified':False,
            'ik_paths_verified':False,'inter_arm_clearance_verified':False,
            'tcp_source':'prior left axial estimate' if arm=='left' else 'rough right axial estimate 15.5 mm; full XYZ pending',
            'home_source':'convex interior candidate; not an observed or IK-validated parking pose',
            'suggested_home_tcp_m':suggested_home,
            'survey_source':str(geometry.get('source','supplied offline geometry')),
            'required_checks':['Current camera extrinsics and table plane','Physical table perimeter and obstacles',
                               'Full installed-tool TCP transform','Arm-specific complete paths and parking poses',
                               'Inter-arm link/tool/held-object collision geometry'],
        }
        if fit is not None:
            cfg['site_candidate']['tcp_source']='fixed-tip pivot fit; independent field check still pending'
            cfg['site_candidate']['tcp_fit']=copy.deepcopy(fit)
        p['calibration_evidence']={'status':'fulltable_candidate_unverified',
                                   'survey_source':cfg['site_candidate']['survey_source'],
                                   'tcp_source':cfg['site_candidate']['tcp_source']}
        cfg.pop('motion_audit_path',None)
        p['notes']='OFFLINE FULL-TABLE CANDIDATE ONLY. Plane and footprint require onsite confirmation; TCP, paths and collision geometry are not commissioned.'
        validate_profile(p)
        output[arm]=cfg
    return output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template',type=Path,required=True)
    parser.add_argument('--geometry',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--left-tcp-fit',type=Path)
    parser.add_argument('--right-tcp-fit',type=Path)
    parser.add_argument('--motion-budget',type=int,default=64)
    parser.add_argument('--lease-seconds',type=float,default=600)
    parser.add_argument('--transit-speed-rad-s',type=float,
                        help='Free-motion speed for the new candidate. Default preserves template limits.')
    parser.add_argument('--contact-speed-rad-s',type=float,
                        help='Final contact speed. Default preserves the template contact cap, or uses up to 0.03 rad/s.')
    parser.add_argument('--grasp-symmetry',choices=('square','fixed'),
                        help='Square tests four quarter-turn yaws for cube tasks. Fixed keeps the configured tool orientation. Default uses the template yaw candidates, or square if absent.')
    args=parser.parse_args(argv)
    fits={arm:json.loads(path.read_text(encoding='utf-8')) for arm in ('left','right')
          if (path:=getattr(args,arm+'_tcp_fit')) is not None}
    profiles=build_candidate_profiles(json.loads(args.template.read_text(encoding='utf-8')),
                                      json.loads(args.geometry.read_text(encoding='utf-8')),fits,
                                      transit_speed=args.transit_speed_rad_s,contact_speed=args.contact_speed_rad_s,
                                      grasp_symmetry=args.grasp_symmetry)
    from .dual_bridge import build_dual_candidate
    dual=build_dual_candidate(profiles['left'],profiles['right'],args.motion_budget,args.lease_seconds)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for arm,cfg in dual['arms'].items():
        path=args.output_dir/(arm+'-fulltable.local.json')
        path.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(str(path))
    path=args.output_dir/'baxter-dual.local.json'
    path.write_text(json.dumps(dual,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(str(path))


if __name__=='__main__':main()
