"""Capture stationary eye-on-base samples from a D455 and Baxter's right endpoint (read-only)."""
import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import uuid

import numpy as np

from .pi05_contract import quaternion, rotate


def refine_tag_edges(gray, corners):
    """Fit the black outer border using outward black-to-white gradients on each decoded side."""
    corners=np.asarray(corners,dtype=float).reshape(4,2)
    center=corners.mean(axis=0)
    lines=[]; diagnostics=[]
    for a,b in zip(corners,np.roll(corners,-1,axis=0)):
        length=np.linalg.norm(b-a)
        if length<12:raise ValueError('Tag edge too short in pixels')
        normal=np.array([a[1]-b[1],b[0]-a[0]])/length
        if normal @ (center-a)>0:normal=-normal
        base=a+(b-a)*np.linspace(.12,.88,60)[:,None]
        offset=np.arange(-5.,5.01,.125)
        xy=base[:,None,:]+offset[None,:,None]*normal
        x,y=xy[:,:,0],xy[:,:,1]
        if np.min(x)<0 or np.min(y)<0 or np.max(x)>=gray.shape[1]-1 or np.max(y)>=gray.shape[0]-1:
            raise ValueError('Tag border profiles cross image edge')
        i,j=x.astype(int),y.astype(int)
        dx,dy=x-i,y-j
        values=((1-dx)*(1-dy)*gray[j,i]+dx*(1-dy)*gray[j,i+1]+
                (1-dx)*dy*gray[j+1,i]+dx*dy*gray[j+1,i+1])
        gradient=np.gradient(values,.125,axis=1)
        peak=np.argmax(gradient[:,2:-2],axis=1)+2
        good=gradient[np.arange(len(peak)),peak]>12
        if good.sum()<15:
            raise ValueError('Insufficient black-to-white edge contrast')
        points=base[good]+offset[peak[good],None]*normal
        # Huber-weighted orthogonal line fit. Keep the raw fit error for later review.
        weights=np.ones(len(points))
        for _ in range(8):
            mid=np.average(points,axis=0,weights=weights)
            _,_,vh=np.linalg.svd((points-mid)*np.sqrt(weights[:,None]),full_matrices=False)
            n=vh[-1];errors=(points-mid) @ n
            weights=np.minimum(1.,.6/np.maximum(abs(errors),1e-9))
        if (abs(errors)<1.).sum()<.7*len(points):
            raise ValueError('Inconsistent outer border line')
        lines.append(np.r_[n,-n@mid])
        diagnostics.append({'samples':len(points),'rms_px':float(np.sqrt(np.mean(errors**2)))})
    result=[]
    for i in range(4):
        cross=np.cross(lines[i-1],lines[i])
        if abs(cross[2])<.15:raise ValueError('Unstable border intersection')
        result.append(cross[:2]/cross[2])
    result=np.asarray(result)
    if np.max(np.linalg.norm(result-corners,axis=1))>8.:
        raise ValueError('Border intersection too far from decoded corner')
    return result,diagnostics


def intrinsics_data(intr):
    return dict({key:getattr(intr,key) for key in ('width','height','fx','fy','ppx','ppy','coeffs')},
                model=str(intr.model))


def tag_from_depth(corners, depth_m, ray, width_m=None, height_m=None):
    """Canonical ArUco corner ordering; ray(u,v) deprojects optical Z=1 using camera calibration."""
    corners = np.asarray(corners, dtype=float).reshape(4, 2)
    height, width = depth_m.shape
    if (not np.isfinite(corners).all() or (corners < 0).any() or
            (corners[:, 0] >= width).any() or (corners[:, 1] >= height).any()):
        raise ValueError('Tag corners outside aligned depth')
    # Sample inside the black boundary. Keep enough coverage to avoid fitting
    # the background or a small isolated object behind a depth hole.
    fractions = np.linspace(.14, .86, 13)
    u, v = np.meshgrid(fractions, fractions)
    u, v = u.reshape(-1, 1), v.reshape(-1, 1)
    pixels = ((1-u)*(1-v)*corners[0] + u*(1-v)*corners[1] +
              u*v*corners[2] + (1-u)*v*corners[3])
    xy = np.unique(np.rint(pixels).astype(int), axis=0)
    depths = depth_m[xy[:, 1], xy[:, 0]]
    valid = np.isfinite(depths) & (depths > .1) & (depths < 3.)
    if len(xy)<64 or valid.sum()<.65*len(xy):
        raise ValueError('Insufficient valid aligned depth across tag')
    rays = np.array([ray(p.tolist()) for p in xy[valid]])
    points = rays * depths[valid, None]
    triplets = np.random.default_rng(281).integers(0, len(points), (72, 3))
    a, b, c = points[triplets].transpose(1, 0, 2)
    normals = np.cross(b-a, c-a)
    lengths = np.linalg.norm(normals, axis=1)
    usable = lengths > 1e-6
    if not usable.any():
        raise ValueError('Degenerate depth samples')
    normals = normals[usable] / lengths[usable, None]
    offsets = (a[usable] * normals).sum(axis=1)
    inliers = np.abs(points @ normals.T - offsets) <= .004
    keep = inliers[:, np.argmax(inliers.sum(axis=0))]
    if keep.sum()<.65*len(xy):
        raise ValueError('No dominant tag depth plane')
    cloud = points[keep]
    center = cloud.mean(axis=0)
    _, singular, vh = np.linalg.svd(cloud-center, full_matrices=False)
    if singular[1] < .015:
        raise ValueError('Insufficient spread of tag depth inliers')
    normal = vh[-1]
    if normal[2]<0:
        normal = -normal
    residuals = (cloud-center) @ normal
    corner_rays = np.array([ray(p.tolist()) for p in corners])
    denom = corner_rays @ normal
    if np.min(denom) < .18:
        raise ValueError('Tag too oblique for depth localization')
    xyz = corner_rays * ((center @ normal)/denom)[:, None]
    axis_x = (xyz[1]-xyz[0]+xyz[2]-xyz[3])/2
    axis_y = (xyz[3]-xyz[0]+xyz[2]-xyz[1])/2
    sides = np.linalg.norm(np.roll(xyz,-1,axis=0)-xyz,axis=1)
    w, h = float((sides[0]+sides[2])/2), float((sides[1]+sides[3])/2)
    if min(w,h)<.01 or max(w,h)>.7:
        raise ValueError('Implausible measured marker dimensions')
    if max(abs(sides[0]-sides[2]),abs(sides[1]-sides[3])) > max(.004,.06*max(w,h)):
        raise ValueError('Opposite marker edges disagree')
    x = axis_x/np.linalg.norm(axis_x)
    y = axis_y/np.linalg.norm(axis_y)
    if abs(x @ y)>.1:
        raise ValueError('Measured marker edges are not perpendicular')
    for measured, reference, name in ((w,width_m,'width'),(h,height_m,'height')):
        if reference is not None and abs(measured-reference)>max(.005,.04*reference):
            raise ValueError('Depth {} {:.4f} m disagrees with ruler {:.4f} m'.format(name,measured,reference))
    z = np.cross(x,y)
    z /= np.linalg.norm(z)
    transform = np.eye(4)
    transform[:3,:3] = np.column_stack((x,np.cross(z,x),z))
    transform[:3,3] = xyz.mean(axis=0)
    limits = {'width_m':width_m,'height_m':height_m}
    return {'camera_T_tag':transform.tolist(),'corners_px':corners.tolist(),
            'corners_optical_m':xyz.tolist(),'width_m':w,'height_m':h,
            'ruler_reference':limits,'plane_inliers':len(cloud),'sampled_pixels':len(xy),
            'plane_rms_m':float(np.sqrt(np.mean(residuals**2))),
            'plane_max_error_m':float(np.max(np.abs(residuals))),
            'plane_normal':normal.tolist(),'plane_offset_m':float(-normal @ center)}


def pose_matrix(pos, ori):
    q = quaternion([ori.x,ori.y,ori.z,ori.w])
    frame = np.eye(4)
    frame[:3,:3] = np.array([rotate(q, axis) for axis in np.eye(3)]).T
    frame[:3,3] = [pos.x,pos.y,pos.z]
    if not np.isfinite(frame).all():
        raise ValueError('Non-finite endpoint')
    return frame


def pose_difference(a, b):
    a, b = np.asarray(a), np.asarray(b)
    cosine = (np.trace(a[:3,:3].T @ b[:3,:3])-1)/2
    return float(np.linalg.norm(a[:3,3]-b[:3,3])), math.acos(float(np.clip(cosine,-1.,1.)))


class EndpointReader:
    def __init__(self, rospy):
        from baxter_core_msgs.msg import EndpointState
        from sensor_msgs.msg import JointState
        import tf2_ros
        from .kinematics import ContinuousIK
        self.fk = ContinuousIK('right')
        self.rospy=rospy
        self.tf=tf2_ros.Buffer()
        self.tf_listener=tf2_ros.TransformListener(self.tf)
        self.tf_errors=(tf2_ros.LookupException,tf2_ros.ConnectivityException,tf2_ros.ExtrapolationException)
        self.lock = threading.Lock()
        self.buffers = {name:deque(maxlen=128) for name in ('endpoint','joints')}
        self.subs = [
            rospy.Subscriber('/robot/limb/right/endpoint_state',EndpointState,self.receive,callback_args='endpoint',queue_size=1),
            rospy.Subscriber('/robot/joint_states',JointState,self.receive,callback_args='joints',queue_size=16)]
    def receive(self, msg, name):
        arrival = time.monotonic()
        if name=='joints' and (len(msg.name)!=len(msg.position) or not set(self.fk.names).issubset(msg.name)):
            return
        with self.lock:
            self.buffers[name].append((arrival,msg))
    def sample(self, arrival):
        readings = {}
        with self.lock:
            for name,buffer in self.buffers.items():
                if not buffer:
                    raise ValueError('Waiting for right '+name)
                readings[name]=min(buffer,key=lambda item:abs(item[0]-arrival))
            recent_joints=list(self.buffers['joints'])
        ep_t, ep = readings['endpoint']
        joint_t, joints = readings['joints']
        ep_stamp=ep.header.stamp.to_sec()
        skipped=[]
        # At a joint limit the encoder can alternate either side. Only replace
        # a rejected sample with genuinely measured, time-compatible feedback.
        for t,msg in sorted(recent_joints,key=lambda item:abs(item[0]-arrival)):
            if abs(t-arrival)>.12:break
            if abs(msg.header.stamp.to_sec()-ep_stamp)>.15:continue
            observed=dict(zip(msg.name,msg.position))
            bad={n:observed[n] for i,n in enumerate(self.fk.names)
                 if not self.fk.lower[i]<=observed[n]<=self.fk.upper[i]}
            if not bad:
                joint_t,joints=t,msg
                break
            skipped.append({'stamp_s':msg.header.stamp.to_sec(),'outside_rad':bad})
        skew = max(abs(arrival-ep_t),abs(arrival-joint_t))
        stamps = [m.header.stamp.to_sec() for m in (ep,joints)]
        if skew>.12 or min(stamps)<=0 or abs(stamps[0]-stamps[1])>.15:
            raise ValueError('Endpoint, joints and RGB-D timestamps do not match')
        if ep.header.frame_id.lstrip('/') not in ('','base'):
            raise ValueError('Unexpected endpoint frame '+ep.header.frame_id)
        positions=dict(zip(joints.name,joints.position))
        if not all(name in positions for name in self.fk.names):
            raise ValueError('Incomplete right-arm joint feedback')
        right={name:positions[name] for name in self.fk.names}
        if not all(math.isfinite(v) for v in right.values()):raise ValueError('Nonfinite measured right joints')
        outside={name:{'rad':right[name],'lower_rad':self.fk.lower[i],'upper_rad':self.fk.upper[i]}
                 for i,name in enumerate(self.fk.names) if not self.fk.lower[i]<=right[name]<=self.fk.upper[i]}
        matrix=pose_matrix(ep.pose.position,ep.pose.orientation)
        verification={'joint_limit_violations':outside,'skipped_joint_samples':skipped}
        if outside:
            # FK deliberately refuses out-of-model joints. Verify this read-only
            # calibration pose from fresh TF; it cannot acknowledge a drive gate.
            try:
                ref=self.tf.lookup_transform('base','right_gripper',self.rospy.Time(0))
                dt=ref.header.stamp.to_sec()-stamps[0]
                if ref.header.stamp.to_sec()<=0 or abs(dt)>.15:
                    raise ValueError('stale TF (stamp difference {:.3f}s)'.format(dt))
            except self.tf_errors+(ValueError,) as error:
                raise ValueError('Right joints outside URDF '+str(outside)+'; no fresh independent TF: '+str(error)) from error
            reference=pose_matrix(ref.transform.translation,ref.transform.rotation)
            verification.update(pose_check='tf_readonly',tf_stamp_s=ref.header.stamp.to_sec())
        else:
            fk=self.fk.forward(right)
            reference=pose_matrix(fk['position'],fk['orientation'])
            verification.update(pose_check='urdf_fk')
        distance,angle=pose_difference(matrix,reference)
        if distance>.006 or angle>.025:
            raise ValueError('SDK to right_gripper {} disagreement {:.4f} m / {:.2f} deg; joints outside URDF {}'.format(
                verification['pose_check'],distance,math.degrees(angle),outside))
        return {'base_T_effector':matrix.tolist(),'endpoint_ros_stamp_s':stamps[0],
                'joint_ros_stamp_s':stamps[1],'endpoint_header_frame':ep.header.frame_id,
                'max_host_arrival_skew_s':skew,'right_joint_rad':right,
                'reference_error_m':distance,'reference_error_rad':angle,
                'fk_error_m':None if outside else distance,'fk_error_rad':None if outside else angle,**verification}


def stationary(window):
    return window_status(window) is None


def window_status(window):
    if len(window)<4 or window[-1]['host_time_s']-window[0]['host_time_s']<.5:
        span=window[-1]['host_time_s']-window[0]['host_time_s'] if window else 0.
        return 'Collecting stationary samples ({:d} frames, {:.3f}s)'.format(len(window),span)
    for key in ('endpoint_ros_stamp_s','joint_ros_stamp_s','color_stamp_ms','depth_stamp_ms'):
        stamps=[frame[key] for frame in window]
        if any(b<a for a,b in zip(stamps,stamps[1:])) or len(set(stamps))<4:
            return 'Stale or out-of-order '+key
    first=window[0]
    for frame in window:
        for key,d,a in [('base_T_effector',.003,.02),('camera_T_tag',.007,.06)]:
            distance,angle=pose_difference(frame[key],first[key])
            if distance>d or angle>a:
                return 'Unstable {} over window ({:.2f} mm, {:.2f} deg)'.format(key,distance*1000,math.degrees(angle))
    return None


def gate_pose(path, measured):
    """Validate an atomic target written by the separate supervised pose driver."""
    try:
        value=json.loads(path.read_text(encoding='utf-8'))
        name=value['id']; pose=value['base_T_effector']
        if not isinstance(name,str) or not name.strip():
            raise ValueError('Missing pose ID')
        xyz,xyzw=pose['xyz'],pose['xyzw']
        if len(xyz)!=3 or len(xyzw)!=4:raise ValueError('Wrong pose length')
        goal=pose_matrix(SimpleNamespace(**dict(zip('xyz',xyz))),
                         SimpleNamespace(**dict(zip('xyzw',xyzw))))
    except (OSError,KeyError,ValueError,TypeError,AttributeError) as error:
        raise ValueError('Waiting for valid pose gate ({})'.format(type(error).__name__)) from error
    d,a=pose_difference(goal,measured)
    if d>.005 or a>math.radians(2):
        raise ValueError('Waiting for target {} ({:.1f} mm, {:.2f} deg)'.format(name,1000*d,math.degrees(a)))
    return name,goal


def duplicate_observation(pose, accepted):
    """Describe the closest already recorded pose within the capture novelty limits."""
    duplicates=[]
    for idx,prior in enumerate(accepted,1):
        distance,angle=pose_difference(pose,prior)
        if distance<.035 and angle<math.radians(12):
            duplicates.append((max(distance/.035,angle/math.radians(12)),idx,distance,angle))
    if duplicates:
        _,idx,distance,angle=min(duplicates)
        return {'nearest_pose_index':idx-1,'nearest_frame_id':'pose-{:03}'.format(idx),'distance_mm':round(1000*distance,3),
                'angle_deg':round(math.degrees(angle),3)}
    return None


def write_json(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,allow_nan=False,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


def volume_capture_plan(path):
    from .handeye import transform
    raw = path.read_bytes()
    poses = {}
    for entry in json.loads(raw)['poses']:
        key = entry['id']
        if not isinstance(key, str) or not key or key in poses or entry['split'] not in ('train','holdout'):
            raise ValueError('Invalid or duplicate volume pose ID or split')
        poses[key] = (entry['split'], transform(entry['base_T_effector']), entry['cell'])
    counts = {s: sum(v[0] == s for v in poses.values()) for s in ('train','holdout')}
    if counts['train'] < 24 or counts['holdout'] < 6:
        raise ValueError('Formal capture requires at least 24 training and six held-out locations')
    return poses, hashlib.sha256(raw).hexdigest(), counts


def load_volume_checkpoint(out, manifest, poses):
    old = json.loads((out/'manifest.json').read_text(encoding='utf8'))
    for k,v in manifest.items():
        if k in ('intrinsics','native_depth_intrinsics') and k in old:
            verify_optical_profile(old[k],v)
        elif old.get(k) != v:
            raise ValueError('Capture plan, camera, stream calibration or ruler reference changed: '+k)
    sets = {s: json.loads((out/(s+'.json')).read_text(encoding='utf8')) for s in ('train','holdout')}
    accepted = set()
    frames = set()
    for split, group in sets.items():
        for sample in group:
            key = sample['drive_pose_id']
            frame = sample['frame_id']
            if (key in accepted or key not in poses or poses[key][0] != split or frame in frames
                    or sample.get('volume_plan_sha256') != manifest['volume_plan_sha256']
                    or sample.get('volume_split') != split or sample.get('volume_cell') != poses[key][2]
                    or not frame.startswith('pose-') or not frame[5:].isdigit()
                    or not np.allclose(poses[key][1], sample['drive_target_base_T_effector'], rtol=0, atol=1e-7)
                    or not all((out/(frame+ext)).is_file() for ext in ('.jpg','.npz'))):
                raise ValueError('Inconsistent or missing previously recorded volume sample')
            session = sample.get('capture_session')
            if session is not None:
                if (not isinstance(session,str) or Path(session).name!=session or not session.startswith('capture-session-')
                        or hashlib.sha256((out/session).read_bytes()).hexdigest()!=sample.get('capture_session_sha256')):
                    raise ValueError('Recorded per-session camera calibration is missing or has changed')
            accepted.add(key)
            frames.add(frame)
    return sets


def verify_optical_profile(old, current):
    # The D455 SDK can report temperature-dependent focal lengths at startup.
    # Reject changes in sensor mode or distortion. Bound the full-frame pixel
    # shift, and always deproject each new sample using its actual SDK profile.
    for k in ('width','height','model','coeffs'):
        if old[k] != current[k]:
            raise ValueError('Optical stream mode or distortion changed: '+k)
    shifts=[]
    for length,pp,focal in (('width','ppx','fx'),('height','ppy','fy')):
        values=[float(a[k]) for a in (old,current) for k in (pp,focal)]
        if not all(map(math.isfinite,values)) or min(old[focal],current[focal]) <= 0:
            raise ValueError('Invalid SDK focal length or principal point')
        shifts.append(max(abs((u-old[pp]) * (current[focal]/old[focal]-1) + current[pp]-old[pp])
                          for u in (0,old[length]-1)))
    error=math.hypot(*shifts)
    if error>.75:
        raise ValueError('SDK optical profile changed by {:.3f} pixels at the image boundary'.format(error))
    return error


def capture(args):
    import cv2
    import pyrealsense2 as rs
    import rospy

    out=args.output_dir
    poses = {}
    resume = getattr(args, 'resume', False)
    if getattr(args, 'volume_plan', None):
        poses, volume_hash, counts = volume_capture_plan(args.volume_plan)
        if args.pose_gate is None or (args.train,args.holdout) != (counts['train'],counts['holdout']):
            raise ValueError('Volume capture needs a pose gate and counts matching the frozen plan')
    if resume:
        if not poses or not out.is_dir():
            raise ValueError('Resume requires an existing capture and its original volume plan')
    else:
        out.mkdir(parents=True,exist_ok=False)
    rospy.init_node('baxter_right_handeye_capture',anonymous=True,disable_signals=True)
    robot=EndpointReader(rospy)
    aruco=cv2.aruco
    dictionary=aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    params=aruco.DetectorParameters()
    params.cornerRefinementMethod=aruco.CORNER_REFINE_NONE
    detector=aruco.ArucoDetector(dictionary,params)
    pipe=rs.pipeline()
    config=rs.config()
    config.enable_device(args.serial)
    cw,ch=map(int,args.color_mode.split('x'))
    dw,dh=map(int,args.depth_mode.split('x'))
    config.enable_stream(rs.stream.depth,dw,dh,rs.format.z16,30)
    config.enable_stream(rs.stream.color,cw,ch,rs.format.bgr8,30)
    selected=pipe.start(config)
    try:
        scale=selected.get_device().first_depth_sensor().get_depth_scale()
        color_profile=selected.get_stream(rs.stream.color)
        depth_profile=selected.get_stream(rs.stream.depth)
        intr=color_profile.as_video_stream_profile().get_intrinsics()
        extr=depth_profile.get_extrinsics_to(color_profile)
        manifest={'applied':False,'arm':'right','effector':'right_gripper','frame':'base',
                  'tag_family':'tag36h11','tag_id':0,'camera_serial':args.serial,
                  'tag_axes':'canonical decoded corners 0->1 x, 0->3 y, x cross y z',
                  'camera_frame':'color optical','intrinsics':intrinsics_data(intr),
                  'native_depth_intrinsics':intrinsics_data(depth_profile.as_video_stream_profile().get_intrinsics()),
                  'native_depth_to_color':{'rs_rotation_column_major':extr.rotation,'translation_m':extr.translation},
                  'corner_method':'decoded corner order, outward subpixel edge profiles, Huber line intersections',
                  'distortion':str(intr.model),'depth_scale_m':scale,'ruler_width_m':args.width_m,'ruler_height_m':args.height_m}
        if poses:
            manifest['volume_plan_sha256'] = volume_hash
        if resume:
            sets = load_volume_checkpoint(out,manifest,poses)
            original_manifest = json.loads((out/'manifest.json').read_text(encoding='utf8'))
        else:
            write_json(out/'manifest.json',manifest)
            sets={'train':[],'holdout':[]}
            for name in sets:write_json(out/(name+'.json'),[])
        align=rs.align(rs.stream.color)
        aligned_verified=False
        session_file='capture-session-'+uuid.uuid4().hex+'.json'
        ray=lambda pixel:rs.rs2_deproject_pixel_to_point(intr,pixel,1.)
        began=time.monotonic(); last_notice=began-5; previous_frame=-1
        prior = sorted(sets['train']+sets['holdout'],key=lambda s:s['frame_id'])
        window=deque(); reason='Waiting for tag'
        accepted=[s['base_T_effector'] for s in prior]
        accepted_ids={s['drive_pose_id'] for s in prior}
        next_frame=max((int(path.stem[5:]) for path in out.glob('pose-*.*') if path.stem[5:].isdigit()), default=0)+1
        if len(accepted)==args.train+args.holdout:return 0
        while time.monotonic()-began<args.duration_s and not rospy.is_shutdown():
            try:
                frame_set=pipe.wait_for_frames(1000)
            except RuntimeError:
                window.clear(); reason='D455 frame timeout'
                if time.monotonic()-last_notice>=3:
                    print(json.dumps({'waiting':reason,'accepted':len(accepted)}),flush=True);last_notice=time.monotonic()
                continue
            arrival=time.monotonic()
            try:
                frames=align.process(frame_set)
                color,depth=frames.get_color_frame(),frames.get_depth_frame()
                if not color or not depth or color.get_frame_number()<=previous_frame:
                    raise ValueError('Missing or repeated camera frame')
                previous_frame=color.get_frame_number()
                if not aligned_verified:
                    aligned_intr=depth.profile.as_video_stream_profile().get_intrinsics()
                    actual_intrinsics=intrinsics_data(aligned_intr)
                    if resume and 'aligned_depth_intrinsics' in original_manifest:
                        verify_optical_profile(original_manifest['aligned_depth_intrinsics'],actual_intrinsics)
                    if any(abs(getattr(aligned_intr,k)-getattr(intr,k))>.01 for k in ('fx','fy','ppx','ppy','width','height')):
                        raise RuntimeError('SDK aligned-depth intrinsics disagree with the color stream')
                    manifest['aligned_depth_intrinsics']=actual_intrinsics
                    if not resume or 'aligned_depth_intrinsics' not in original_manifest:
                        write_json(out/'manifest.json',manifest)
                    write_json(out/session_file,manifest)
                    session_digest=hashlib.sha256((out/session_file).read_bytes()).hexdigest()
                    aligned_verified=True
                # The two RealSense streams use the same capture device. Compare
                # their sensor stamps only when their domains agree.
                if color.get_frame_timestamp_domain()!=depth.get_frame_timestamp_domain() or abs(color.get_timestamp()-depth.get_timestamp())>60:
                    raise ValueError('D455 color and aligned depth not time-paired')
                bgr=np.asanyarray(color.get_data()); z16=np.asanyarray(depth.get_data())
                if z16.shape!=bgr.shape[:2] or z16.shape!=(intr.height,intr.width):
                    raise ValueError('Aligned depth size differs from active color intrinsics')
                gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
                corners,ids,_=detector.detectMarkers(gray)
                matches=[] if ids is None else np.flatnonzero(ids.ravel()==0)
                if len(matches)!=1:
                    raise ValueError('Expected one tag36h11 ID 0, detected '+str([] if ids is None else ids.ravel().tolist()))
                coarse=corners[matches[0]].reshape(4,2)
                refined,edge_quality=refine_tag_edges(gray,coarse)
                info=tag_from_depth(refined,z16*scale,ray,args.width_m,args.height_m)
                info.update(decoded_corners_px=coarse.tolist(),edge_fit=edge_quality)
                info.update(robot.sample(arrival),host_time_s=arrival,color_frame=previous_frame,
                            depth_frame=depth.get_frame_number(),color_stamp_ms=color.get_timestamp(),
                            depth_stamp_ms=depth.get_timestamp(),camera_timestamp_domain=str(color.get_frame_timestamp_domain()))
                if args.pose_gate is not None:
                    if info['joint_limit_violations']:
                        raise ValueError('Drive gate cannot acknowledge out-of-model right joints: '+str(info['joint_limit_violations']))
                    name,goal=gate_pose(args.pose_gate,info['base_T_effector'])
                    if poses and (name not in poses or not np.allclose(goal,poses[name][1], rtol=0, atol=1e-7)):
                        raise ValueError('Requested ID/target differs from the frozen volume plan')
                    if poses:
                        info.update(volume_plan_sha256=volume_hash,volume_cell=poses[name][2],volume_split=poses[name][0])
                    if name in accepted_ids:
                        raise ValueError('Pose ID already recorded: '+name)
                    info.update(drive_pose_id=name,drive_target_base_T_effector=goal.tolist())
                    if window and (window[-1]['drive_pose_id']!=name or
                                   window[-1]['drive_target_base_T_effector']!=info['drive_target_base_T_effector']):
                        window.clear()
                if window and arrival-window[-1]['host_time_s']>.25:window.clear()
                window.append(info)
                while len(window)>1 and arrival-window[1]['host_time_s']>=.5:window.popleft()
                status=window_status(window)
                duplicate=duplicate_observation(info['base_T_effector'],accepted) if status is None else None
                if status is not None:
                    reason=status
                elif duplicate is not None:
                    reason='Pose already recorded near {} ({:.2f} mm, {:.2f} deg)'.format(
                        duplicate['nearest_frame_id'],duplicate['distance_mm'],duplicate['angle_deg'])
                    if arrival-last_notice>=3:
                        with (out/'rejections.jsonl').open('a',encoding='utf-8') as log:
                            log.write(json.dumps({'host_s':arrival,'color_frame':previous_frame,'reason':reason,
                                                 'drive_pose_id':info.get('drive_pose_id'),**duplicate})+'\n')
                else:
                    n=len(accepted)+1
                    hold_due=n*args.holdout//(args.train+args.holdout)>len(sets['holdout'])
                    split=poses[info['drive_pose_id']][0] if poses else ('holdout' if hold_due else 'train')
                    evidence='pose-{:03}'.format(next_frame)
                    next_frame+=1
                    if not cv2.imwrite(str(out/(evidence+'.jpg')),bgr):raise OSError('Could not save color frame')
                    np.savez_compressed(out/(evidence+'.npz'),aligned_z16=z16)
                    record=dict(info,arm='right',tag_family='tag36h11',tag_id=0,camera_serial=args.serial,
                                frame_id=evidence,stationary_frames=list(window),
                                capture_session=session_file,capture_session_sha256=session_digest)
                    sets[split].append(record)
                    write_json(out/(split+'.json'),sets[split])
                    accepted.append(info['base_T_effector'])
                    if args.pose_gate is not None:accepted_ids.add(info['drive_pose_id'])
                    print(json.dumps({'recorded':evidence,'split':split,'drive_pose_id':info.get('drive_pose_id'),
                                      'width_m':info['width_m'],'height_m':info['height_m'],
                                      'pose_check':info['pose_check'],'joint_limit_violations':info['joint_limit_violations'],
                                      'fk_error_m':info['fk_error_m'],'plane_rms_m':info['plane_rms_m']}),flush=True)
                    window.clear()
                    if len(accepted)==args.train+args.holdout:return 0
                    reason='Waiting for next distinct stationary pose'
            except ValueError as error:
                window.clear();reason=str(error)
                with (out/'rejections.jsonl').open('a',encoding='utf-8') as log:
                    log.write(json.dumps({'host_s':arrival,'color_frame':previous_frame,'reason':reason})+'\n')
            if arrival-last_notice>=3:
                print(json.dumps({'waiting':reason,'accepted':len(accepted)}),flush=True);last_notice=arrival
        print(json.dumps({'status':'incomplete','train':len(sets['train']),'holdout':len(sets['holdout']),'last_reason':reason}),flush=True)
        return 2
    finally:
        pipe.stop()
        for subscriber in robot.subs:subscriber.unregister()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--serial',default='241122306976')
    p.add_argument('--color-mode',choices=('640x480','1280x720'),default='1280x720')
    p.add_argument('--depth-mode',choices=('640x480','848x480','1280x720'),default='640x480')
    p.add_argument('--duration-s',type=float,default=240.)
    p.add_argument('--train',type=int,default=12)
    p.add_argument('--holdout',type=int,default=4)
    p.add_argument('--pose-gate',type=Path,help='Require the measured endpoint to match an external pose ID/target throughout the window')
    p.add_argument('--volume-plan',type=Path,help='Immutable spatial plan assigning each target and its train/holdout group by ID')
    p.add_argument('--resume',action='store_true',help='Continue the same spatial plan using the same camera calibration and recorded files')
    p.add_argument('--width-m',type=float,help='Ruler measurement of black outer border along decoded x')
    p.add_argument('--height-m',type=float,help='Ruler measurement of black outer border along decoded y')
    args=p.parse_args(argv)
    if not (0<args.duration_s<=7200 and 1<=args.train<=100 and 0<=args.holdout<=100):
        p.error('Invalid capture duration or sample count')
    if any(v is not None and not .01<=v<=.7 for v in (args.width_m,args.height_m)):
        p.error('Ruler dimensions must be in metres')
    return capture(args)


if __name__=='__main__':
    raise SystemExit(main())
