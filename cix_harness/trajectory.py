"""Dependency-free continuous joint interpolation, with analytic derivative bounds.

Each joint stays in the coordinate interval between adjacent IK knots. Callers
must validate the entire joint box (not only the straight joint-space chord),
and check FK/TCP along sampled_positions before enabling this trajectory.
"""
import bisect
import math


def motion_limits(cfg, phase=None):
    speed=cfg.get('joint_speed_rad_s',.12)
    overrides=cfg.get('joint_speed_by_phase_rad_s',{})
    if not isinstance(overrides,dict) or any(k not in ('transit','contact','orientation','retreat','clearance','lift') for k in overrides):
        raise ValueError('Unknown motion phase speed override')
    for name,value in [('joint speed',speed),('joint acceleration',cfg.get('joint_acceleration_rad_s2',.12))]:
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
            raise ValueError('Invalid '+name)
    if speed>.12:raise ValueError('Joint speed exceeds bridge limit')
    for value in overrides.values():
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<value<=speed:
            raise ValueError('Motion phase speed must be positive and no higher than global limit')
    acceleration=cfg.get('joint_acceleration_rad_s2',.12)
    if acceleration>.5:raise ValueError('Joint acceleration exceeds bridge limit')
    phase={'approach':'contact','placement':'contact'}.get(phase,phase)
    if phase=='lift' and phase not in overrides:phase='contact'
    return overrides.get(phase,speed),acceleration


class JointTrajectory:
    def __init__(self,initial,points,speed,acceleration=.12):
        if not initial or not points:raise ValueError('A trajectory requires initial and target joints')
        for value in (speed,acceleration):
            if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
                raise ValueError('Trajectory limits must be finite and positive')
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in initial.values()):
            raise ValueError('Nonfinite initial trajectory joints')
        self.names=tuple(initial);self.points=[dict(initial)];self.knots=[0.]
        for point in points:
            if set(point)!=set(initial):raise ValueError('Incomplete trajectory joint state')
            if any(type(v) not in (int,float) or not math.isfinite(v) for v in point.values()):
                raise ValueError('Nonfinite trajectory joint state')
            delta=max(abs(point[n]-self.points[-1][n]) for n in self.names)
            if delta<1e-12:continue
            self.knots.append(self.knots[-1]+delta/speed)
            self.points.append(dict(point))
        self.coefficients=[]
        self.joint_travel={n:sum(abs(b[n]-a[n]) for a,b in zip(self.points,self.points[1:])) for n in self.names}
        vmax=amax=0.
        if len(self.points)>1:
            # Give densely sampled start/end knots finite ramp time; otherwise
            # a tiny first chord would force the entire trajectory to crawl.
            lengths=[t*speed for t in self.knots];length=lengths[-1]
            top=min(speed,math.sqrt(length*acceleration))
            ramp=top/acceleration;distance=.5*top*ramp
            total=2*ramp+(length-2*distance)/top
            self.knots=[(math.sqrt(2*s/acceleration) if s<distance else
                         total-math.sqrt(max(0.,2*(length-s)/acceleration)) if s>length-distance else
                         ramp+(s-distance)/top) for s in lengths]
            intervals=[b-a for a,b in zip(self.knots,self.knots[1:])]
            arc_intervals=[b-a for a,b in zip(lengths,lengths[1:])]
            arc_speeds=[min(top,math.sqrt(2*acceleration*s),math.sqrt(max(0.,2*acceleration*(length-s)))) for s in lengths]
            derivatives={n:[0.]*len(self.points) for n in self.names}
            for n in self.names:
                slopes=[(b[n]-a[n])/ds for a,b,ds in zip(self.points,self.points[1:],arc_intervals)]
                for i in range(1,len(self.points)-1):
                    before,after=slopes[i-1:i+1]
                    if before*after>0:
                        # Shape-preserving tangent in arc length, mapped through
                        # the start/end speed ramps. Dense knots keep ramp speed.
                        w1=2*arc_intervals[i]+arc_intervals[i-1];w2=arc_intervals[i]+2*arc_intervals[i-1]
                        derivatives[n][i]=(w1+w2)/(w1/before+w2/after)*arc_speeds[i]
                # Fritsch-Carlson limiter keeps every interval in its joint box.
                for i,dt in enumerate(intervals):
                    slope=(self.points[i+1][n]-self.points[i][n])/dt
                    if slope==0:derivatives[n][i]=derivatives[n][i+1]=0.;continue
                    alpha=derivatives[n][i]/slope;beta=derivatives[n][i+1]/slope
                    norm=math.hypot(alpha,beta)
                    if norm>3:
                        derivatives[n][i]*=3/norm;derivatives[n][i+1]*=3/norm
            for i,dt in enumerate(intervals):
                row={}
                for n in self.names:
                    a,b=self.points[i][n],self.points[i+1][n]
                    v,w=derivatives[n][i]*dt,derivatives[n][i+1]*dt
                    # Polynomial in normalized interval position u.
                    coeff=(a,v,3*(b-a)-2*v-w,2*(a-b)+v+w)
                    row[n]=coeff
                    _,c1,c2,c3=coeff
                    locations=[0.,1.]
                    if abs(c3)>1e-15:
                        critical=-c2/(3*c3)
                        if 0<critical<1:locations.append(critical)
                    vmax=max(vmax,max(abs(c1+2*c2*u+3*c3*u*u)/dt for u in locations))
                    amax=max(amax,abs(2*c2)/(dt*dt),abs(2*c2+6*c3)/(dt*dt))
                self.coefficients.append(row)
        scale=max(1.,vmax/speed,math.sqrt(amax/acceleration))
        self.knots=[t*scale for t in self.knots]
        self.duration=max(.15,self.knots[-1])
        self.peak_velocity=vmax/scale
        self.peak_acceleration=amax/(scale*scale)
        self.metadata={'interpolation':'monotone_cubic_joint_boxes','input_knots':len(points)+1,
                       'distinct_knots':len(self.points),'nominal_duration_s':self.duration,
                       'joint_travel_rad':self.joint_travel,'peak_velocity_rad_s':self.peak_velocity,
                       'peak_acceleration_rad_s2':self.peak_acceleration,
                       'joint_speed_limit_rad_s':speed,'joint_acceleration_limit_rad_s2':acceleration}

    def sample(self,t):
        if t<=0:return dict(self.points[0])
        if t>=self.knots[-1]:return dict(self.points[-1])
        i=bisect.bisect_right(self.knots,t)-1
        u=(t-self.knots[i])/(self.knots[i+1]-self.knots[i])
        return {n:((c3*u+c2)*u+c1)*u+c0 for n,(c0,c1,c2,c3) in self.coefficients[i].items()}

    def sampled_positions(self,max_joint_step=.01):
        if not math.isfinite(max_joint_step) or max_joint_step<=0:raise ValueError('Invalid sampling step')
        yield dict(self.points[0])
        for i in range(len(self.points)-1):
            begin,end=self.knots[i:i+2]
            count=max(1,math.ceil((end-begin)*self.peak_velocity/max_joint_step))
            for j in range(1,count+1):yield self.sample(begin+(end-begin)*j/count)


class ControllerSafetyError(RuntimeError):
    pass


def check_controller_feedback(states,now,max_age=.5):
    """Require current factory diagnostics before issuing a motor command."""
    for key in ('avoidance','collision'):
        entry=states.get(key)
        if entry is None or not 0<=now-entry[0]<max_age:
            raise ControllerSafetyError('Missing or stale factory '+key+' telemetry')
    avoidance=states['avoidance'][1]
    objects=avoidance.get('objects',[])
    if avoidance.get('other_arm') or objects:
        raise ControllerSafetyError('Factory collision avoidance active: '+(', '.join(sorted(set(objects))) or 'other_arm'))
    if states['collision'][1]:raise ControllerSafetyError('Factory contact collision detected')
