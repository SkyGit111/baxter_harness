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
    if not isinstance(overrides,dict) or any(k not in ('transit','contact','orientation','retreat','clearance','lift','lift_clear') for k in overrides):
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
    if phase=='lift_clear':
        return overrides.get('lift_clear',overrides.get('lift',overrides.get('transit',speed))),acceleration
    if phase=='lift' and phase not in overrides:phase='contact'
    return overrides.get(phase,speed),acceleration


class _ArcSchedule:
    """Accelerate, cruise and brake over one joint-space interval."""

    def __init__(self, length, entry, exit_speed, speed_cap, acceleration):
        self.length = length
        self.entry = entry
        self.acceleration = acceleration
        self.peak = min(speed_cap, math.sqrt(acceleration * length + (entry**2 + exit_speed**2) / 2))
        self.ramp_s = max(0., (self.peak - entry) / acceleration)
        self.ramp_length = (self.peak + entry) * self.ramp_s / 2
        brake_s = max(0., (self.peak - exit_speed) / acceleration)
        brake_length = (self.peak + exit_speed) * brake_s / 2
        self.cruise_s = max(0., length - self.ramp_length - brake_length) / self.peak
        self.duration = self.ramp_s + self.cruise_s + brake_s

    def fraction(self, seconds):
        if seconds <= self.ramp_s:
            distance = seconds * (self.entry + self.acceleration * seconds / 2)
        else:
            after_ramp = seconds - self.ramp_s
            braking = max(0., after_ramp - self.cruise_s)
            distance = self.ramp_length + self.peak * after_ramp - self.acceleration * braking**2 / 2
        return min(1., max(0., distance / self.length))


class JointTrajectory:
    def __init__(self, initial, points, speed, acceleration=.12):
        if not initial or not points:
            raise ValueError('A trajectory requires initial and target joints')
        for value in (speed, acceleration):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('Trajectory limits must be finite and positive')

        self.names = tuple(initial)
        self.points = []
        lengths = []
        for point in [initial, *points]:
            if set(point) != set(initial):
                raise ValueError('Incomplete trajectory joint state')
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in point.values()):
                raise ValueError('Nonfinite trajectory joint state')
            if self.points:
                delta = max(abs(point[n] - self.points[-1][n]) for n in initial)
                if delta < 1e-12:
                    continue
                lengths.append(delta)
            self.points.append(dict(point))

        self.joint_travel = {n: sum(abs(b[n] - a[n]) for a,b in zip(self.points, self.points[1:]))
                             for n in initial}
        self.coefficients = [{} for _ in lengths]
        slopes = [0.] * len(lengths)
        curvature = [0.] * len(lengths)
        for joint in initial:
            values = [point[joint] for point in self.points]
            secants = [(b - a) / ds for a,b,ds in zip(values, values[1:], lengths)]
            tangents = [0.] * len(values)
            # Monotone PCHIP derivatives in geometric arc length. Rest at the
            # two endpoints comes from the time schedule, so straight strokes
            # keep their tangent even at the first and last short IK interval.
            if secants:
                tangents[0], tangents[-1] = secants[0], secants[-1]
            for i in range(1, len(values) - 1):
                before, after = secants[i-1:i+1]
                if before * after > 0:
                    w1 = 2 * lengths[i] + lengths[i-1]
                    w2 = lengths[i] + 2 * lengths[i-1]
                    tangents[i] = (w1 + w2) / (w1 / before + w2 / after)
            for i,ds in enumerate(lengths):
                # Express in differences to avoid cancellation at large joint offsets.
                delta = values[i+1] - values[i]
                v,w = tangents[i] * ds, tangents[i+1] * ds
                c2,c3 = 3 * delta - 2*v - w, v + w - 2*delta
                self.coefficients[i][joint] = (values[i], v, c2, c3)
                slope = max(abs(v), abs(v + 2*c2 + 3*c3)) / ds
                if c3 != 0:
                    extremum = -c2 / (3*c3)
                    if 0 < extremum < 1:
                        slope = max(slope, abs(v + 2*c2*extremum + 3*c3*extremum**2) / ds)
                slopes[i] = max(slopes[i], slope)
                curvature[i] = max(curvature[i], abs(2*c2) / ds**2, abs(2*c2 + 6*c3) / ds**2)

        # q''(time) = q''(arc) * arc_speed**2 + q'(arc) * arc_acceleration.
        # Slow only the neighbourhood of a bend. Maintain an acceleration
        # budget for both terms, using all of it for ramping on straight paths.
        caps = []
        ramps = []
        for slope,bend in zip(slopes, curvature):
            cap = speed / slope
            if bend > 0:
                cap = min(cap, math.sqrt(.5 * acceleration / bend))
            caps.append(cap)
            ramps.append((acceleration - bend * cap**2) / slope)
        speeds = [0.] + [min(a,b) for a,b in zip(caps, caps[1:])] + [0.]
        for i,ds in enumerate(lengths):
            speeds[i+1] = min(speeds[i+1], math.sqrt(speeds[i]**2 + 2*ramps[i]*ds))
        for i in reversed(range(len(lengths))):
            speeds[i] = min(speeds[i], math.sqrt(speeds[i+1]**2 + 2*ramps[i]*lengths[i]))

        self.intervals = []
        # For FK preflight, sample the geometric curve in normalized arc
        # position. A slow corner does not need extra samples because time
        # ticks accumulate there. This bound applies to each individual joint.
        self.interval_derivative_bounds = [ds * slope for ds,slope in zip(lengths, slopes)]
        self.knots = [0.]
        self.peak_velocity = self.peak_acceleration = 0.
        for i,ds in enumerate(lengths):
            schedule = _ArcSchedule(ds, speeds[i], speeds[i+1], caps[i], ramps[i])
            self.intervals.append(schedule)
            self.knots.append(self.knots[-1] + schedule.duration)
            self.peak_velocity = max(self.peak_velocity, slopes[i] * schedule.peak)
            self.peak_acceleration = max(self.peak_acceleration,
                                         curvature[i] * schedule.peak**2 + slopes[i] * ramps[i])
        self.duration = max(.15, self.knots[-1])
        self.metadata = {'interpolation':'monotone_cubic_joint_boxes', 'retiming':'local_arc_limits',
                         'input_knots':len(points)+1, 'distinct_knots':len(self.points),
                         'nominal_duration_s':self.duration, 'joint_travel_rad':self.joint_travel,
                         'peak_velocity_rad_s':self.peak_velocity, 'peak_acceleration_rad_s2':self.peak_acceleration,
                         'joint_speed_limit_rad_s':speed, 'joint_acceleration_limit_rad_s2':acceleration}

    def sample(self, t):
        if t <= 0:
            return dict(self.points[0])
        if t >= self.knots[-1]:
            return dict(self.points[-1])
        i = bisect.bisect_right(self.knots, t) - 1
        u = self.intervals[i].fraction(t - self.knots[i])
        return self._position_at_fraction(i, u)

    def _position_at_fraction(self, i, u):
        return {n: ((c3*u + c2)*u + c1)*u + c0 for n,(c0,c1,c2,c3) in self.coefficients[i].items()}

    def sampled_positions(self,max_joint_step=.01):
        if not math.isfinite(max_joint_step) or max_joint_step<=0:raise ValueError('Invalid sampling step')
        yield dict(self.points[0])
        for i,bound in enumerate(self.interval_derivative_bounds):
            count = max(1, math.ceil(bound / max_joint_step))
            for j in range(1, count):
                yield self._position_at_fraction(i, j / count)
            yield dict(self.points[i+1])


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
