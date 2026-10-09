"""Causal position estimator and receding-horizon Delta paddle controller.

The controller accepts timestamped positions and robot encoder feedback only.
It has no access to the serve command, plant ball velocity or trial outcomes.
"""
from dataclasses import dataclass
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import kinematics as kin
from run_hitting import (BASE_POS, BALL_R, FACE_CLEAR, HOME_PLAT, N, NET_TOP,
                        P_OFF, QDOT_MAX, RECOVER_T, TABLE_TOP,
                        V_PAD_MAX, WINDOW, Z_FLOOR_REL, ball_qv, set_ball)


# Online stroke settings; the preserved rehearsal uses its original ramp.
RAMP_T = .03
COMPUTE_BUDGET = .10

def flat_paddle_model():
    """Exact planar rubber faces with the existing 170 x 150 x 12 mm outline."""
    root = ET.parse(Path(__file__).with_name('scene_pingpong.xml')).getroot()
    included = ET.parse(Path(__file__).with_name('delta_robot_paddle.xml')).getroot()
    root.remove(root.find('include'))
    for node in included:
        target = root.find(node.tag)
        if target is None:
            root.append(node)
        else:
            target.extend(list(node))
    angles = np.arange(64)*2*np.pi/64
    vertices = np.array([[.085*np.cos(a), .075*np.sin(a), z]
                         for z in (-.006,.006) for a in angles])
    ET.SubElement(root.find('asset'), 'mesh', name='flat_blade_contact',
                  vertex=' '.join(map(str,vertices.ravel())))
    face = root.find(".//geom[@name='paddle_face']")
    face.set('type','mesh');face.set('mesh','flat_blade_contact');face.attrib.pop('size')
    return root


def ball_model():
    """Ball-only prediction scene with the plant's air/table parameters."""
    root = ET.parse(Path(__file__).with_name('scene_pingpong.xml')).getroot()
    root.remove(root.find('include'))
    root.remove(root.find('sensor'))
    contacts = root.find('contact')
    contacts.remove(contacts.find("pair[@name='ball_paddle_face']"))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


@dataclass(frozen=True)
class Observation:
    captured: float
    delivered: float
    position: np.ndarray


class PositionEstimator:
    """Model-based alpha-beta filter, with table-induced spin propagated internally."""
    def __init__(self):
        self.model = ball_model()
        self.data = mujoco.MjData(self.model)
        self.qadr, self.vadr = ball_qv(self.model)
        self.history = []
        self.timestamp = None
        self.count = 0
        self.last_innovation = 0.0

    def observe(self, observation):
        t, position = observation.captured, observation.position
        if self.timestamp is not None and t <= self.timestamp:
            raise ValueError('Observation timestamps must increase')
        self.count += 1
        if self.timestamp is None:
            self.history.append((t, position.copy()))
            if len(self.history) < 8:
                return
            times = np.array([s[0] - t for s in self.history])
            positions = np.array([s[1] for s in self.history])
            positions[:,2] += 4.905*times**2
            centred = times-times.mean()
            velocity = np.sum(centred[:,None]*positions,axis=0)/(centred@centred)
            position = positions.mean(axis=0)-velocity*times.mean()
            set_ball(self.data, self.qadr, self.vadr, position, velocity)
            self.timestamp = t
            mujoco.mj_forward(self.model, self.data)
            return
        duration = t - self.timestamp
        for _ in range(round(duration / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        innovation = position - self.data.qpos[self.qadr:self.qadr+3]
        self.last_innovation = float(np.linalg.norm(innovation))
        self.data.qpos[self.qadr:self.qadr+3] += 0.35 * innovation
        self.data.qvel[self.vadr:self.vadr+3] += 0.04 / duration * innovation
        self.timestamp = t
        mujoco.mj_forward(self.model, self.data)

    def state(self):
        if self.timestamp is None:
            return None
        return (self.timestamp, self.data.qpos[self.qadr:self.qadr+3].copy(),
                self.data.qvel[self.vadr:self.vadr+3].copy(),
                self.data.qvel[self.vadr+3:self.vadr+6].copy())


class FlightPredictor:
    def __init__(self):
        self.model = ball_model()
        self.data = mujoco.MjData(self.model)
        self.qadr, self.vadr = ball_qv(self.model)
    def incoming(self, state, now, horizon=0.9):
        timestamp, pos, vel, spin = state
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        set_ball(d, self.qadr, self.vadr, pos, vel, spin)
        mujoco.mj_forward(m, d)
        samples = []
        for step in range(0, int((now + horizon - timestamp) / m.opt.timestep), 8):
            absolute = timestamp + d.time
            if step % 8 == 0 and absolute >= now:
                samples.append((absolute, d.qpos[self.qadr:self.qadr+3].copy(),
                                d.qvel[self.vadr:self.vadr+3].copy(),
                                d.qvel[self.vadr+3:self.vadr+6].copy()))
            mujoco.mj_step(m, d, nstep=8)
            if d.qpos[self.qadr] < BASE_POS[0] - 0.20 or d.qpos[self.qadr+2] < 0.5:
                break
        return samples

    @staticmethod
    def fast_outgoing(velocity, paddle_velocity):
        """Approximate planar-rubber impact; plant outcome is never a filter."""
        normal = N * np.dot(paddle_velocity - velocity, N) * 1.60
        tangent = 0.18 * (paddle_velocity - velocity - np.dot(paddle_velocity - velocity, N) * N)
        return velocity + normal + tangent

    @staticmethod
    def fast_landing(position, velocity):
        """Vacuum landing estimate, sufficient to rank nearby stroke speeds."""
        dz = position[2] - (TABLE_TOP + BALL_R)
        disc = velocity[2] * velocity[2] + 19.62 * dz
        if disc <= 0 or velocity[0] <= 0:
            return None
        dt = (velocity[2] + np.sqrt(disc)) / 9.81
        if dt <= 0:
            return None
        landing = position[:2] + velocity[:2] * dt
        net_t = -position[0] / velocity[0] if velocity[0] > 1e-6 else -1
        net_z = position[2] + velocity[2] * net_t - 4.905 * net_t * net_t if net_t > 0 else None
        legal = (net_z is not None and net_z > NET_TOP + BALL_R + 0.012
                 and 0.06 < landing[0] < 1.31 and abs(landing[1]) < 0.70)
        return landing, legal, net_z


class Quintic:
    def __init__(self, start_time, duration, p0, v0, a0, p1, v1=None, a1=None):
        self.start = start_time
        self.duration = duration
        v1 = np.zeros(3) if v1 is None else v1
        a1 = np.zeros(3) if a1 is None else a1
        # Normalised time keeps the endpoint solve well conditioned for short moves.
        c0, c1, c2 = p0, v0*duration, a0*duration**2/2
        rhs = np.array([p1-c0-c1-c2, v1*duration-c1-2*c2,
                        a1*duration**2-2*c2])
        aa, bb, cc = rhs
        tail = np.array([10*aa-4*bb+.5*cc, -15*aa+7*bb-cc, 6*aa-3*bb+.5*cc])
        self.coefficients = np.array([c0, c1, c2, *tail])

    def at(self, t):
        s = np.clip((t-self.start)/self.duration, 0, 1)
        c = self.coefficients
        p = np.array([1, s, s*s, s**3, s**4, s**5]) @ c
        v = np.array([0, 1, 2*s, 3*s*s, 4*s**3, 5*s**4]) @ c / self.duration
        a = np.array([0, 0, 2, 6*s, 12*s*s, 20*s**3]) @ c / self.duration**2
        return p, v, a


class StrokePlan:
    def __init__(self, start, state, intercept_time, platform, velocity):
        self.start = start
        self.impact = intercept_time
        self.platform = platform.copy()
        self.velocity = velocity.copy()
        self.ramp_start = intercept_time-WINDOW/2-RAMP_T
        ready = platform-velocity*(WINDOW/2+RAMP_T/2)
        self.approach = Quintic(start, self.ramp_start-start, *state, ready)
        self.stroke_end = intercept_time+WINDOW/2+RAMP_T
        stop = platform+velocity*(WINDOW/2+RAMP_T/2)
        self.recovery = Quintic(self.stroke_end, RECOVER_T, stop, np.zeros(3),
                                np.zeros(3), HOME_PLAT)
        self.end = self.stroke_end+RECOVER_T

    def at(self, t):
        v = self.velocity
        if t < self.ramp_start:
            return self.approach.at(t)
        if t < self.ramp_start+RAMP_T:
            s = (t-self.ramp_start)/RAMP_T
            ready = self.platform-v*(WINDOW/2+RAMP_T/2)
            return (ready+v*RAMP_T*(s**3-.5*s**4), v*(3*s*s-2*s**3),
                    v*(6*s-6*s*s)/RAMP_T)
        if t < self.impact+WINDOW/2:
            return self.platform+v*(t-self.impact), v, np.zeros(3)
        if t < self.stroke_end:
            s = (t-self.impact-WINDOW/2)/RAMP_T
            return (self.platform+v*WINDOW/2+v*RAMP_T*(s-s**3+.5*s**4),
                    v*(1-3*s*s+2*s**3), -v*(6*s-6*s*s)/RAMP_T)
        if t < self.end:
            return self.recovery.at(t)
        return HOME_PLAT.copy(), np.zeros(3), np.zeros(3)


def safe_command(model, position, velocity):
    q = kin.ik_safe(position)
    if q is None or position[2] < Z_FLOOR_REL:
        return None
    # Differentiate the three forearm sphere constraints analytically.
    radial = np.asarray(kin._U)
    elbow = radial*(kin.R_B+kin.L_U*np.cos(q))[:, None]
    elbow[:, 2] = kin.L_U*np.sin(q)
    r = position+radial*kin.R_E-elbow
    derivative = -radial*(kin.L_U*np.sin(q))[:, None]
    derivative[:, 2] = kin.L_U*np.cos(q)
    denom = np.sum(r*derivative, axis=1)
    if np.min(np.abs(denom)) < 1e-8:
        return None
    jac = r/denom[:, None]
    qdot = jac @ velocity
    gain = -model.actuator_biasprm[0, 2]/model.actuator_gainprm[0, 0]
    command = q+gain*qdot
    if (np.linalg.norm(velocity) > V_PAD_MAX+1e-8 or np.max(np.abs(qdot)) > QDOT_MAX
            or np.any(command < model.actuator_ctrlrange[:, 0])
            or np.any(command > model.actuator_ctrlrange[:, 1])):
        return None
    return command


def safe_plan(model, plan):
    """Vectorised check of every 2 ms command, using analytic constraints."""
    times = np.arange(plan.start, plan.end+.002, .002)
    p = np.empty((len(times),3)); v = np.empty_like(p)
    for curve, mask in ((plan.approach, times<plan.ramp_start),
                        (plan.recovery, times>=plan.stroke_end)):
        z = np.clip((times[mask]-curve.start)/curve.duration,0,1)
        c = curve.coefficients
        p[mask] = sum(z[:,None]**k*c[k] for k in range(6))
        v[mask] = sum(k*z[:,None]**(k-1)*c[k] for k in range(1,6))/curve.duration
    for i in np.flatnonzero((times>=plan.ramp_start)&(times<plan.stroke_end)):
        p[i],v[i],_ = plan.at(times[i])
    radial = np.asarray(kin._U); tangent = np.asarray(kin._T)
    a = p @ radial.T + kin.R_E-kin.R_B
    b = p[:, 2:3]
    h2 = kin.L_F**2-(p @ tangent.T)**2
    val = (a*a+b*b+kin.L_U**2-h2)/(2*kin.L_U*np.hypot(a,b))
    if np.any(h2<=0) or np.any(np.abs(val)>=.9995):
        return False
    q = np.arctan2(b,a)+np.arccos(val)
    if (np.any(q<kin.Q_SAFE_MIN) or np.any(q>kin.Q_SAFE_MAX)
            or np.any(p[:,2]<Z_FLOOR_REL) or np.any(np.linalg.norm(v,axis=1)>V_PAD_MAX)):
        return False
    elbow = radial[None,:,:]*(kin.R_B+kin.L_U*np.cos(q))[:,:,None]
    elbow[:,:,2] = kin.L_U*np.sin(q)
    r = p[:,None,:]+radial[None,:,:]*kin.R_E-elbow
    derivative = -radial[None,:,:]*(kin.L_U*np.sin(q))[:,:,None]
    derivative[:,:,2] = kin.L_U*np.cos(q)
    denom = np.sum(r*derivative,axis=2)
    if np.any(np.abs(denom)<1e-8):
        return False
    qdot = np.sum(r*v[:,None,:],axis=2)/denom
    gain = -model.actuator_biasprm[0,2]/model.actuator_gainprm[0,0]
    command = q+gain*qdot
    return bool(np.all(np.abs(qdot)<=QDOT_MAX)
                and np.all(command>=model.actuator_ctrlrange[:,0])
                and np.all(command<=model.actuator_ctrlrange[:,1]))


class OnlineController:
    def __init__(self, model, period=0.06):
        self.model = model
        self.period = period
        self.estimator = PositionEstimator()
        self.predictor = FlightPredictor()
        self.plan = None
        self.pending = None
        self.next_update = 0.0
        self.events = []
        self.activations = []
        self.prediction = []
        self.command_faults = 0

    def commanded_state(self, t):
        if self.plan is None:
            return HOME_PLAT.copy(), np.zeros(3), np.zeros(3)
        return self.plan.at(t)

    def choose(self, state, now):
        samples = self.predictor.incoming(state, now)
        self.prediction = [s[1] for s in samples]
        candidates = []
        for t, pos, vel, spin in samples:
            if t < now+.14 or vel[0] > -.3:
                continue
            platform = pos-N*FACE_CLEAR-P_OFF-BASE_POS
            if not (-.12 < platform[0] < .10 and .80 < pos[2] < 1.04):
                continue
            score = -abs(pos[0]+.645)
            candidates.append((score, t, pos, vel, spin, platform))
        candidates.sort(key=lambda c: c[0], reverse=True)
        # Use the full planar face: its centre stays at a safe height while
        # interception time and lateral position are selected from observations.
        # A conservative stroke template avoids the old velocity overshoot.
        for _, t, pos, vel, spin, platform in candidates:
            centre_platform = platform.copy()
            centre_platform[2] = -.217
            stroke = np.array([.97, np.clip(-.5*pos[1], -.1, .1), .20])
            plan = StrokePlan(now+COMPUTE_BUDGET, self.commanded_state(now+COMPUTE_BUDGET),
                              t, centre_platform, stroke)
            if plan.ramp_start-plan.start < .01 or any(
                    safe_command(self.model, *plan.at(tt)[:2]) is None
                    for tt in (plan.ramp_start, t-.01, t, t+.01, plan.stroke_end)):
                continue
            if not safe_plan(self.model, plan):
                continue
            forecast = self.predictor.fast_landing(pos,
                self.predictor.fast_outgoing(vel, stroke))
            if forecast is None:
                continue
            landing, legal, _ = forecast
            # An approximate prediction is logged, never substituted for the
            # actual MuJoCo return event evaluation.
            return t, centre_platform, stroke, pos, landing, legal
        return None

    def update(self, now, feedback, contacted=False):
        if contacted:
            self.pending = None
            return
        if np.any(feedback < kin.Q_PHYS_MIN) or np.any(feedback > kin.Q_PHYS_MAX):
            self.events.append(dict(time_s=now, status='encoder-limit', compute_s=0.))
            return
        if self.pending is not None and now >= self.pending.start:
            self.plan = self.pending
            self.pending = None
            self.activations.append(float(now))
        if contacted or now < self.next_update or self.pending is not None:
            return
        if self.plan is not None and now >= self.plan.impact-.12:
            return
        self.next_update = now+self.period
        started = time.perf_counter()
        state = self.estimator.state()
        if state is None:
            return
        proposal = self.choose(state, now)
        event = dict(time_s=now, observation_time_s=state[0],
                     observation_age_s=now-state[0], status='no-intercept')
        if proposal is not None:
            impact, platform, velocity, ball, landing, legal = proposal
            # Fixed publication budget: old plan runs until ready. The measured
            # estimation + planning + validation cost MUST fit inside it.
            ready = now+COMPUTE_BUDGET
            event.update(impact_time_s=float(impact), predicted_ball_x_m=float(ball[0]),
                         predicted_ball_z_m=float(ball[2]), predicted_legal=int(legal),
                         predicted_landing_x_m=float(landing[0]))
            if impact-ready-WINDOW/2-RAMP_T < 0.01:
                event['status'] = 'too-late'
            else:
                plan = StrokePlan(ready, self.commanded_state(ready), impact, platform, velocity)
                safe = safe_plan(self.model, plan)
                elapsed = time.perf_counter()-started
                if elapsed > COMPUTE_BUDGET:
                    event['status'] = 'deadline-miss'
                elif safe:
                    self.pending = plan
                    event.update(status='scheduled', ready_time_s=ready)
                else:
                    event['status'] = 'unsafe-plan'
        event['compute_s'] = time.perf_counter()-started
        self.events.append(event)

    def command(self, now):
        p, v, _ = self.commanded_state(now)
        command = safe_command(self.model, p, v)
        if command is None:
            self.command_faults += 1
        return command
