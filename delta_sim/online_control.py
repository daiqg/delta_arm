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
                        P_OFF, QDOT_MAX, RAMP_T, RECOVER_T, TABLE_TOP,
                        V_PAD_MAX, WINDOW, Z_FLOOR_REL, ball_qv, set_ball)


def ball_model(paddle=False):
    """Reuse scene contact/aerodynamic settings without the robot mechanism."""
    root = ET.parse(Path(__file__).with_name('scene_pingpong.xml')).getroot()
    root.remove(root.find('include'))
    root.remove(root.find('sensor'))
    contacts = root.find('contact')
    pair = contacts.find("pair[@name='ball_paddle_face']")
    if not paddle:
        contacts.remove(pair)
    else:
        body = ET.SubElement(root.find('worldbody'), 'body', name='predict_paddle')
        ET.SubElement(body, 'freejoint', name='predict_paddle_joint')
        ET.SubElement(body, 'inertial', pos='0 0 0', mass='0.167',
                      diaginertia='0.0007 0.0004 0.0003')
        ET.SubElement(body, 'geom', name='paddle_face', type='ellipsoid',
                      size='0.006 0.075 0.085', contype='0', conaffinity='0')
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
            if len(self.history) < 4:
                return
            times = np.array([s[0] - t for s in self.history])
            coefficients = np.polynomial.polynomial.polyfit(
                times, np.array([s[1] for s in self.history]), 2)
            set_ball(self.data, self.qadr, self.vadr, coefficients[0], coefficients[1])
            self.timestamp = t
            mujoco.mj_forward(self.model, self.data)
            return
        duration = t - self.timestamp
        for _ in range(round(duration / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        innovation = position - self.data.qpos[self.qadr:self.qadr+3]
        self.last_innovation = float(np.linalg.norm(innovation))
        self.data.qpos[self.qadr:self.qadr+3] += 0.65 * innovation
        self.data.qvel[self.vadr:self.vadr+3] += 0.12 / duration * innovation
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
        self.contact_model = ball_model(paddle=True)
        self.contact_data = mujoco.MjData(self.contact_model)
        self.cq, self.cv = ball_qv(self.contact_model)
        joint = self.contact_model.joint('predict_paddle_joint').id
        self.pq = self.contact_model.jnt_qposadr[joint]
        self.pv = self.contact_model.jnt_dofadr[joint]

    def incoming(self, state, now, horizon=0.9):
        timestamp, pos, vel, spin = state
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        set_ball(d, self.qadr, self.vadr, pos, vel, spin)
        mujoco.mj_forward(m, d)
        samples = []
        for step in range(int((now + horizon - timestamp) / m.opt.timestep)):
            absolute = timestamp + d.time
            if step % 8 == 0 and absolute >= now:
                samples.append((absolute, d.qpos[self.qadr:self.qadr+3].copy(),
                                d.qvel[self.vadr:self.vadr+3].copy(),
                                d.qvel[self.vadr+3:self.vadr+6].copy()))
            mujoco.mj_step(m, d)
            if d.qpos[self.qadr] < BASE_POS[0] - 0.20 or d.qpos[self.qadr+2] < 0.5:
                break
        return samples

    def outgoing(self, position, velocity, spin, paddle_velocity):
        """Short local contact forecast; no robot rollout and no plant state."""
        m, d = self.contact_model, self.contact_data
        mujoco.mj_resetData(m, d)
        centre = position - N * FACE_CLEAR
        set_ball(d, self.cq, self.cv, position, velocity, spin)
        d.qpos[self.pq:self.pq+3] = centre
        d.qpos[self.pq+3:self.pq+7] = [1, 0, 0, 0]
        d.qvel[self.pv:self.pv+3] = paddle_velocity
        mujoco.mj_forward(m, d)
        touched = False
        ball_id, face_id = m.geom('ball_geom').id, m.geom('paddle_face').id
        for step in range(80):
            d.qpos[self.pq:self.pq+3] = centre + paddle_velocity*d.time
            d.qpos[self.pq+3:self.pq+7] = [1, 0, 0, 0]
            d.qvel[self.pv:self.pv+3] = paddle_velocity
            d.qvel[self.pv+3:self.pv+6] = 0
            mujoco.mj_step(m, d)
            touched |= any({c.geom1, c.geom2} == {ball_id, face_id}
                           for c in d.contact[:d.ncon])
            if touched and d.qvel[self.cv] > 0 and step > 30:
                return (d.qpos[self.cq:self.cq+3].copy(),
                        d.qvel[self.cv:self.cv+3].copy(),
                        d.qvel[self.cv+3:self.cv+6].copy())
        return None

    def landing(self, state):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        set_ball(d, self.qadr, self.vadr, *state)
        mujoco.mj_forward(m, d)
        previous_x = state[0][0]
        net_z = None
        for _ in range(2200):
            mujoco.mj_step(m, d)
            position = d.qpos[self.qadr:self.qadr+3]
            if previous_x < 0 <= position[0]:
                net_z = float(position[2])
            previous_x = position[0]
            if d.qvel[self.vadr+2] < 0 and position[2] <= TABLE_TOP + BALL_R + 0.002:
                legal = (net_z is not None and net_z > NET_TOP + BALL_R + 0.012
                         and 0.06 < position[0] < 1.31 and abs(position[1]) < 0.70)
                return position[:2].copy(), legal, net_z
            if position[2] < 0.5:
                break
        return None, False, net_z

    @staticmethod
    def fast_outgoing(velocity, paddle_velocity):
        """Fast rigid-face approximation used only to rank online candidates.

        The selected command is still checked once through ``outgoing`` below.
        This prevents candidate enumeration from consuming the available stroke
        lead time while retaining MuJoCo contact physics at the decision point.
        """
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
        tail = np.linalg.solve(np.array([[1, 1, 1], [3, 4, 5], [6, 12, 20]]), rhs)
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
    jac = kin.jacobian(q)
    if jac is None:
        return None
    qdot = jac @ velocity
    gain = -model.actuator_biasprm[0, 2]/model.actuator_gainprm[0, 0]
    command = q+gain*qdot
    if (np.linalg.norm(velocity) > V_PAD_MAX+1e-8 or np.max(np.abs(qdot)) > QDOT_MAX
            or np.any(command < model.actuator_ctrlrange[:, 0])
            or np.any(command > model.actuator_ctrlrange[:, 1])):
        return None
    return command


class OnlineController:
    def __init__(self, model, period=0.12):
        self.model = model
        self.period = period
        self.estimator = PositionEstimator()
        self.predictor = FlightPredictor()
        self.plan = None
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
            if t < now+0.14 or vel[0] > -0.3:
                continue
            platform = pos-N*FACE_CLEAR-P_OFF-BASE_POS
            q = kin.ik_safe(platform)
            if q is None or platform[2] < Z_FLOOR_REL+0.020:
                continue
            # A forward return stroke consumes about 30 mm of platform X
            # travel.  Reserve that room before accepting a static IK point.
            if platform[0] > -0.005:
                continue
            slack = min(np.min(q-kin.Q_SAFE_MIN), np.min(kin.Q_SAFE_MAX-q))
            if slack < np.deg2rad(8):
                continue
            score = slack-.5*abs(platform[2]+.21)-.8*(t-now)
            candidates.append((score, t, pos, vel, spin, platform))
        candidates.sort(key=lambda c: c[0], reverse=True)
        if not candidates:
            return None
        _, t, pos, vel, spin, platform = candidates[0]
        best = None
        # Rank candidate strokes with a constant-time collision/flight estimate.
        # No candidate is executed in the robot plant and no serve is discarded here.
        for vx, vz in ((.62, .28), (.70, .35), (.78, .32), (.85, .30), (.90, .25)):
            stroke = np.array([vx, np.clip(-0.5*pos[1], -.10, .10), vz])
            if np.linalg.norm(stroke) > V_PAD_MAX:
                continue
            v_out = self.predictor.fast_outgoing(vel, stroke)
            forecast = self.predictor.fast_landing(pos, v_out)
            if forecast is None:
                continue
            landing, legal, _ = forecast
            cost = np.linalg.norm(landing-np.array([.38, 0]))+(0 if legal else 2)
            if best is None or cost < best[0]:
                best = (cost, stroke, landing, legal)
        if best is None:
            return None
        # One MuJoCo free-paddle contact verifies the selected candidate.  It
        # has a fixed cost independent of the number of candidate strokes.
        outgoing = self.predictor.outgoing(pos, vel, spin, best[1])
        if outgoing is not None:
            landing, legal, _ = self.predictor.landing(outgoing)
            if landing is not None:
                best = (np.linalg.norm(landing-np.array([.38, 0])) + (0 if legal else 2),
                        best[1], landing, legal)
        return t, platform, best[1], pos, best[2], best[3]

    def update(self, now, _feedback, contacted=False):
        if contacted or now < self.next_update:
            return
        # Keep a committed stroke.  Replanning before its approach begins can
        # replace a feasible command with a later, infeasible one.
        if self.plan is not None:
            return
        self.next_update = now+self.period
        state = self.estimator.state()
        if state is None:
            return
        started = time.perf_counter()
        proposal = self.choose(state, now)
        elapsed = time.perf_counter()-started
        event = dict(time_s=now, observation_time_s=state[0],
                     observation_age_s=now-state[0], compute_s=elapsed,
                     status='no-intercept', ready_time_s=None, impact_time_s=None)
        if proposal is not None:
            impact, platform, velocity, ball, landing, legal = proposal
            event.update(impact_time_s=float(impact), predicted_ball_x_m=float(ball[0]),
                         predicted_ball_z_m=float(ball[2]), paddle_vx_mps=float(velocity[0]),
                         paddle_vz_mps=float(velocity[2]),
                         platform_x_m=float(platform[0]), platform_y_m=float(platform[1]),
                         platform_z_m=float(platform[2]))
            # A result cannot affect the plant until its computation has finished.
            ready = now+elapsed
            if impact-ready-WINDOW/2-RAMP_T < 0.07:
                event['status'] = 'too-late'
            else:
                start_state = self.commanded_state(ready)
                plan = StrokePlan(ready, start_state, impact, platform, velocity)
                checks = [(t, safe_command(self.model, *plan.at(t)[:2]))
                          for t in np.arange(ready, plan.end+0.002, 0.006)]
                safe = all(command is not None for _, command in checks)
                if not safe:
                    event['unsafe_time_s'] = float(next(t for t, command in checks if command is None))
                elapsed = time.perf_counter()-started
                ready = now+elapsed
                # Include command validation in measured latency; rebuild continuity.
                if safe and impact-ready-WINDOW/2-RAMP_T >= 0.07:
                    plan = StrokePlan(ready, self.commanded_state(ready), impact,
                                      platform, velocity)
                    checks = [(t, safe_command(self.model, *plan.at(t)[:2]))
                              for t in np.arange(ready, plan.end+0.002, 0.006)]
                    safe = all(command is not None for _, command in checks)
                    if not safe:
                        event['unsafe_time_s'] = float(next(t for t, command in checks if command is None))
                    final_elapsed = time.perf_counter()-started
                    # The plan starts at the measured completion time.  It is
                    # stored now, but StrokePlan.at() holds the current command
                    # until ``ready``; no command acts before computation ends.
                    if safe and final_elapsed <= self.period:
                        self.plan = plan
                        self.activations.append(float(now+final_elapsed))
                        event.update(status='scheduled', ready_time_s=now+final_elapsed,
                                     impact_time_s=impact, predicted_ball_x_m=ball[0],
                                     predicted_ball_z_m=ball[2], paddle_vx_mps=velocity[0],
                                     paddle_vz_mps=velocity[2], predicted_legal=int(legal),
                                     predicted_landing_x_m=landing[0])
                    else:
                        event['status'] = 'deadline-miss' if final_elapsed > self.period else 'unsafe-plan'
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
