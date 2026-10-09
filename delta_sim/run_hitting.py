"""Shared MuJoCo Delta paddle simulation, planning and contact checks.

Run continuous_hitting.py for the constrained random serve demonstration.
"""
import os, sys, time, math
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco
import kinematics as kin
import make_paddle_robot
from trajectory_view import TrajectoryOverlay, sync_trajectory

# ---------------- 场景常量 ----------------
BASE_POS     = np.array([-0.69, 0.0, 1.25])     # 使竖直拍面拦截规范发球的第二次弹跳
N            = np.array(make_paddle_robot.PADDLE_NORMAL)
T1           = np.array([-N[2], 0.0, N[0]])     # 拍面切向 (竖直面内, 下前方向)
T2           = np.array([0.0, 1.0, 0.0])        # 拍面切向 (水平横向)
P_OFF        = np.array(make_paddle_robot.PADDLE_OFFSET)
FACE_CLEAR   = 0.020 + make_paddle_robot.PADDLE_HALF_THICK + 0.002
HOME_PLAT    = np.array([-0.055, 0.0, -0.215])  # 待机平台位置 (基座系)
Z_FLOOR_REL  = 0.76 + 0.010 - BASE_POS[2] - make_paddle_robot.PADDLE_BOTTOM_OFFSET

LAUNCH_POS   = np.array([1.40, 0.0, 1.02])      # 发球机出球口
LAUNCH_V     = np.array([-6.5, 0.0, -1.0])      # 固定演示来球速度
TARGET_R     = 0.10                             # 成功半径 [m]

PERCEPT_DELAY = 0.08        # 感知+规划延迟 [s]
WINDOW        = 0.04        # 恒速过窗时长 (±20ms; 关节行程 |v|·W·J ≈ 35° 需在量程内)
RECOVER_T     = 0.50        # 回位时间
RAMP_T        = 0.020       # Local acceleration/deceleration around the stroke
V_PAD_MAX     = 1.2         # 拍速上限 [m/s] (挥拍关节行程受电机量程 60° 限制)
QDOT_MAX      = 20.0        # 关节速度上限 [rad/s]
EVAL_TIMEOUT  = 3.2         # 单次试验评估超时
NET_TOP       = 0.9125
TABLE_TOP     = 0.76
BALL_R        = 0.02


def load_models():
    """主模型 + 预测模型 (机器人移至远处, 只留台面/网/球) + 预演评估模型"""
    make_paddle_robot.ensure_paddle_robot()
    model = mujoco.MjModel.from_xml_path("scene_pingpong.xml")
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "delta_base")
    model.body_pos[bid] = BASE_POS
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)

    pred_model = mujoco.MjModel.from_xml_path("scene_pingpong.xml")
    # 把机器人整体移走: <pair> 显式接触对会绕过 contype 过滤 (球-拍面),
    # 复位姿态的拍面会挡在来球路径上, 必须让预测模型中机器人远离球
    pred_model.body_pos[bid] = np.array([50.0, 0.0, 50.0])
    pred_data = mujoco.MjData(pred_model)
    mujoco.mj_setConst(pred_model, pred_data)

    imodel = mujoco.MjModel.from_xml_path("scene_pingpong.xml")
    imodel.body_pos[bid] = BASE_POS
    idata = mujoco.MjData(imodel)
    mujoco.mj_setConst(imodel, idata)
    return model, data, pred_model, pred_data, imodel, idata


def ball_qv(model):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
    return model.jnt_qposadr[jid], model.jnt_dofadr[jid]


def set_ball(data, qadr, vadr, pos, vel, spin=(0, 0, 0)):
    data.qpos[qadr:qadr+3] = pos
    data.qpos[qadr+3:qadr+7] = [1, 0, 0, 0]
    data.qvel[vadr:vadr+3] = vel
    data.qvel[vadr+3:vadr+6] = spin


class HittingSim:
    def __init__(self):
        (self.model, self.data, self.pmodel, self.pdata,
         self.imodel, self.idata) = load_models()
        self.iface = kin.DeltaMujocoInterface(self.model, self.data)
        self.iface_i = kin.DeltaMujocoInterface(self.imodel, self.idata)
        self.qadr, self.vadr = ball_qv(self.model)
        self.pqadr, self.pvadr = ball_qv(self.pmodel)
        self.iqadr, self.ivadr = ball_qv(self.imodel)
        self.kp = self.model.actuator_gainprm[0, 0]
        self.kv = -self.model.actuator_biasprm[0, 2]
        self.gid_ball = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        self.gid_face = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "paddle_face")
        self.gid_table = self.model.geom('table_top').id
        self.gid_net = self.model.geom('net').id
        self.sid_paddle = self.model.site('paddle_center').id
        self.motor_dofs = self.model.jnt_dofadr[self.iface.jids]
        self.bias = np.zeros(2)          # 自适应落点偏差
        self.e, self.lam = 0.60, 0.15    # 碰撞近似模型参数 (粗筛用, 标定后更新)
        self.q_home = kin.ik_safe(HOME_PLAT)

    # ---------- 弹道预测 (预测模型: 机器人移远处) ----------
    def predict_incoming(self, pos, vel, t_horizon=1.8):
        """前向仿真来球, 返回样本 (t, pos, vel, spin) 列表"""
        m, d = self.pmodel, self.pdata
        mujoco.mj_resetData(m, d)
        set_ball(d, self.pqadr, self.pvadr, pos, vel)
        mujoco.mj_forward(m, d)
        samples = []
        stop_x = BASE_POS[0] - 0.15
        n_max = int(t_horizon / m.opt.timestep)
        for i in range(n_max):
            if i % 4 == 0:   # 2ms 采样
                samples.append((i * m.opt.timestep,
                                d.qpos[self.pqadr:self.pqadr+3].copy(),
                                d.qvel[self.pvadr:self.pvadr+3].copy(),
                                d.qvel[self.pvadr+3:self.pvadr+6].copy()))
            mujoco.mj_step(m, d)
            p = d.qpos[self.pqadr:self.pqadr+3]
            if p[0] < stop_x or p[2] < 0.5:
                samples.append(((i+1) * m.opt.timestep, p.copy(),
                                d.qvel[self.pvadr:self.pvadr+3].copy(),
                                d.qvel[self.pvadr+3:self.pvadr+6].copy()))
                break
        return samples

    def inspect_serve(self, pos, vel, t_horizon=1.8):
        """Validate the two table bounces and net crossing in MuJoCo."""
        m, d = self.pmodel, self.pdata
        mujoco.mj_resetData(m, d)
        set_ball(d, self.pqadr, self.pvadr, pos, vel)
        mujoco.mj_forward(m, d)
        samples = []
        bounces = []
        net_cross = None
        net_contact = False
        prev_x = pos[0]
        touching_table = False
        for step in range(int(t_horizon / m.opt.timestep)):
            if step % 4 == 0:
                samples.append((d.time, d.qpos[self.pqadr:self.pqadr+3].copy(),
                                d.qvel[self.pvadr:self.pvadr+3].copy(),
                                d.qvel[self.pvadr+3:self.pvadr+6].copy()))
            mujoco.mj_step(m, d)
            p = d.qpos[self.pqadr:self.pqadr+3]
            table_now = False
            for c in range(d.ncon):
                pair = {d.contact[c].geom1, d.contact[c].geom2}
                if pair == {self.gid_ball, self.gid_table}:
                    table_now = True
                if pair == {self.gid_ball, self.gid_net}:
                    net_contact = True
            if table_now and not touching_table:
                bounces.append((float(d.time), p.copy()))
            touching_table = table_now
            if prev_x > 0 >= p[0] and net_cross is None:
                net_cross = (float(d.time), p.copy())
            prev_x = p[0]
            if len(bounces) >= 2 and p[0] < BASE_POS[0] - 0.15:
                break
            if p[2] < 0.5:
                break
        service = (len(bounces) >= 2 and net_cross is not None and not net_contact
                   and 0.08 < bounces[0][1][0] < 1.32
                   and -1.32 < bounces[1][1][0] < -0.08
                   and all(abs(b[1][1]) < 0.70 for b in bounces[:2])
                   and bounces[0][0] < net_cross[0] < bounces[1][0]
                   and net_cross[1][2] > NET_TOP + BALL_R + 0.015)
        return samples, dict(valid=bool(service), bounces=bounces[:2],
                             net_cross=net_cross, net_contact=net_contact)

    # ---------- 拦截点选择 ----------
    def choose_intercept(self, samples, t_now_min):
        """返回 (score, t, ball_pos, ball_vel, plat_t[基座系], q, slack) 或 None"""
        best = None
        for (t, pos, vel, spin) in samples:
            if t < t_now_min:
                continue
            if vel @ N > -0.3:               # 必须迎着拍面来
                continue
            plat_t = pos - N * FACE_CLEAR - P_OFF - BASE_POS   # 基座系
            if plat_t[2] < Z_FLOOR_REL + 0.020:
                continue
            q = kin.ik_safe(plat_t)
            if q is None:
                continue
            slack = min(np.min(q - kin.Q_SAFE_MIN), np.min(kin.Q_SAFE_MAX - q))
            if slack < math.radians(10):    # 关节余量 (挥拍行程需要)
                continue
            z_score = abs(plat_t[2] + 0.21)  # 偏离最佳工作高度
            score = slack - 0.5 * z_score - 0.8 * t   # 越早拦截越好
            if best is None or score > best[0]:
                best = (score, t, pos, vel, plat_t, q, slack)
        return best

    # ---------- 轨迹规划 (预演与正式共用) ----------
    def make_plan(self, plat_t, v_pad, t_imp_rel):
        """平台轨迹规划 (基座系, 时间轴相对发球时刻)。返回 plan(τ)->(p,v)"""
        p_home = HOME_PLAT
        t_start = PERCEPT_DELAY
        t_A_end = t_imp_rel - WINDOW / 2      # 接近段结束
        p_A = plat_t - v_pad * (WINDOW / 2)
        t_B_end = t_A_end + WINDOW
        p_B = plat_t + v_pad * (WINDOW / 2)
        t_ramp = t_A_end - RAMP_T
        p_ready = p_A - v_pad * RAMP_T / 2
        p_stop = p_B + v_pad * RAMP_T / 2

        def smooth_move(t, duration, start, end):
            s = np.clip(t / duration, 0.0, 1.0)
            h = 10*s**3 - 15*s**4 + 6*s**5
            dh = (30*s**2 - 60*s**3 + 30*s**4) / duration
            return start + h*(end-start), dh*(end-start)

        def plan(τ):
            if τ < t_start:
                return p_home, np.zeros(3)
            if τ < t_ramp:
                return smooth_move(τ-t_start, t_ramp-t_start, p_home, p_ready)
            if τ < t_A_end:
                s = (τ-t_ramp) / RAMP_T
                h = 3*s*s - 2*s**3
                integral = s**3 - 0.5*s**4
                return p_ready + v_pad*RAMP_T*integral, v_pad*h
            if τ < t_B_end:
                return plat_t + v_pad * (τ - t_imp_rel), v_pad
            if τ < t_B_end + RAMP_T:
                s = (τ-t_B_end) / RAMP_T
                h = 3*s*s - 2*s**3
                integral = s - s**3 + 0.5*s**4
                return p_B + v_pad*RAMP_T*integral, v_pad*(1-h)
            if τ < t_B_end + RAMP_T + RECOVER_T:
                return smooth_move(τ-t_B_end-RAMP_T, RECOVER_T, p_stop, p_home)
            return p_home, np.zeros(3)
        return plan

    def inspect_plan(self, plat_t, v_pad, t_imp_rel):
        """Check the entire 500 Hz command sequence, including feed-forward."""
        plan = self.make_plan(plat_t, v_pad, t_imp_rel)
        peak_v = peak_qdot = 0.0
        min_margin = float('inf')
        duration = t_imp_rel + WINDOW/2 + RAMP_T + RECOVER_T
        for t in np.arange(0.0, duration + 0.002, 0.002):
            p, v = plan(t)
            q = kin.ik_safe(p)
            if q is None or p[2] < Z_FLOOR_REL:
                return dict(valid=False, reason='workspace-or-table')
            jac = kin.jacobian(q)
            if jac is None:
                return dict(valid=False, reason='singular')
            qdot = jac @ v
            command = q + (self.kv / self.kp)*qdot
            peak_v = max(peak_v, float(np.linalg.norm(v)))
            peak_qdot = max(peak_qdot, float(np.max(np.abs(qdot))))
            min_margin = min(min_margin, float(np.min(q-kin.Q_PHYS_MIN)),
                             float(np.min(kin.Q_PHYS_MAX-q)))
            if (peak_v > V_PAD_MAX + 1e-9 or peak_qdot > QDOT_MAX or
                    np.any(command < self.model.actuator_ctrlrange[:, 0]) or
                    np.any(command > self.model.actuator_ctrlrange[:, 1])):
                return dict(valid=False, reason='speed-or-command-limit')
        return dict(valid=True, peak_paddle_speed_mps=peak_v,
                    peak_joint_speed_radps=peak_qdot,
                    min_joint_margin_rad=min_margin)

    def servo_ctrl(self, model, data, plan, τ):
        """位置伺服 + 速度前馈 (向 data.ctrl 写入)"""
        p_des, v_des = plan(τ)
        p_des = p_des.copy()
        p_des[2] = max(p_des[2], Z_FLOOR_REL)     # 拍不入台
        q_des = kin.ik(p_des)
        if q_des is not None:
            Jq = kin.jacobian(q_des)
            if Jq is not None:
                q_des = q_des + (self.kv / self.kp) * (Jq @ v_des)
            data.ctrl[:] = q_des

    # ---------- 预演评估 ----------
    def rehearse(self, ball_state0, plan, t_imp_rel):
        """在评估模型中完整重演一次击球。
        ball_state0: 感知时刻 (t=PERCEPT_DELAY) 的球状态 (pos, vel, spin)。
        返回 (landing or None, net_ok, contacted, v_out or None)"""
        m, d = self.imodel, self.idata
        mujoco.mj_resetData(m, d)
        self.iface_i.set_config(self.q_home)
        self.iface_i.ctrl_set(self.q_home)
        pos0, vel0, spin0 = ball_state0
        set_ball(d, self.iqadr, self.ivadr, pos0, vel0, spin0)
        mujoco.mj_forward(m, d)
        t0 = d.time
        net_ok = True
        contacted = False
        v_out = None
        prev_x = pos0[0]
        n = int((t_imp_rel + 1.4) / m.opt.timestep)
        for step in range(n):
            τ = d.time - t0 + PERCEPT_DELAY
            # The plant integrates at 2 kHz; command refresh is 500 Hz,
            # representative of a real high-rate servo bus.
            if step % 4 == 0:
                self.servo_ctrl(m, d, plan, τ)
            mujoco.mj_step(m, d)
            bp = d.qpos[self.iqadr:self.iqadr+3]
            bv = d.qvel[self.ivadr:self.ivadr+3]
            # 接触检测
            if not contacted:
                for c in range(d.ncon):
                    ct = d.contact[c].geom1, d.contact[c].geom2
                    if self.gid_ball in ct and self.gid_face in ct:
                        contacted = True
                        break
            if contacted and v_out is None:
                # 球脱离拍面后记录出射速度
                if bp[0] > BASE_POS[0] + 0.05 and bv[0] > 0:
                    v_out = bv.copy()
            # 过网检查
            if contacted:
                if prev_x <= 0.0 < bp[0] and bp[2] < NET_TOP + BALL_R:
                    net_ok = False
                prev_x = bp[0]
                # 落台 / 出界
                if bv[2] < 0 and bp[2] <= TABLE_TOP + BALL_R + 0.002:
                    if abs(bp[0]) <= 1.37 and abs(bp[1]) <= 0.7625 and bp[0] > 0.02:
                        return (bp[0], bp[1]), net_ok, contacted, v_out
                    return None, net_ok, contacted, v_out
                if bp[2] < 0.1:
                    return None, net_ok, contacted, v_out
            else:
                prev_x = bp[0]
        return None, net_ok, contacted, v_out

    # ---------- 碰撞近似模型 (仅粗筛用) ----------
    def impact_model(self, v_in, v_pad):
        lam = np.clip(self.lam, 0.05, 0.9)
        beta = 1.0 - lam + self.e
        M = lam * np.eye(3) + beta * np.outer(N, N)
        return v_in + M @ (v_pad - v_in)

    def vacuum_landing(self, p0, v_out):
        z0 = p0[2] - (TABLE_TOP + BALL_R)
        vz = v_out[2]
        if z0 <= 0:
            return None
        t = (vz + math.sqrt(vz*vz + 2*9.81*z0)) / 9.81
        return p0[:2] + v_out[:2] * t

    def calibrate_impact(self):
        """开机标定: 球沿 -N + 切向 撞击静止拍面 (仅供粗筛)"""
        m, d = self.model, self.data
        q = kin.ik_safe(np.array([0.0, 0.0, -0.20]))
        mujoco.mj_resetData(m, d)
        p_plat = self.iface.set_config(q)
        self.iface.ctrl_set(q)
        for _ in range(500):
            mujoco.mj_step(m, d)
        mujoco.mj_fwdPosition(m, d)
        blade_c = d.site_xpos[self.sid_paddle].copy()
        v_in = -2.5 * N + 1.5 * T2
        set_ball(d, self.qadr, self.vadr, blade_c + N * 0.05, v_in)
        mujoco.mj_forward(m, d)
        v_out = None
        for _ in range(4000):
            mujoco.mj_step(m, d)
            pos = d.qpos[self.qadr:self.qadr+3] - blade_c
            vel = d.qvel[self.vadr:self.vadr+3]
            if pos @ N > 0.045 and vel @ N > 0:
                v_out = vel
                break
        self.e, self.lam = 0.60, 0.15
        if v_out is not None:
            self.e = -(v_out @ N) / (v_in @ N)
            self.lam = (v_out @ T2) / (v_in @ T2)
        print(f"[标定] 拍-球碰撞(粗筛用): e = {self.e:.3f}, lam = {self.lam:.3f}")

    # ---------- 击球求解: 预演式搜索 ----------
    def solve_shot(self, ball_state0, plat_t, q_hit, t_imp_rel,
                   ball_pos, v_in, target):
        """返回 dict(v_pad, err, net_ok, feasible, landing) 或 None"""
        # The far side of the table is more sensitive to modelled drag and
        # contact timing.  A small range-dependent feed-forward correction,
        # identified from the calibration shots, keeps the landing map nearly
        # affine across the reachable x range.
        aim = target - self.bias + np.array([0.45 * (target[0] - 0.58), 0.0])
        J = kin.jacobian(q_hit) if q_hit is not None else None

        def rehearse_eval(v_pad):
            plan = self.make_plan(plat_t, v_pad, t_imp_rel)
            landing, net_ok, contacted, _ = self.rehearse(ball_state0, plan, t_imp_rel)
            if not contacted or landing is None:
                return None
            err = float(np.linalg.norm(np.array(landing) - aim))
            qd = np.abs(J @ v_pad) if J is not None else np.array([1e9])
            feasible = (np.linalg.norm(v_pad) <= V_PAD_MAX and
                        np.all(qd < QDOT_MAX) and net_ok)
            return dict(v_pad=v_pad, err=err, net_ok=net_ok,
                        feasible=feasible, landing=landing)

        # --- 粗筛: 近似碰撞模型 + 真空弹道 ---
        coarse = []
        # Dense analytical grid is cheap; only a few candidates are rehearse-
        # simulated below.  This keeps the numerical search deterministic
        # without running a full MuJoCo rollout hundreds of times.
        for s in np.linspace(0.0, 1.2, 13):
            for c in np.linspace(-0.45, 0.45, 9):
                for p in np.linspace(-1.0, 1.0, 13):
                    v_pad = s * N + c * T1 + p * T2
                    if np.linalg.norm(v_pad) > V_PAD_MAX:
                        continue
                    v_out = self.impact_model(v_in, v_pad)
                    vl = self.vacuum_landing(ball_pos, v_out)
                    if vl is None:
                        continue
                    err_v = np.linalg.norm(vl - aim)
                    if err_v < 0.55:
                        coarse.append((err_v, v_pad))
        coarse.sort(key=lambda x: x[0])
        # --- 预演复核前 10 个 ---
        best = None
        # Rehearse only the best analytical candidates; the old implementation
        # replayed dozens of expensive full rollouts and took more than a
        # minute for one shot.
        for _, v_pad in coarse[:32]:
            r = rehearse_eval(v_pad)
            if r is None:
                continue
            if best is None or (r['feasible'] and not best['feasible']) or (r['feasible'] == best['feasible'] and r['err'] < best['err']):
                best = r
        if best is None:
            return None
        # --- 山式细化: 沿 (s, c, p) 三方向逐步搜索 ---
        steps = [0.06 * N, 0.03 * T1, 0.075 * T2]
        improved = True
        rounds = 0
        while improved and rounds < 4:
            improved = False
            rounds += 1
            for st in steps:
                for sgn in (+1, -1):
                    v_try = best['v_pad'] + sgn * st
                    if np.linalg.norm(v_try) > V_PAD_MAX:
                        continue
                    r = rehearse_eval(v_try)
                    if r is not None and r['feasible'] and r['err'] < best['err'] - 1e-4:
                        best = r
                        improved = True
        return best

    # ---------- 单次试验 ----------
    def run_trial(self, seed, target, record=True, viewer=None, realtime=False,
                  v_pad_override=None, launch_jitter=1.0, launch=None,
                  incoming=None, service=None, finish_recovery=False):
        rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        target = np.array([target[0], target[1]])

        # --- 重置与稳定 ---
        mujoco.mj_resetData(m, d)
        self.iface.set_config(self.q_home)
        self.iface.ctrl_set(self.q_home)
        for _ in range(500):
            mujoco.mj_step(m, d)
        t_launch = d.time

        # --- 发球 (随机) ---
        y0 = launch_jitter * rng.uniform(-0.03, 0.03)
        v_launch = LAUNCH_V + launch_jitter * np.array([rng.uniform(-0.15, 0.15),
                                                        rng.uniform(-0.08, 0.08),
                                                        rng.uniform(-0.10, 0.10)])
        ball0 = LAUNCH_POS + np.array([0.0, y0, 0.0])
        if launch is not None:
            ball0, v_launch = (np.asarray(v, dtype=float).copy() for v in launch)
        set_ball(d, self.qadr, self.vadr, ball0, v_launch)
        mujoco.mj_forward(m, d)

        log = dict(ball=[], paddle=[], t=[], contact=None, landing=None,
                   net_ok=True, target=target, return_net_cross=None,
                   return_net_contact=False, serve_bounces=[],
                   serve_net_cross=None, serve_net_contact=False)

        # --- 预测来球 ---
        if incoming is None or service is None:
            samples, service = self.inspect_serve(ball0, v_launch)
        else:
            samples = incoming
        if not service['valid']:
            return dict(outcome='invalid-serve', target=target, landing=None,
                        err=None, net_ok=False, contact=None, service=service, log=None)
        inter = self.choose_intercept(
            samples, t_now_min=max(PERCEPT_DELAY + 0.20, service['bounces'][1][0] + 0.015))

        if inter is None:
            for _ in range(int(EVAL_TIMEOUT / m.opt.timestep)):
                mujoco.mj_step(m, d)
            return dict(outcome='no-intercept', target=target, landing=None,
                        err=None, net_ok=False, contact=None, service=service, log=None)

        _, t_imp_rel, ball_pos, v_in, plat_t, q_hit, slack = inter
        t_imp = t_launch + t_imp_rel

        # --- 感知时刻的球状态 (预演起点) ---
        ball_state0 = None
        for (t, pos, vel, spin) in samples:
            if abs(t - PERCEPT_DELAY) < 1.5e-3:
                ball_state0 = (pos.copy(), vel.copy(), spin.copy())
        if ball_state0 is None:
            ball_state0 = (ball0, v_launch, np.zeros(3))

        # --- 击球求解 (预演式) ---
        if v_pad_override is None:
            shot = self.solve_shot(ball_state0, plat_t, q_hit, t_imp_rel,
                                   ball_pos, v_in, target)
            if shot is None or not shot['feasible']:
                v_pad = np.array([0.0, 0.0, 0.15])
            else:
                v_pad = shot['v_pad']
        else:
            shot = None
            v_pad = np.asarray(v_pad_override, dtype=float).copy()

        # --- 平台轨迹规划 (与预演一致) ---
        plan = self.make_plan(plat_t, v_pad, t_imp_rel)
        plan_metrics = self.inspect_plan(plat_t, v_pad, t_imp_rel)
        if v_pad_override is not None and not plan_metrics['valid']:
            return dict(outcome='invalid-plan', target=target, landing=None,
                        err=None, net_ok=False, contact=None, service=service, log=None)

        # --- 主仿真循环 ---
        prev_x = ball0[0]
        impact_detected = False
        touching_table = False
        landing_time = None
        peak_joint_speed = 0.0
        peak_paddle_speed = 0.0
        site_velocity = np.zeros(6)
        min_joint_margin = float('inf')
        overlay = TrajectoryOverlay([s[1] for s in samples]) if viewer is not None else None
        wall_start = time.perf_counter()
        n_steps = int((EVAL_TIMEOUT + 1.0) / m.opt.timestep)
        for step in range(n_steps):
            τ = d.time - t_launch
            if step % 4 == 0:
                self.servo_ctrl(m, d, plan, τ)
            mujoco.mj_step(m, d)
            peak_joint_speed = max(peak_joint_speed, float(np.max(np.abs(d.qvel[self.motor_dofs]))))
            angles = d.qpos[self.iface.qadr]
            min_joint_margin = min(min_joint_margin, float(np.min(angles-kin.Q_PHYS_MIN)),
                                   float(np.min(kin.Q_PHYS_MAX-angles)))

            if viewer is not None and step % 40 == 0:
                if not viewer.is_running():
                    break
                overlay.append(d.qpos[self.qadr:self.qadr+3])
                sync_trajectory(viewer, overlay)
                if realtime:
                    time.sleep(max(0.0, d.time-t_launch-(time.perf_counter()-wall_start)))

            # 接触检测
            if not impact_detected:
                for c in range(d.ncon):
                    ct = d.contact[c].geom1, d.contact[c].geom2
                    if self.gid_ball in ct and self.gid_face in ct:
                        impact_detected = True
                        log['contact'] = (τ, d.qpos[self.qadr:self.qadr+3].copy())
                        break

            # 球状态
            bp = d.qpos[self.qadr:self.qadr+3].copy()
            bv = d.qvel[self.vadr:self.vadr+3].copy()
            table_now = False
            net_now = False
            for c in range(d.ncon):
                pair = {d.contact[c].geom1, d.contact[c].geom2}
                table_now |= pair == {self.gid_ball, self.gid_table}
                net_now |= pair == {self.gid_ball, self.gid_net}
            if record and step % 4 == 0:
                log['ball'].append((τ, bp.copy()))
                mujoco.mj_fwdPosition(m, d)
                log['paddle'].append((τ, d.site_xpos[self.sid_paddle].copy()))
            if step % 4 == 0:
                mujoco.mj_fwdPosition(m, d)
                mujoco.mj_fwdVelocity(m, d)
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE,
                                        self.sid_paddle, site_velocity, 0)
                peak_paddle_speed = max(peak_paddle_speed, float(np.linalg.norm(site_velocity[3:])))

            if not impact_detected:
                if table_now and not touching_table:
                    log['serve_bounces'].append((float(τ), bp.copy()))
                if net_now:
                    log['serve_net_contact'] = True
                if prev_x > 0 >= bp[0] and log['serve_net_cross'] is None:
                    log['serve_net_cross'] = (float(τ), bp.copy())

            # 过网检查 (击球后)
            if impact_detected:
                if net_now and landing_time is None:
                    log['return_net_contact'] = True
                if prev_x <= 0.0 < bp[0] and landing_time is None:
                    log['return_net_cross'] = (float(τ), bp.copy())
                    if bp[2] < NET_TOP + BALL_R:
                        log['net_ok'] = False
                prev_x = bp[0]
                if table_now and not touching_table and landing_time is None:
                    landing_time = float(τ)
                    if 0.02 < bp[0] < 1.35 and abs(bp[1]) < 0.74:
                        log['landing'] = (bp[0], bp[1])
                    if not finish_recovery:
                        break
                if bp[2] < 0.1 and not finish_recovery:
                    break
            else:
                prev_x = bp[0]
            touching_table = table_now
            if (finish_recovery and landing_time is not None and
                    τ >= max(landing_time + 0.35,
                             t_imp_rel + WINDOW/2 + RAMP_T + RECOVER_T + 0.10)):
                break

            if τ > EVAL_TIMEOUT:
                break

        # --- 评估 ---
        if log['landing'] is not None and impact_detected:
            err = np.linalg.norm(np.array(log['landing']) - target)
            success = bool(err < TARGET_R and log['net_ok'] and
                           log['return_net_cross'] is not None and
                           not log['return_net_contact'])
            if log['net_ok']:
                aim = target - self.bias + np.array([0.45 * (target[0] - 0.58), 0.0])
                self.bias = self.bias + 0.40 * (np.array(log['landing']) - aim)
            outcome = 'success' if success else 'missed'
        elif impact_detected:
            err = None
            outcome = 'missed' if log['net_ok'] else 'net'
        else:
            err = None
            outcome = 'missed'
        bounces = log['serve_bounces']
        net_cross = log['serve_net_cross']
        actual_serve_valid = bool(len(bounces) == 2 and net_cross is not None and
                                  not log['serve_net_contact'] and
                                  0.08 < bounces[0][1][0] < 1.32 and
                                  -1.32 < bounces[1][1][0] < -0.08 and
                                  all(abs(b[1][1]) < 0.70 for b in bounces[:2]) and
                                  bounces[0][0] < net_cross[0] < bounces[1][0] and
                                  net_cross[1][2] > NET_TOP + BALL_R + 0.015)
        legal_return = bool(actual_serve_valid and impact_detected and
                            log['landing'] is not None and
                            log['return_net_cross'] is not None and
                            log['net_ok'] and not log['return_net_contact'])
        if viewer is not None:
            if viewer.is_running():
                overlay.append(d.qpos[self.qadr:self.qadr+3])
                sync_trajectory(viewer, overlay)
            else:
                outcome = 'viewer-closed'
        return dict(outcome=outcome, target=target, landing=log['landing'],
                    err=err, net_ok=log['net_ok'], log=log if record else None,
                    contact=log['contact'], v_pad=v_pad.copy(), service=service,
                    legal_return=legal_return,
                    actual_serve_valid=actual_serve_valid,
                    actual_serve_bounces=bounces[:2],
                    actual_serve_net_cross=net_cross,
                    actual_serve_net_contact=log['serve_net_contact'],
                    return_net_cross=log['return_net_cross'],
                    return_net_contact=log['return_net_contact'],
                    intercept=(t_imp_rel, ball_pos.copy(), plat_t.copy(), float(slack)),
                    plan_metrics=plan_metrics,
                    actual_peak_joint_speed_radps=peak_joint_speed,
                    actual_peak_paddle_speed_mps=peak_paddle_speed,
                    actual_min_joint_margin_rad=min_joint_margin,
                    shot=shot if v_pad_override is None else None)
