"""
run_hitting.py — Delta 机械臂 + 乒乓球拍 击球仿真 (v3, 预演式求解)
====================================================================
场景: 标准球台 (2.74x1.525x0.76m) + 球网, 发球机在台端外发球,
      Delta 机械臂吊装在己方台端上方, 拍面前倾30°刚性固定于动平台。

击球几何 (与真实接攻球一致):
  发球 -> 发球方台面弹跳 -> 过网 -> 接球方台面弹跳 -> 球升至顶点后下落
  -> 机械臂在台面上方以下落姿态拦截来球 (拍面法线与来球近共线)
  -> 反弹 + 挥拍前送 -> 球过网落在对方台面目标落点

核心设计 —— 预演式击球求解 (rehearsal solver):
  1. 发球机随机发球; 感知延迟 80ms 后用预测模型 (机器人移至远处) 前向仿真
     来球轨迹 (含自旋), 得到全弹道样本。
  2. 在可达窗口内选拦截点 (解析 IK + 关节余量 + 拍面迎球)。
  3. 击球求解: 对每个候选拍速 v_pad (s·N + c·T1 + p·T2 网格):
     在专用评估模型中"预演"完整击球 —— 机器人从待机位按真实轨迹规划
     (Hermite 接近 -> 恒速过窗 -> 回位) 挥拍, 球带真实速度+自旋飞来,
     一直仿真到球落台。预演与正式试验物理完全一致 (同模型同控制),
     预测落点 = 实际落点 (确定性系统), 从根本上消除碰撞简化模型误差
     (台面弹跳赋予的球自旋无法用 e/lambda 线性模型描述)。
  4. 真空弹道 + 线性碰撞模型只作粗筛; 山式细化搜索最小落点误差。
  5. 正式执行: 位置伺服 + 解析 IK + 速度前馈, 拍心精确扫过触球点。
  6. 自适应偏差修正吸收残余系统误差 (数值级别, 仅微调)。

用法:
  python run_hitting.py                 # 20 次试验
  python run_hitting.py --n 50
"""
import os, sys, time, math, argparse
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco
import kinematics as kin
import make_paddle_robot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- 场景常量 ----------------
BASE_POS     = np.array([-0.69, 0.0, 1.18])     # 机器人基座世界位置 (定心于拦截窗口)
N            = np.array([math.cos(math.radians(30)), 0.0,
                         math.sin(math.radians(30))])   # 拍面法线 (前倾30°)
T1           = np.array([-N[2], 0.0, N[0]])     # 拍面切向 (竖直面内, 下前方向)
T2           = np.array([0.0, 1.0, 0.0])        # 拍面切向 (水平横向)
P_OFF        = np.array([0.045, 0.0, -0.095])   # 拍心相对平台中心 (平台系=基座系)
FACE_CLEAR   = 0.020 + 0.006 + 0.002            # 球半径+拍半厚+余量
HOME_PLAT    = np.array([-0.055, 0.0, -0.215])  # 待机平台位置 (基座系)
Z_FLOOR_REL  = -0.235                           # 平台下限 (拍缘不入台面)

LAUNCH_POS   = np.array([1.58, 0.0, 1.02])      # 发球机出球口
LAUNCH_V     = np.array([-5.0, 0.0, -1.0])      # 基础发球速度
TARGETS      = [(0.45, 0.0), (0.58, 0.18), (0.58, -0.18),
                (0.32, 0.0), (0.68, 0.0)]       # 目标落点 (对方台面)
TARGET_R     = 0.10                             # 成功半径 [m]

PERCEPT_DELAY = 0.08        # 感知+规划延迟 [s]
WINDOW        = 0.04        # 恒速过窗时长 (±20ms; 关节行程 |v|·W·J ≈ 35° 需在量程内)
RECOVER_T     = 0.50        # 回位时间
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


def hermite(t, T, p0, v0, p1, v1):
    """三次 Hermite: 返回 (p, v)"""
    s = min(max(t / T, 0.0), 1.0)
    h00 = 2*s**3 - 3*s**2 + 1
    h10 = s**3 - 2*s**2 + s
    h01 = -2*s**3 + 3*s**2
    h11 = s**3 - s**2
    dh00 = (6*s**2 - 6*s) / T
    dh10 = (3*s**2 - 4*s + 1) / T
    dh01 = (-6*s**2 + 6*s) / T
    dh11 = (3*s**2 - 2*s) / T
    p = h00*p0 + h10*T*v0 + h01*p1 + h11*T*v1
    v = dh00*p0 + dh10*T*v0 + dh01*p1 + dh11*T*v1
    return p, v


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
        self.gid_handle = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "paddle_handle")
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
        T_A = t_A_end - t_start
        p_A = plat_t - v_pad * (WINDOW / 2)
        t_B_end = t_A_end + WINDOW
        p_B = plat_t + v_pad * (WINDOW / 2)
        T_R = RECOVER_T

        def plan(τ):
            if τ < t_start:
                return p_home, np.zeros(3)
            if τ < t_A_end:
                return hermite(τ - t_start, T_A, p_home, np.zeros(3), p_A, v_pad)
            if τ < t_B_end:
                return plat_t + v_pad * (τ - t_A_end), v_pad
            if τ < t_B_end + T_R:
                return hermite(τ - t_B_end, T_R, p_B, v_pad, p_home, np.zeros(3))
            return p_home, np.zeros(3)
        return plan

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
                    if (self.gid_ball in ct and
                            (self.gid_face in ct or self.gid_handle in ct)):
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
        blade_c = BASE_POS + p_plat + P_OFF
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
                  v_pad_override=None, launch_jitter=1.0):
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
        set_ball(d, self.qadr, self.vadr, ball0, v_launch)
        mujoco.mj_forward(m, d)

        log = dict(ball=[], paddle=[], t=[], contact=None, landing=None,
                   net_ok=True, target=target)

        # --- 预测来球 ---
        samples = self.predict_incoming(ball0, v_launch)
        inter = self.choose_intercept(samples, t_now_min=PERCEPT_DELAY + 0.20)

        if inter is None:
            for _ in range(int(EVAL_TIMEOUT / m.opt.timestep)):
                mujoco.mj_step(m, d)
            return dict(outcome='no-intercept', target=target, landing=None,
                        err=None, net_ok=False, log=None)

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
            v_pad = np.asarray(v_pad_override, dtype=float).copy()

        # --- 平台轨迹规划 (与预演一致) ---
        plan = self.make_plan(plat_t, v_pad, t_imp_rel)

        # --- 主仿真循环 ---
        prev_x = ball_pos[0]
        impact_detected = False
        n_steps = int((EVAL_TIMEOUT + 1.0) / m.opt.timestep)
        for step in range(n_steps):
            τ = d.time - t_launch
            if step % 4 == 0:
                self.servo_ctrl(m, d, plan, τ)
            mujoco.mj_step(m, d)

            if viewer is not None and step % 4 == 0:
                if not viewer.is_running():
                    return dict(outcome='viewer-closed', target=target, landing=None,
                                err=None, net_ok=False, log=log, contact=None)
                viewer.sync()
                if realtime:
                    time.sleep(0.001)

            # 接触检测
            if not impact_detected:
                for c in range(d.ncon):
                    ct = d.contact[c].geom1, d.contact[c].geom2
                    if self.gid_ball in ct and (self.gid_face in ct or self.gid_handle in ct):
                        impact_detected = True
                        log['contact'] = (τ, d.qpos[self.qadr:self.qadr+3].copy())
                        break

            # 球状态
            bp = d.qpos[self.qadr:self.qadr+3].copy()
            bv = d.qvel[self.vadr:self.vadr+3].copy()
            if record and step % 4 == 0:
                log['ball'].append((τ, bp.copy()))
                log['paddle'].append((τ, (BASE_POS + self.iface.ee_pos() + P_OFF).copy()))

            # 过网检查 (击球后)
            if impact_detected:
                if prev_x <= 0.0 < bp[0] and bp[2] < NET_TOP + BALL_R:
                    log['net_ok'] = False
                prev_x = bp[0]
                # 落台 / 出界
                if bv[2] < 0 and bp[2] <= TABLE_TOP + BALL_R + 0.002:
                    if abs(bp[0]) <= 1.37 and abs(bp[1]) <= 0.7625 and bp[0] > 0.02:
                        log['landing'] = (bp[0], bp[1])
                    break
                if bp[2] < 0.1:
                    break
            else:
                prev_x = bp[0]

            if τ > EVAL_TIMEOUT:
                break

        # --- 评估 ---
        if log['landing'] is not None and impact_detected:
            err = np.linalg.norm(np.array(log['landing']) - target)
            success = bool(err < TARGET_R and log['net_ok'])
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
        return dict(outcome=outcome, target=target, landing=log['landing'],
                    err=err, net_ok=log['net_ok'], log=log if record else None,
                    contact=log['contact'], v_pad=v_pad.copy())

    # ---------- 批量试验 ----------
    def run(self, n_trials=20, seed0=1000, view=False):
        results = []
        t0 = time.time()
        for i in range(n_trials):
            rng = np.random.default_rng(seed0 + i)
            target = TARGETS[rng.integers(len(TARGETS))]
            r = self.run_trial(seed0 + i, target, record=(i < 6 or i % 5 == 0))
            results.append(r)
            err_s = f"{r['err']*100:5.1f}cm" if r['err'] is not None else "  ---"
            print(f"  trial {i+1:2d}: {r['outcome']:>12s}  err={err_s}  "
                  f"target=({target[0]:.2f},{target[1]:.2f})")
        self.results = results
        self.elapsed = time.time() - t0
        return results

    # ---------- 统计与绘图 ----------
    def report(self, outdir="results"):
        os.makedirs(outdir, exist_ok=True)
        rs = self.results
        n = len(rs)
        succ = [r for r in rs if r['outcome'] == 'success']
        landed = [r for r in rs if r['landing'] is not None]
        errs = np.array([r['err'] for r in rs if r['err'] is not None])
        net_fail = sum(1 for r in rs if not r['net_ok'] and r['outcome'] != 'no-intercept')
        noic = sum(1 for r in rs if r['outcome'] == 'no-intercept')

        print("\n" + "=" * 62)
        print(f"  试验数        : {n}")
        print(f"  成功 (落点<{TARGET_R*100:.0f}cm 且过网) : {len(succ)}/{n}  ({100*len(succ)/n:.0f}%)")
        if len(errs):
            print(f"  落点误差      : mean {errs.mean()*100:.1f} cm | max {errs.max()*100:.1f} cm")
        print(f"  落台率        : {len(landed)}/{n}")
        print(f"  触网失败      : {net_fail}/{n}")
        print(f"  无法拦截      : {noic}/{n}")
        print(f"  总耗时        : {self.elapsed:.1f}s")
        print(f"  自适应偏差    : dx={self.bias[0]*100:+.1f}cm dy={self.bias[1]*100:+.1f}cm")
        print("=" * 62)

        # --- 落点散点图 ---
        fig, ax = plt.subplots(figsize=(8, 5.6))
        ax.add_patch(plt.Rectangle((0, -0.7625), 1.37, 1.525, fill=False, ec='k', lw=1.5))
        ax.plot([0, 0], [-0.95, 0.95], 'g-', lw=2, label='net')
        cmap = {'success': 'tab:green', 'missed': 'tab:orange', 'net': 'tab:red',
                'no-intercept': 'tab:gray'}
        for r in rs:
            tx, ty = r['target']
            ax.add_patch(plt.Circle((tx, ty), TARGET_R, fill=False, ec='b', ls='--', alpha=0.6))
            if r['landing'] is not None:
                ax.plot(*r['landing'], 'o', ms=8, color=cmap[r['outcome']], alpha=0.85)
                ax.annotate('', xy=r['landing'], xytext=(tx, ty),
                            arrowprops=dict(arrowstyle='->', color='0.6', lw=0.7))
            else:
                ax.plot(tx, ty, 'x', ms=10, color=cmap[r['outcome']], mew=2)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.set_title(f'Landing distribution (success {len(succ)}/{n})')
        handles = [plt.Line2D([], [], marker='o', ls='', color=c, label=l)
                   for l, c in cmap.items() if any(r['outcome'] == l for r in rs)]
        ax.legend(handles=handles, loc='upper left', fontsize=9)
        ax.set_xlim(-0.15, 1.45); ax.set_ylim(-0.85, 0.85); ax.set_aspect('equal'); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "hit_landing_scatter.png"), dpi=140)
        plt.close(fig)

        # --- 误差时间序列 ---
        if len(errs):
            fig, ax = plt.subplots(figsize=(8, 3.4))
            xs = [i+1 for i, r in enumerate(rs) if r['err'] is not None]
            es = [r['err']*100 for r in rs if r['err'] is not None]
            cols = ['tab:green' if r['outcome'] == 'success' else 'tab:orange'
                    for r in rs if r['err'] is not None]
            ax.bar(xs, es, color=cols, alpha=0.85)
            ax.axhline(TARGET_R*100, color='r', ls='--', label=f'success radius {TARGET_R*100:.0f}cm')
            ax.set_xlabel('trial'); ax.set_ylabel('landing error (cm)')
            ax.set_title('Landing error (adaptive bias learning)')
            ax.legend(); ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, "hit_error_series.png"), dpi=140)
            plt.close(fig)

        # --- 示例轨迹 3D ---
        plotted = 0
        for i, r in enumerate(rs):
            if plotted >= 3:
                break
            if r['log'] is None or len(r['log']['ball']) < 10:
                continue
            log = r['log']
            fig = plt.figure(figsize=(9, 6))
            ax = fig.add_subplot(111, projection='3d')
            bp = np.array([p for _, p in log['ball']])
            pd = np.array([p for _, p in log['paddle']])
            ax.plot(bp[:, 0], bp[:, 1], bp[:, 2], 'r-', lw=1.5, label='ball')
            ax.plot(pd[:, 0], pd[:, 1], pd[:, 2], 'b-', lw=1.2, label='paddle center')
            ax.scatter(*r['target'], color='g', s=80, marker='*', label='target')
            if r['landing'] is not None:
                ax.scatter(*r['landing'], color='orange', s=60, label='landing')
            xs = np.array([-1.37, 1.37])
            ys = np.array([-0.7625, 0.7625])
            X, Y = np.meshgrid(xs, ys)
            ax.plot_surface(X, Y, 0.76*np.ones_like(X), alpha=0.15, color='b')
            ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
            ax.set_title(f"trial {i+1}: {r['outcome']}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"hit_example_{i+1}.png"), dpi=130)
            plt.close(fig)
            plotted += 1
        print(f"  figures saved to {outdir}/hit_*.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--seed', type=int, default=1000)
    args = ap.parse_args()

    sim = HittingSim()
    sim.calibrate_impact()
    print(f"\n开始 {args.n} 次击球试验 (随机发球 + 随机目标落点)...")
    sim.run(n_trials=args.n, seed0=args.seed)
    sim.report()


if __name__ == "__main__":
    main()
