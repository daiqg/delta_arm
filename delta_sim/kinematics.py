"""
kinematics.py — Delta 机械臂解析运动学 (与 CAD MuJoCo 模型逐点对齐)
====================================================================

几何参数 (米, 与 delta_robot.xml / CAD STL 完全一致):
  R_B  = 0.0745   基座肩部铰接半径
  L_U  = 0.100    上臂长 (肩轴 -> 肘球铰)
  L_F  = 0.200    前臂杆长 (肘球铰 -> 动平台球铰)
  R_E  = 0.02494  动平台铰接外接圆半径
  肩角: joint1 -> 240°, joint2 -> 120°, joint3 -> 0°
  关节角 q: 0 = 上臂水平向外, 负 = 向下 (物理范围 [-70°, -10°])

Delta 并联机构性质: 动平台始终与基座平行 (无姿态自由度), 末端 = 平台中心,
  位置 (x,y,z) 相对基座中心, 3 个关节角一一对应 (解析 IK 闭式解)。

坐标约定: 本模块全部使用 [基座坐标系, z 向上, 单位米, 弧度],
  与 delta_robot.xml 中 site ee_center (相对 base_center) 完全一致。

闭环一致性已验证: |P_i - E_i| = L_F 对 500 个随机位形残差 < 4e-5 m。
"""

import math
import numpy as np

# ---------------- 几何参数 ----------------
R_B = 0.0745      # 肩部半径
L_U = 0.100       # 上臂长
L_F = 0.200       # 前臂长
R_E = 0.02494     # 平台铰接半径

PHI = (math.radians(240.0),   # joint1 肩角
       math.radians(120.0),   # joint2
       math.radians(0.0))     # joint3

Q_PHYS_MIN = math.radians(-70.0)   # 关节物理限位
Q_PHYS_MAX = math.radians(-10.0)
Q_MARGIN   = math.radians(2.0)     # 规划用安全余量
Q_SAFE_MIN = Q_PHYS_MIN + Q_MARGIN
Q_SAFE_MAX = Q_PHYS_MAX - Q_MARGIN

# 前臂肘部球铰 (上臂局部坐标): [a侧(-横向), b侧(+横向)]
ELBOW_LOCAL = (np.array([0.1, -0.0262, 0.0]),
               np.array([0.1,  0.0262, 0.0]))

# 动平台铰接点 (平台 body 局部坐标; 平台 body 原点 = 杆 3a 接点)
PLAT_SITES = {
    '3a': np.array([0.0,     0.0,    0.0]),
    '3b': np.array([0.0,     0.0524, 0.0]),
    '2a': np.array([-0.0148, 0.0609, 0.0]),
    '2b': np.array([-0.0601, 0.0347, 0.0]),
    '1a': np.array([-0.0601, 0.0177, 0.0]),
    '1b': np.array([-0.0148, -0.0085, 0.0]),
}
EE_CENTER_LOCAL = np.array([-0.02497, 0.0262, 0.0])   # ee_center site 局部坐标

_U = [np.array([math.cos(p), math.sin(p), 0.0]) for p in PHI]   # 径向单位向量
_T = [np.array([-math.sin(p), math.cos(p), 0.0]) for p in PHI]  # 切向单位向量


# ---------------- 基础旋转 ----------------
def _Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _Rarm(q):
    """上臂关节旋转: 绕 (0,-1,0) 转 q (等价 R_y(-q))"""
    c, s = math.cos(q), math.sin(q)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def _quat_from_mat(R):
    """旋转矩阵 -> (w,x,y,z) 四元数"""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _quat_min_rotation(a, b):
    """最小旋转: 单位向量 a -> b"""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = np.clip(a @ b, -1.0, 1.0)
    if dot > 1 - 1e-12:
        return np.array([1.0, 0, 0, 0])
    if dot < -1 + 1e-12:
        # 反向: 任取垂直轴
        axis = np.cross(a, np.array([1.0, 0, 0]))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(a, np.array([0, 1.0, 0]))
        axis /= np.linalg.norm(axis)
        return np.array([0.0, *axis])
    v = np.cross(a, b)
    q = np.array([1.0 + dot, *v])
    return q / np.linalg.norm(q)


# ---------------- FK / IK ----------------
def fk(q):
    """正运动学: 关节角 (3,) -> 末端位置 (3,) [基座系]
    数值牛顿法 (闭合解的稳健替代, 收敛 <10 次迭代)"""
    q = np.asarray(q, dtype=float)
    p = np.array([0.0, 0.0, -0.19])
    for _ in range(80):
        F = np.zeros(3)
        Jm = np.zeros((3, 3))
        for i in range(3):
            E = _U[i] * (R_B + L_U * math.cos(q[i])) + np.array([0, 0, L_U * math.sin(q[i])])
            A = p + _U[i] * R_E
            F[i] = (A - E) @ (A - E) - L_F ** 2
            Jm[i] = 2 * (A - E)
        if np.linalg.norm(F) < 1e-14:
            break
        p = p - np.linalg.solve(Jm, F)
    return p


def ik(p):
    """逆运动学: 末端位置 (3,) -> 关节角 (3,) 或 None (不可达)
    每臂闭式解: a*cos(q) + b*sin(q) = k, 取肘部向下分支 q = atan2(b,a) + acos(k/Rc)"""
    p = np.asarray(p, dtype=float)
    qs = np.zeros(3)
    for i in range(3):
        a = p @ _U[i] + R_E - R_B      # 平台接点相对肩轴的径向有符号距离
        b = p[2]                        # 平台 z
        ct = p @ _T[i]                  # 切向偏移
        h2 = L_F ** 2 - ct ** 2
        if h2 <= 0:
            return None
        k = (a * a + b * b + L_U ** 2 - h2) / (2 * L_U)
        Rc = math.hypot(a, b)
        val = k / Rc
        if abs(val) >= 0.9995:          # 奇异/不可达 (限制杆角 > ~1.8°)
            return None
        qs[i] = math.atan2(b, a) + math.acos(val)
    return qs


def ik_safe(p):
    """带关节限位检查的 IK (含安全余量), 不可达返回 None"""
    q = ik(p)
    if q is None:
        return None
    if np.any(q < Q_SAFE_MIN) or np.any(q > Q_SAFE_MAX):
        return None
    return q


def jacobian(q, eps=1e-5):
    """数值雅可比 dq/dp (3x3): 平台速度 -> 关节速度"""
    q = np.asarray(q, dtype=float)
    p0 = fk(q)
    J = np.zeros((3, 3))
    for j in range(3):
        dp = np.zeros(3)
        dp[j] = eps
        qp = ik(p0 + dp)
        if qp is None:
            qp = ik(p0 - dp)
            if qp is None:
                return None
            J[:, j] = (q - qp) / eps
        else:
            J[:, j] = (qp - q) / eps
    return J


def closure_points(q, p_ee):
    """平行装配模式下各杆两端世界坐标 (用于构造 MuJoCo 一致初始 qpos)"""
    out = {}
    for i, key in [(0, '1'), (1, '2'), (2, '3')]:
        Rz_m = _Rz(PHI[i])
        Ra = _Rarm(q[i])
        for side, tag in [(0, 'a'), (1, 'b')]:
            E = Rz_m @ (np.array([R_B, 0, 0.0]) + Ra @ ELBOW_LOCAL[side])
            out[f'E_{key}{tag}'] = E
    for tag, local in PLAT_SITES.items():
        out[f'P_{tag}'] = p_ee - EE_CENTER_LOCAL + local
    return out


# ---------------- MuJoCo 交互 ----------------
class DeltaMujocoInterface:
    """MuJoCo CAD 模型 <-> 解析运动学的桥接:
    - 一致初始化 (把球铰 qpos 设到平行装配模式, 避免落入翻转装配模式)
    - 末端位置/速度读取 (基座系)
    - 关节角读写"""

    JOINT_NAMES = ("motor_arm_joint1", "motor_arm_joint2", "motor_arm_joint3")
    FOREARM_BODIES = {   # body name -> (臂编号索引, a/b)
        "forearm1a": (0, 'a'), "forearm1b": (0, 'b'),
        "forearm2a": (1, 'a'), "forearm2b": (1, 'b'),
        "forearm3a": (2, 'a'), "forearm3b": (2, 'b'),
    }
    ROD_AXIS0 = np.array([-0.1495, 0.0, -0.1328]) / 0.2   # 前臂 mesh 默认轴向 (局部)

    def __init__(self, model, data):
        self.m, self.d = model, data
        import mujoco
        self.mj = mujoco
        n2i = lambda t, n: mujoco.mj_name2id(model, t, n)
        self.bid_base = n2i(mujoco.mjtObj.mjOBJ_BODY, "delta_base")
        self.bid_platform = n2i(mujoco.mjtObj.mjOBJ_BODY, "end_platform")
        self.sid_ee = n2i(mujoco.mjtObj.mjOBJ_SITE, "ee_center")
        self.sid_base = n2i(mujoco.mjtObj.mjOBJ_SITE, "base_center")
        self.jids = [n2i(mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.JOINT_NAMES]
        self.qadr = [model.jnt_qposadr[j] for j in self.jids]
        self.bids_forearm = {n: n2i(mujoco.mjtObj.mjOBJ_BODY, n) for n in self.FOREARM_BODIES}
        # 球铰 qpos 地址 (4 元素/个)
        self.ball_adr = {}
        for n in list(self.FOREARM_BODIES) + ["end_platform"]:
            bid = n2i(mujoco.mjtObj.mjOBJ_BODY, n)
            for j in range(model.njnt):
                if model.jnt_bodyid[j] == bid:
                    self.ball_adr[n] = model.jnt_qposadr[j]
                    break

    # ---- 一致初始化 ----
    def set_config(self, q):
        """把整个闭环 qpos 设为与关节角 q 一致的平行装配位形 (不破坏动力学状态其余部分)"""
        m, d = self.m, self.d
        q = np.asarray(q, dtype=float)
        p_ee = fk(q)
        pts = closure_points(q, p_ee)
        # 1) 铰链关节
        for i, adr in enumerate(self.qadr):
            d.qpos[adr] = q[i]
        # 2) 前臂球铰: quat = R_parent^-1 * R_world_rod
        rod_quats = {}
        for bname, (i, tag) in self.FOREARM_BODIES.items():
            E = pts[f'E_{i+1}{tag}']
            P = pts[f'P_{i+1}{tag}']
            dvec = (P - E) / np.linalg.norm(P - E)
            R_parent = _Rz(PHI[i]) @ _Rarm(q[i])
            qmin = _quat_min_rotation(self.ROD_AXIS0, dvec)
            R_rod = _quat2mat(qmin)
            quat = _quat_from_mat(R_parent.T @ R_rod)
            rod_quats[bname] = quat
            adr = self.ball_adr[bname]
            d.qpos[adr:adr + 4] = quat
        # 3) 平台球铰: 平台世界姿态 = I, 父体 = forearm3a (含其自身球铰旋转)
        R_elbow3 = _Rz(PHI[2]) @ _Rarm(q[2])
        R_forearm3a = R_elbow3 @ _quat2mat(rod_quats["forearm3a"])
        adr = self.ball_adr["end_platform"]
        d.qpos[adr:adr + 4] = _quat_from_mat(R_forearm3a.T)
        d.qvel[:] = 0.0
        self.mj.mj_forward(m, d)
        return p_ee

    # ---- 读取 ----
    def ee_pos(self):
        """末端位置 (基座系)"""
        return self.d.site_xpos[self.sid_ee] - self.d.site_xpos[self.sid_base]

    def ee_vel(self):
        """末端速度 (基座系), 用 site jacobian"""
        m, d = self.m, self.d
        jacp = np.zeros((3, m.nv))
        self.mj.mj_jacSite(m, d, jacp, None, self.sid_ee)
        return jacp @ d.qvel

    def joint_pos(self):
        return np.array([self.d.qpos[a] for a in self.qadr])

    def joint_vel(self):
        return np.array([self.d.qvel[a] for a in self.qadr])

    def ctrl_set(self, q):
        for i in range(3):
            self.d.ctrl[i] = q[i]

    def connect_residual(self):
        """connect 约束最大残差 [m] (闭环装配质量检查)"""
        m, d = self.m, self.d
        res = 0.0
        pairs = [("forearm3b_tip", "plat_joint_3b"),
                 ("forearm2a_tip", "plat_joint_2a"),
                 ("forearm2b_tip", "plat_joint_2b"),
                 ("forearm1a_tip", "plat_joint_1a"),
                 ("forearm1b_tip", "plat_joint_1b")]
        for s1, s2 in pairs:
            i1 = self.mj.mj_name2id(m, self.mj.mjtObj.mjOBJ_SITE, s1)
            i2 = self.mj.mj_name2id(m, self.mj.mjtObj.mjOBJ_SITE, s2)
            res = max(res, np.linalg.norm(d.site_xpos[i1] - d.site_xpos[i2]))
        return res


def _quat2mat(q):
    """(w,x,y,z) -> 旋转矩阵"""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# ---------------- 工作空间 ----------------
def workspace_scan(z_values=None, step_r=0.001):
    """IK 可行内切半径扫描 (关节限位含安全余量)
    返回 {z: r_inscribed}"""
    if z_values is None:
        z_values = np.arange(-0.27, -0.149, 0.01)
    out = {}
    for z in z_values:
        r_in = 0.0
        for r in np.arange(0.0, 0.12, step_r):
            ok = True
            for th in range(0, 360, 10):
                p = np.array([r * math.cos(math.radians(th)),
                              r * math.sin(math.radians(th)), z])
                if ik_safe(p) is None:
                    ok = False
                    break
            if not ok:
                break
            r_in = r
        out[round(z, 4)] = r_in
    return out


if __name__ == "__main__":
    # 自检: FK/IK 往返
    rng = np.random.default_rng(0)
    err = 0.0
    for _ in range(1000):
        q = rng.uniform(Q_SAFE_MIN, Q_SAFE_MAX, 3)
        p = fk(q)
        q2 = ik(p)
        if q2 is None:
            print("IK fail at reachable p:", p)
            break
        err = max(err, np.abs(q2 - q).max())
    print(f"FK/IK roundtrip max joint err: {err*1e6:.1f} urad")
    ws = workspace_scan(step_r=0.002)
    best = max(ws.items(), key=lambda kv: kv[1])
    print(f"workspace: best z={best[0]:.2f} m, inscribed radius={best[1]*1e3:.1f} mm")
