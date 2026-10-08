"""
verify_kinematics.py — 解析运动学 vs MuJoCo CAD 模型交叉验证 + 工作空间分析
============================================================================
验证内容:
  1. set_config 一致初始化后, MuJoCo 末端位置 == 解析 FK (随机位形)
  2. 从 MuJoCo 末端位置做解析 IK, 关节角应与 MuJoCo 当前关节角一致
  3. connect 约束残差 (闭环装配质量)
  4. 动力学步进若干秒后 (伺服保持), 机构保持在平行装配模式
  5. IK 可行工作空间 (内切半径 ~ z 曲线)
"""
import os, sys, math
import numpy as np
import mujoco

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kinematics as kin

m = mujoco.MjModel.from_xml_string("""
<mujoco model="verify">
  <compiler meshdir="meshes" angle="radian" inertiafromgeom="auto"/>
  <option timestep="0.001" integrator="implicitfast"/>
  <include file="delta_robot.xml"/>
</mujoco>""")
d = mujoco.MjData(m)
iface = kin.DeltaMujocoInterface(m, d)

# Check compiled geometry, since a mesh attribute alone can produce a sphere.
cad_geoms = np.flatnonzero(m.geom_dataid >= 0)
assert len(cad_geoms) == 36, "Incomplete CAD assembly"
assert np.all(m.geom_type[cad_geoms] == mujoco.mjtGeom.mjGEOM_MESH), (
    "CAD components must compile as mesh geoms, not fitted primitives")
print(f"CAD geometry check: {len(cad_geoms)} mesh components")

rng = np.random.default_rng(42)

print("=" * 64)
print("1) FK 一致性: set_config 后 MuJoCo ee vs 解析 fk (80 随机位形)")
errs, ik_errs, res_all = [], [], []
for _ in range(80):
    q = rng.uniform(kin.Q_SAFE_MIN, kin.Q_SAFE_MAX, 3)
    p_an = iface.set_config(q)                       # 返回解析 FK
    p_mj = iface.ee_pos()
    errs.append(np.linalg.norm(p_mj - p_an))
    q_ik = kin.ik(p_mj)
    if q_ik is not None:
        ik_errs.append(np.abs(q_ik - iface.joint_pos()).max())
    res_all.append(iface.connect_residual())
print(f"   末端位置误差 : max {max(errs)*1e3:.4f} mm   mean {np.mean(errs)*1e3:.4f} mm")
print(f"   IK 关节误差  : max {np.rad2deg(max(ik_errs)):.4f} deg")
print(f"   connect 残差 : max {max(res_all)*1e3:.4f} mm")

print("2) 动力学稳定性: 伺服保持 q=-35°, 步进 3 s (重力开启)")
m.opt.gravity[:] = [0, 0, -9.81]
q0 = np.array([math.radians(-35)] * 3)
iface.set_config(q0)
iface.ctrl_set(q0)
for _ in range(3000):
    mujoco.mj_step(m, d)
p_mj, q_mj = iface.ee_pos(), iface.joint_pos()
p_an = kin.fk(q0)
print(f"   3s 后: ee = {np.round(p_mj,5)} (解析 {np.round(p_an,5)})")
print(f"   偏差 {np.linalg.norm(p_mj-p_an)*1e3:.3f} mm, 关节保持偏差 {np.rad2deg(np.abs(q_mj-q0)).max():.3f} deg (伺服刚度内)")
print(f"   connect 残差 {iface.connect_residual()*1e3:.4f} mm")

print("3) 工作空间 (IK 可行, 关节限位 + 2° 余量, 杆角奇异保护)")
ws = kin.workspace_scan(step_r=0.002)
for z, r in ws.items():
    print(f"   z = {z:+.2f} m   内切半径 = {r*1e3:5.1f} mm")
best = max(ws.items(), key=lambda kv: kv[1])
print(f"   >>> 最佳工作高度 z = {best[0]:.2f} m, 内切半径 {best[1]*1e3:.1f} mm")
print("=" * 64)
