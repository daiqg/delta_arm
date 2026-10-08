"""
run_trajectory.py — Delta 机械臂末端轨迹绘制仿真
====================================================================
让末端(动平台中心)沿常见几何轨迹运动, 位置伺服 + 解析 IK 闭环跟踪,
记录期望/实际轨迹并输出对比图与误差统计。

轨迹尺寸基于 IK 实测工作空间 (z=-0.21 m 处内切半径 74 mm) 取 ~75%:
  circle  : 半径 55 mm   (直径 110 mm)
  square  : 边长 80 mm
  figure8 : 90 mm x 45 mm Lissajous
  star    : 外径 50 mm 五角星
  helix   : 半径 40 mm, z 行程 40 mm (3D)

用法:
  python run_trajectory.py                     # 全部形状, MuJoCo 窗口
  python run_trajectory.py --shape circle --view
  python run_trajectory.py --shape all --headless
"""
import os, sys, time, argparse
from contextlib import nullcontext
from pathlib import Path
import json
import numpy as np
import math

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco
import kinematics as kin
from trajectory_view import TrajectoryOverlay, configure_viewer, sync_trajectory, trajectory_camera

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
available_fonts = {f.name for f in font_manager.fontManager.ttflist}
for font in ('Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC'):
    if font in available_fonts:
        plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
        break
plt.rcParams['axes.unicode_minus'] = False

CTRL_FREQ_HZ = 1000.0     # 控制频率 = 仿真步频 (1 kHz)
Z_DRAW = -0.210           # 绘制平面 (工作空间最优高度)
BASE_POS = (0.0, 0.0, 0.60)   # 基座世界安装位置


def load_model(scene="scene_draw.xml", base_pos=BASE_POS):
    """加载场景并把基座平移到安装位置 (刚体平移, mj_setConst 重算常量)"""
    model = mujoco.MjModel.from_xml_path(str(ROOT / scene))
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "delta_base")
    model.body_pos[bid] = np.asarray(base_pos, dtype=float)
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)
    return model, data

# ---------------- 轨迹生成器 ----------------
def traj_circle(t, T, c, size):
    w = 2 * np.pi / T
    pos = np.array([c[0] + size * np.cos(w * t), c[1] + size * np.sin(w * t), c[2]])
    vel = np.array([-size * w * np.sin(w * t), size * w * np.cos(w * t), 0.0])
    return pos, vel

def traj_square(t, T, c, size):
    """边长 2*size, 4 段匀速; 角点用 5 次多项式平滑 (限加速度)"""
    half = size
    corners = np.array([[ c[0]+half, c[1]+half], [c[0]-half, c[1]+half],
                        [c[0]-half, c[1]-half], [c[0]+half, c[1]-half]])
    seg = T / 4.0
    tm = t % T
    i = min(int(tm / seg), 3)
    frac = np.clip((tm - i * seg) / seg, 0, 1)
    # 平滑角点: smoothstep 替代线性, 加速度连续
    s = frac**3 * (10 - 15*frac + 6*frac**2)
    p0, p1 = corners[i], corners[(i + 1) % 4]
    pos = np.array([p0[0] + s * (p1[0]-p0[0]), p0[1] + s * (p1[1]-p0[1]), c[2]])
    ds = 30 * frac**2 * (1 - frac)**2 / seg
    vel = np.array([(p1[0]-p0[0]) * ds, (p1[1]-p0[1]) * ds, 0.0])
    return pos, vel

def traj_figure8(t, T, c, size):
    """90mm x 45mm: x = size*sin(2w t), y = 0.5*size*sin(w t)"""
    w = 2 * np.pi / T
    pos = np.array([c[0] + size * np.sin(2*w*t), c[1] + 0.5*size*np.sin(w*t), c[2]])
    vel = np.array([2*size*w*np.cos(2*w*t), 0.5*size*w*np.cos(w*t), 0.0])
    return pos, vel

def traj_star(t, T, c, size):
    """五角星, 外径 size, 内径 0.42 size, 10 段 smoothstep"""
    r_out, r_in = size, 0.42 * size
    verts = []
    for i in range(5):
        a_o = -np.pi/2 + i * 2*np.pi/5
        a_i = a_o + np.pi/5
        verts.append((c[0]+r_out*np.cos(a_o), c[1]+r_out*np.sin(a_o)))
        verts.append((c[0]+r_in*np.cos(a_i), c[1]+r_in*np.sin(a_i)))
    verts = np.array(verts)
    seg = T / 10.0
    tm = t % T
    i = min(int(tm / seg), 9)
    frac = np.clip((tm - i * seg) / seg, 0, 1)
    s = frac**3 * (10 - 15*frac + 6*frac**2)
    p0, p1 = verts[i], verts[(i + 1) % 10]
    pos = np.array([p0[0]+s*(p1[0]-p0[0]), p0[1]+s*(p1[1]-p0[1]), c[2]])
    ds = 30 * frac**2 * (1 - frac)**2 / seg
    vel = np.array([(p1[0]-p0[0])*ds, (p1[1]-p0[1])*ds, 0.0])
    return pos, vel

def traj_helix(t, T, c, size):
    """One-turn cylindrical helix helper (multi-turn handling is in reference_at)."""
    w = 2 * np.pi / T
    pos = np.array([c[0] + size*np.cos(w*t), c[1] + size*np.sin(w*t),
                    c[2] - size/2 + size*t/T])
    vel = np.array([-size*w*np.sin(w*t), size*w*np.cos(w*t),
                    size/T])
    return pos, vel

SHAPES = {
    'circle':  (traj_circle,  0.055, 4.0, '圆形 r=55mm'),
    'square':  (traj_square,  0.040, 5.0, '正方形 边长80mm'),
    'figure8': (traj_figure8, 0.045, 5.0, '8字形 90x45mm'),
    'star':    (traj_star,    0.050, 6.0, '五角星 外径50mm'),
    'helix':   (traj_helix,   0.040, 4.0, '螺旋线 r=40mm z±20mm'),
}


def velocity_ff(model, q_des, v_des):
    """速度前馈: 补偿位置伺服 kv 阻尼造成的稳态速度滞后
       ctrl = q_des + (kv/kp) * (dq/dp) * v_des  (误差从 mm 级降到 0.04mm)"""
    J = kin.jacobian(q_des)
    if J is None:
        return q_des
    kp = model.actuator_gainprm[0, 0]
    kv = -model.actuator_biasprm[0, 2]
    return q_des + (kv / kp) * (J @ v_des)


def motion_phase(t, duration, ramp=0.5):
    """C2 time scaling: trace exactly duration seconds of geometric phase.

    Quintic phase-rate ramps start/end with zero velocity and acceleration.
    Their combined area is ramp, so execution takes duration + ramp seconds.
    """
    def integral(u):
        return 2.5*u**4 - 3*u**5 + u**6

    def rate(u):
        return u**3 * (10 - 15*u + 6*u**2)

    if duration < ramp or ramp <= 0:
        raise ValueError('duration must be >= ramp > 0')
    if t <= 0:
        return 0.0, 0.0
    if t < ramp:
        u = t / ramp
        return ramp * integral(u), rate(u)
    if t <= duration:
        return t - ramp/2, 1.0
    if t < duration + ramp:
        u = (t - duration) / ramp
        return duration - ramp/2 + ramp*(u - integral(u)), 1 - rate(u)
    return duration, 0.0


def reference_at(t, fn, period, center, size, cycles):
    phase, phase_rate = motion_phase(t, cycles * period)
    if fn is traj_helix:
        # A real helix: XY makes ``cycles`` turns while Z rises monotonically
        # over the complete command.  The previous implementation used a
        # second sinusoid in Z, which produced a saddle/wavy curve.
        angle = 2*np.pi*phase/period
        z_half = size/2.0
        pos = np.array([center[0] + size*np.cos(angle),
                        center[1] + size*np.sin(angle),
                        center[2] - z_half + size*phase/(cycles*period)])
        vel = np.array([-size*(2*np.pi/period)*np.sin(angle),
                        size*(2*np.pi/period)*np.cos(angle),
                        size/(cycles*period)]) * phase_rate
        return pos, vel
    pos, vel = fn(phase, period, center, size)
    return pos, vel * phase_rate


def checked_command(model, pos, vel, ff):
    q = kin.ik_safe(pos)
    if q is None:
        raise ValueError(f'Trajectory is outside the safe workspace: {pos}')
    cmd = velocity_ff(model, q, vel) if ff else q
    if (not np.all(np.isfinite(cmd)) or
            np.any(cmd < model.actuator_ctrlrange[:, 0]) or
            np.any(cmd > model.actuator_ctrlrange[:, 1])):
        raise ValueError('Feed-forward command exceeds actuator limits')
    return cmd


def equal_xyz_axes(ax, points_mm):
    """Identical physical scale on all axes; never magnify micron Z errors."""
    lo, hi = points_mm.min(axis=0), points_mm.max(axis=0)
    mid = (lo + hi)/2
    half = max(float(np.max(hi-lo))*0.55, 1.0)
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), mid):
        setter(c-half, c+half)
    ax.set_box_aspect((1, 1, 1))
    ax.ticklabel_format(useOffset=False, style='plain')


def run_shape(model, shape, cycles=2, view=False, ff=True, viewer=None, data=None,
              realtime=True):
    if cycles < 1:
        raise ValueError('cycles must be >= 1')
    fn, size, T, cname = SHAPES[shape]
    data = mujoco.MjData(model) if data is None else data
    mujoco.mj_resetData(model, data)
    iface = kin.DeltaMujocoInterface(model, data)
    center = np.array([0.0, 0.0, Z_DRAW])
    dt = model.opt.timestep
    total_time = cycles*T + 0.5 + 0.5  # phase ramp overhead + stationary hold
    steps = int(round(total_time / dt))
    ts = np.arange(steps + 1) * dt

    # Preflight the complete command sequence, including velocity feed-forward.
    # An unreachable point must fail explicitly, never silently hold a stale IK.
    refs = [reference_at(t, fn, T, center, size, cycles) for t in ts]
    des = np.array([p for p, _ in refs])
    commands = np.array([checked_command(model, p, v, ff) for p, v in refs])
    q_start = kin.ik_safe(des[0])
    iface.set_config(q_start)
    iface.ctrl_set(q_start)
    for _ in range(round(0.8 / dt)):
        mujoco.mj_step(model, data)
    mujoco.mj_fwdPosition(model, data)

    if view and viewer is None:
        import mujoco.viewer as mj_viewer
        context = mj_viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
    else:
        context = nullcontext(viewer)
    base_world = data.site_xpos[iface.sid_base].copy()
    # The target path is in base coordinates; display geometry uses world coordinates.
    display_reference = des if fn is traj_helix else np.array([
        fn(t, T, center, size)[0] for t in np.linspace(0.0, T, 1001)])
    overlay = TrajectoryOverlay(display_reference + base_world)
    act, closure, forces = [], [], []
    print(f"\n--- {cname} | period {T}s x {cycles} ---")
    with context as viewer:
        if viewer is not None:
            configure_viewer(viewer, base_world)
        start_wall = time.perf_counter()
        for step, t in enumerate(ts):
            if viewer is not None and not viewer.is_running():
                break
            # site_xpos is refreshed after integration: desired and measured
            # samples now describe the SAME instant, without a one-step delay.
            act.append(iface.ee_pos().copy())
            overlay.append(data.site_xpos[iface.sid_ee])
            closure.append(iface.connect_residual())
            forces.append(data.actuator_force.copy())
            if viewer is not None and step % max(1, round(0.02/dt)) == 0:
                if realtime:
                    time.sleep(max(0.0, start_wall + t - time.perf_counter()))
                sync_trajectory(viewer, overlay)
            if step == steps:
                break
            data.ctrl[:] = commands[step]
            mujoco.mj_step(model, data)
            mujoco.mj_fwdPosition(model, data)
        if viewer is not None and viewer.is_running():
            sync_trajectory(viewer, overlay)

    if not act:
        raise RuntimeError('Viewer closed before any sample was collected')
    act = np.array(act)
    ts, des = ts[:len(act)], des[:len(act)]
    err_xyz = act - des
    err = np.linalg.norm(err_xyz, axis=1)
    z_err = err_xyz[:, 2]
    # Include startup, every corner, stopping, and the final hold in statistics.
    stats = dict(shape=shape, name=cname, mean=float(err.mean()),
                 rms=float(np.sqrt(np.mean(err**2))), mx=float(err.max()),
                 z_max_m=float(np.max(np.abs(z_err))),
                 z_peak_to_peak_m=float(np.ptp(z_err)),
                 z_error_step_max_m=float(np.max(np.abs(np.diff(z_err)))) if len(act)>1 else 0.0,
                 closure_max_m=float(np.max(closure)),
                 actuator_peak_nm=float(np.max(np.abs(forces))),
                 samples=len(act), sample_period_s=float(dt),
                 completed=bool(len(act) == steps+1))
    print(f"    full-run RMS {stats['rms']*1e3:.4f} mm | "
          f"max {stats['mx']*1e3:.4f} mm | Z max {stats['z_max_m']*1e3:.4f} mm")

    # Real physical proportions in 3D; separate panels expose the unfiltered
    # sub-mm errors instead of stretching them to the height of the whole robot.
    fig = plt.figure(figsize=(16, 9), layout='constrained')
    mm = 1e3
    ax = fig.add_subplot(2, 3, 1)
    ax.plot(des[:,0]*mm, des[:,1]*mm, 'r--', lw=2, label='期望')
    ax.plot(act[:,0]*mm, act[:,1]*mm, 'b-', lw=1, label='实际')
    ax.set(xlabel='X (mm)', ylabel='Y (mm)', title=f'{cname} XY')
    ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.legend()

    ax3d = fig.add_subplot(2, 3, 2, projection='3d')
    ax3d.plot(des[:,0]*mm, des[:,1]*mm, des[:,2]*mm, 'r--', lw=2)
    ax3d.plot(act[:,0]*mm, act[:,1]*mm, act[:,2]*mm, 'b-', lw=1)
    equal_xyz_axes(ax3d, np.vstack((des, act))*mm)
    ax3d.set(xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)',
             title='3D 轨迹：XYZ 等比例')

    axz = fig.add_subplot(2, 3, 3)
    axz.plot(ts, des[:,2]*mm, 'r--', lw=2, label='期望')
    axz.plot(ts, act[:,2]*mm, 'b-', lw=1, label='实际')
    zlo, zhi = np.min(des[:,2])*mm, np.max(des[:,2])*mm
    pad = max(1.0, 0.05*(zhi-zlo))
    axz.set(ylim=(zlo-pad, zhi+pad), xlabel='t (s)', ylabel='Z (mm)',
            title='Z 位置（基座坐标系）')
    axz.ticklabel_format(useOffset=False, style='plain')
    axz.legend(); axz.grid(alpha=0.3)

    axe = fig.add_subplot(2, 3, 4)
    for i, label in enumerate(('X', 'Y', 'Z')):
        axe.plot(ts, err_xyz[:,i]*mm, lw=0.8, label=label)
    axe.set(xlabel='t (s)', ylabel='实际 − 期望 (mm)', title='三轴误差（未滤波）')
    axe.legend(); axe.grid(alpha=0.3)

    axze = fig.add_subplot(2, 3, 5)
    axze.plot(ts, z_err*mm, 'b-', lw=0.8)
    limit = max(0.1, float(np.max(np.abs(z_err)))*mm*1.15)
    axze.set(ylim=(-limit, limit), xlabel='t (s)', ylabel='Z 误差 (mm)',
             title=f"Z 最大绝对误差 {stats['z_max_m']*mm:.4f} mm")
    axze.axhline(0, color='gray', lw=0.5); axze.grid(alpha=0.3)

    axerr = fig.add_subplot(2, 3, 6)
    axerr.plot(ts, err*mm, 'b-', lw=0.8)
    axerr.set(xlabel='t (s)', ylabel='三维误差 (mm)', title='全程误差（含启停及所有转角）')
    axerr.grid(alpha=0.3)
    outdir = ROOT / 'results'
    outdir.mkdir(exist_ok=True)
    options = mujoco.MjvOption()
    options.geomgroup[5] = 0
    with mujoco.Renderer(model, height=900, width=1200) as renderer:
        renderer.update_scene(data, camera=trajectory_camera(base_world), scene_option=options)
        overlay.draw(renderer.scene)
        plt.imsave(outdir / f'traj_{shape}_mujoco.png', renderer.render())
    fig.savefig(outdir / f'traj_{shape}.png', dpi=140)
    plt.close(fig)
    np.savetxt(outdir / f'traj_{shape}.csv', np.column_stack((ts, des, act, err)),
               delimiter=',', fmt='%.10g',
               header='time_s,des_x_m,des_y_m,des_z_m,act_x_m,act_y_m,act_z_m,error_m',
               comments='')
    (outdir / f'traj_{shape}_metrics.json').write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shape', default='all', choices=list(SHAPES)+['all'])
    ap.add_argument('--cycles', type=int, default=2)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--view', action='store_true', help='打开 MuJoCo 窗口（默认）')
    mode.add_argument('--headless', action='store_true', help='仅保存数据、图表和场景截图')
    ap.add_argument('--auto-close', action='store_true', help='全部轨迹结束后自动关闭窗口')
    ap.add_argument('--no-ff', action='store_true', help='关闭速度前馈')
    args = ap.parse_args()

    model, data = load_model()
    for i in range(model.ngeom):
        if model.geom(i).name.startswith('frame_post'):
            model.geom_group[i] = 5
    shapes = list(SHAPES) if args.shape == 'all' else [args.shape]
    if args.headless:
        context = nullcontext(None)
    else:
        from mujoco import viewer as mj_viewer
        iface = kin.DeltaMujocoInterface(model, data)
        iface.set_config(kin.ik_safe(np.array([0.0, 0.0, Z_DRAW])))
        context = mj_viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
    stats = []
    with context as viewer:
        if viewer is not None:
            configure_viewer(viewer, data.body('delta_base').xpos)
            viewer.sync()
        for shape in shapes:
            if viewer is not None and not viewer.is_running():
                break
            result = run_shape(model, shape, cycles=args.cycles, ff=not args.no_ff,
                               viewer=viewer, data=data)
            stats.append(result)
            if not result['completed']:
                break
        if viewer is not None and stats and not args.auto_close:
            print('Trajectories finished. Close the MuJoCo window to exit.')
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)

    print("\n" + "=" * 60)
    print(f"{'轨迹':<14}{'mean (mm)':>10}{'RMS (mm)':>10}{'max (mm)':>10}")
    for s in stats:
        print(f"{s['name']:<14}{s['mean']*1e3:>10.2f}{s['rms']*1e3:>10.2f}{s['mx']*1e3:>10.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
