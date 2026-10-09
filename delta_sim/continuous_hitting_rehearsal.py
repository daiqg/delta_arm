"""Offline rehearsal demonstration: preselect legal serve/return pairs.

Preserved reference implementation. Use continuous_hitting.py for online control.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import mujoco
import numpy as np

import kinematics as kin
from run_hitting import HittingSim, PERCEPT_DELAY, QDOT_MAX, V_PAD_MAX
from trajectory_view import TrajectoryOverlay


# Each centre was checked against two table contacts, net clearance and IK.
# Small random variations are accepted only after the same checks are repeated.
SERVE_FAMILIES = (
    (1.02, -3.4, -2.0),
    (1.02, -3.4, -1.6),
    (1.02, -3.4, -0.8),
    (1.12, -3.0, -2.0),
    (1.12, -3.0, -0.4),
    (1.12, -3.4, -1.2),
    (1.22, -3.0, -1.6),
    (1.22, -3.0, -0.8),
)


def sample_serve(sim, rng, max_attempts=300):
    for _ in range(max_attempts):
        z, vx, vz = SERVE_FAMILIES[int(rng.integers(len(SERVE_FAMILIES)))]
        pos = np.array([1.40, rng.uniform(-0.012, 0.012), z + rng.uniform(-0.005, 0.005)])
        vel = np.array([vx + rng.uniform(-0.025, 0.025),
                        rng.uniform(-0.035, 0.035), vz + rng.uniform(-0.025, 0.025)])
        samples, service = sim.inspect_serve(pos, vel)
        if not service['valid']:
            continue
        inter = sim.choose_intercept(samples,
                                     max(PERCEPT_DELAY + 0.20, service['bounces'][1][0] + 0.015))
        if inter is not None:
            return pos, vel, samples, service, inter
    raise RuntimeError(f'No legal and reachable serve among {max_attempts} candidates')


STROKES = (
    (1.10, 0.0, 0.45),
    (1.00, 0.0, 0.35),
    (0.95, 0.0, 0.60),
    (1.12, 0.0, 0.40),
    (0.85, 0.0, 0.45),
    (0.80, 0.0, 0.35),
    (0.70, 0.0, 0.45),
    (0.90, 0.0, 0.25),
)


def place_launcher(model, pos):
    for name in ('launcher_body', 'launcher_barrel'):
        geom = model.geom(name)
        model.geom_pos[geom.id, 1] = pos[1]
        model.geom_pos[geom.id, 2] = pos[2]
    post = model.geom('launcher_post').id
    model.geom_pos[post, 1] = pos[1]
    model.geom_pos[post, 2] = pos[2] / 2
    model.geom_size[post, 1] = pos[2] / 2


def prepare_ball(sim, rng, seed, max_candidates=300):
    attempts = 0
    for _ in range(max_candidates):
        pos, vel, samples, service, inter = sample_serve(sim, rng)
        for stroke in STROKES:
            stroke = np.array(stroke)
            jac = kin.jacobian(inter[5])
            if (np.linalg.norm(stroke) > V_PAD_MAX or jac is None or
                    np.any(np.abs(jac @ stroke) > QDOT_MAX)):
                continue
            if not sim.inspect_plan(inter[4], stroke, inter[1])['valid']:
                continue
            attempts += 1
            prior_bias = sim.bias.copy()
            preview = sim.run_trial(seed, (0.58, 0.0), record=False,
                                    v_pad_override=stroke,
                                    launch=(pos, vel), incoming=samples,
                                    service=service, finish_recovery=True)
            sim.bias[:] = prior_bias
            if (preview.get('legal_return') and
                    preview['actual_peak_paddle_speed_mps'] <= V_PAD_MAX and
                    preview['actual_peak_joint_speed_radps'] <= QDOT_MAX and
                    preview['actual_min_joint_margin_rad'] >= kin.Q_MARGIN):
                return pos, vel, samples, service, inter, stroke, attempts
    raise RuntimeError(f'No legal return among {attempts} full MuJoCo trial shots')


def run(nballs=10, headless=False, seed=20261007, realtime=True, output=None,
        auto_close=False):
    if nballs < 1:
        raise ValueError('--nballs must be positive')
    viewer = None
    if not headless:
        try:
            import mujoco.viewer as mj_viewer
        except ImportError as exc:
            raise RuntimeError('Activate the mujoco-sim-win environment for the viewer') from exc

    sim = HittingSim()
    rng = np.random.default_rng(seed)
    out = Path(output) if output else Path(__file__).resolve().parent / 'results'
    out.mkdir(parents=True, exist_ok=True)
    results = []
    preview_attempts = 0
    prepared = []
    for i in range(nballs):
        print(f'Preparing legal serve and return {i+1:02d}/{nballs}...', flush=True)
        prepared.append(prepare_ball(sim, rng, seed+i))
    if not headless:
        viewer = mj_viewer.launch_passive(sim.model, sim.data)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = (0.0, 0.0, 0.85)
        viewer.cam.distance = 3.1
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -28
        viewer.opt.geomgroup[5] = 0
        viewer.sync()
    try:
        for i, prepared_ball in enumerate(prepared):
            if viewer is not None and not viewer.is_running():
                break
            pos, vel, samples, service, inter, stroke, attempts = prepared_ball
            preview_attempts += attempts
            place_launcher(sim.model, pos)
            result = sim.run_trial(seed+i, (0.58, 0.0), viewer=viewer,
                                   realtime=realtime, record=True,
                                   v_pad_override=stroke,
                                   launch=(pos, vel), incoming=samples,
                                   service=service, finish_recovery=True)
            result['launch_pos'] = pos
            result['launch_vel'] = vel
            result['preview_attempts'] = attempts
            results.append(result)
            landing = result['landing']
            print(f'ball {i+1:02d}/{nballs}: '
                  f'serve={"legal" if result.get("actual_serve_valid") else "invalid"} '
                  f'reachable={"yes" if "intercept" in result else "no"} '
                  f'paddle={result.get("contact") is not None} '
                  f'return={"legal" if result.get("legal_return") else "miss"} '
                  f'landing={None if landing is None else np.round(landing, 3)}', flush=True)
            if result['outcome'] == 'viewer-closed':
                break
    except BaseException:
        if viewer is not None:
            viewer.close()
        raise

    rows = []
    for i, r in enumerate(results, 1):
        bounce = r.get('actual_serve_bounces', [])
        net = r.get('actual_serve_net_cross')
        landing = r.get('landing')
        return_net = r.get('return_net_cross')
        rows.append(dict(ball=i, launch_x_m=r['launch_pos'][0], launch_y_m=r['launch_pos'][1],
                         launch_z_m=r['launch_pos'][2], launch_vx_mps=r['launch_vel'][0],
                         launch_vy_mps=r['launch_vel'][1], launch_vz_mps=r['launch_vel'][2],
                         serve_bounce_1_x_m=bounce[0][1][0] if len(bounce) > 0 else None,
                         serve_bounce_1_y_m=bounce[0][1][1] if len(bounce) > 0 else None,
                         serve_net_z_m=net[1][2] if net else None,
                         serve_bounce_2_x_m=bounce[1][1][0] if len(bounce) > 1 else None,
                         serve_bounce_2_y_m=bounce[1][1][1] if len(bounce) > 1 else None,
                         actual_serve_valid=int(r.get('actual_serve_valid', False)),
                         intercept_time_s=r['intercept'][0] if 'intercept' in r else None,
                         intercept_x_m=r['intercept'][1][0] if 'intercept' in r else None,
                         intercept_y_m=r['intercept'][1][1] if 'intercept' in r else None,
                         intercept_z_m=r['intercept'][1][2] if 'intercept' in r else None,
                         paddle_contact=int(r.get('contact') is not None),
                         paddle_vx_mps=r['v_pad'][0] if 'v_pad' in r else None,
                         paddle_vy_mps=r['v_pad'][1] if 'v_pad' in r else None,
                         paddle_vz_mps=r['v_pad'][2] if 'v_pad' in r else None,
                         preview_attempts=r['preview_attempts'],
                         planned_peak_paddle_speed_mps=r['plan_metrics']['peak_paddle_speed_mps'],
                         planned_peak_joint_speed_radps=r['plan_metrics']['peak_joint_speed_radps'],
                         actual_peak_joint_speed_radps=r['actual_peak_joint_speed_radps'],
                         actual_peak_paddle_speed_mps=r['actual_peak_paddle_speed_mps'],
                         actual_min_joint_margin_rad=r['actual_min_joint_margin_rad'],
                         return_net_z_m=return_net[1][2] if return_net else None,
                         return_net_contact=int(r.get('return_net_contact', False)),
                         return_landing_x_m=landing[0] if landing else None,
                         return_landing_y_m=landing[1] if landing else None,
                         legal_return=int(r.get('legal_return', False)),
                         target_error_m=r.get('err')))
    with (out / 'random_10_balls.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys()) if rows else None
        if writer:
            writer.writeheader()
            writer.writerows(rows)
    with (out / 'random_10_trajectories.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['ball', 'time_s', 'ball_x_m', 'ball_y_m', 'ball_z_m',
                         'paddle_x_m', 'paddle_y_m', 'paddle_z_m'])
        for i, result in enumerate(results, 1):
            for (t, ball), (_, paddle) in zip(result['log']['ball'], result['log']['paddle']):
                writer.writerow([i, t, *ball, *paddle])
    summary = dict(seed=seed, requested_balls=nballs, simulated_balls=len(results),
                   preview_attempts=preview_attempts,
                   legal_serves=sum(r.get('actual_serve_valid', False) for r in results),
                   reachable_serves=sum('intercept' in r for r in results),
                   paddle_contacts=sum(r.get('contact') is not None for r in results),
                   legal_returns=sum(r.get('legal_return', False) for r in results),
                   target_hits_100mm=sum(r.get('outcome') == 'success' for r in results))
    summary['mujoco_version'] = mujoco.__version__
    summary['paddle_speed_limit_mps'] = V_PAD_MAX
    summary['joint_speed_limit_radps'] = QDOT_MAX
    summary['motion_within_limits'] = sum(
        r['plan_metrics']['valid'] and r['actual_peak_joint_speed_radps'] <= QDOT_MAX
        and r['actual_peak_paddle_speed_mps'] <= V_PAD_MAX
        and r['actual_min_joint_margin_rad'] >= kin.Q_MARGIN for r in results)
    (out / 'random_10_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    try:
        import matplotlib.pyplot as plt
        renderer = mujoco.Renderer(sim.model, height=720, width=1280)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = (0.0, 0.0, 0.85)
        camera.distance = 3.1
        camera.azimuth = 90
        camera.elevation = -28
        options = mujoco.MjvOption()
        options.geomgroup[5] = 0
        renderer.update_scene(sim.data, camera=camera, scene_option=options)
        if results:
            overlay = TrajectoryOverlay([])
            for _, position in results[-1]['log']['ball']:
                overlay.append(position)
            overlay.draw(renderer.scene)
        plt.imsave(out / 'random_10_final.png', renderer.render())
        renderer.close()
    except Exception as exc:
        print(f'Final frame unavailable: {exc}')
    print(json.dumps(summary, indent=2))
    if viewer is not None:
        try:
            while viewer.is_running() and not auto_close:
                viewer.sync()
                time.sleep(0.02)
        finally:
            viewer.close()
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nballs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20261007)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--no-realtime', action='store_true')
    parser.add_argument('--auto-close', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run(args.nballs, args.headless, args.seed, not args.no_realtime,
        args.output, args.auto_close)
