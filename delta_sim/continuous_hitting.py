"""Causal online Delta-paddle table-tennis demonstration.

Serves are sampled only until they are rule-compliant and geometrically within
the robot workspace.  After launch, the controller sees delayed noisy ball
positions and encoder feedback; it neither receives the launch state nor runs
the robot plant ahead to preselect a return.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

import kinematics as kin
from online_control import Observation, OnlineController, flat_paddle_model, COMPUTE_BUDGET
from run_hitting import (BASE_POS, BALL_R, EVAL_TIMEOUT, NET_TOP, QDOT_MAX,
                         V_PAD_MAX, HittingSim, ball_qv, set_ball)
from trajectory_view import TrajectoryOverlay, sync_trajectory


# This compact, legal launch envelope is chosen from the physical workspace
# and controller lead-time budget, not from a return pre-rollout.  The launch
# pose and all three velocity components remain random for each ball.
ONLINE_SERVE_FAMILIES = ((1.02, -3.4, -2.0),)


def place_launcher(model, pos):
    """Align launcher visuals with the sampled release position."""
    for name in ('launcher_body', 'launcher_barrel'):
        geom = model.geom(name)
        model.geom_pos[geom.id, 1] = pos[1]
        model.geom_pos[geom.id, 2] = pos[2]
    post = model.geom('launcher_post').id
    model.geom_pos[post, 1] = pos[1]
    model.geom_pos[post, 2] = pos[2] / 2
    model.geom_size[post, 1] = pos[2] / 2


def sample_legal_reachable_serve(sim, rng, max_attempts=300):
    """Screen the launcher only for serve rules and static workspace overlap.

    ``samples`` and ``intercept`` deliberately stay in this function.  They
    are not given to the online controller after the ball is launched.
    """
    for attempt in range(1, max_attempts + 1):
        z, vx, vz = ONLINE_SERVE_FAMILIES[int(rng.integers(len(ONLINE_SERVE_FAMILIES)))]
        position = np.array([1.40, rng.uniform(-0.012, 0.012),
                             z + rng.uniform(-0.005, 0.005)])
        velocity = np.array([vx + rng.uniform(-0.025, 0.025),
                             rng.uniform(-0.035, 0.035),
                             vz + rng.uniform(-0.025, 0.025)])
        samples, service = sim.inspect_serve(position, velocity)
        if not service['valid']:
            continue
        # This is launch-envelope validation, not a planned return.  It only
        # guarantees that the post-bounce flight intersects the safe workspace.
        intercept = sim.choose_intercept(samples, service['bounces'][1][0] + 0.015)
        if intercept is not None:
            return position, velocity, service, attempt
    raise RuntimeError(f'No legal, geometrically reachable serve in {max_attempts} draws')


def has_pair(data, first, second):
    return any({data.contact[i].geom1, data.contact[i].geom2} == {first, second}
               for i in range(data.ncon))


def initialise_trial(sim, position, velocity):
    model, data = sim.model, sim.data
    mujoco.mj_resetData(model, data)
    sim.iface.set_config(sim.q_home)
    sim.iface.ctrl_set(sim.q_home)
    for _ in range(500):
        mujoco.mj_step(model, data)
    set_ball(data, sim.qadr, sim.vadr, position, velocity)
    mujoco.mj_forward(model, data)
    return data.time


def run_ball(sim, position, velocity, service, ball_number, rng, viewer,
             realtime, measurement_delay, measurement_noise, control_period):
    model, data = sim.model, sim.data
    launch_time = initialise_trial(sim, position, velocity)
    controller = OnlineController(model, period=control_period)
    overlay = TrajectoryOverlay([]) if viewer is not None else None
    observations = deque()
    log = dict(ball=[], paddle=[], estimate=[], prediction=[], contact=None,
               serve_bounces=[], serve_net_cross=None, serve_net_contact=False,
               return_net_cross=None, return_net_contact=False, landing=None)
    contacted = False
    touching_table = False
    return_landed = False
    previous_x = position[0]
    next_sample = 0.0
    peak_joint_speed = peak_paddle_speed = 0.0
    min_joint_margin = float('inf')
    error_samples = []
    wall_start = time.perf_counter()
    viewer_closed = False

    # 2 kHz plant, 500 Hz servo/measurement handling.  Camera updates are 50 Hz.
    for step in range(int((EVAL_TIMEOUT + 1.0) / model.opt.timestep)):
        now = data.time - launch_time
        if now + 1e-9 >= next_sample:
            measured = data.qpos[sim.qadr:sim.qadr+3].copy()
            measured += rng.normal(0.0, measurement_noise, 3)
            observations.append((Observation(now, now + measurement_delay, measured),
                                 data.qpos[sim.qadr:sim.qadr+3].copy()))
            next_sample += 0.010
        while observations and observations[0][0].delivered <= now:
            observation, capture_truth = observations.popleft()
            controller.estimator.observe(observation)
            state = controller.estimator.state()
            if state is not None and not contacted:
                # Ground truth is used only for an offline estimator metric.
                error_samples.append(float(np.linalg.norm(
                    state[1] - capture_truth)))

        if step % 4 == 0:
            controller.update(now, data.qpos[sim.iface.qadr].copy(), contacted)
            command = controller.command(now)
            if command is not None:
                data.ctrl[:] = command
            else:
                data.ctrl[:] = sim.q_home

        mujoco.mj_step(model, data)
        ball = data.qpos[sim.qadr:sim.qadr+3].copy()
        paddle_contact = has_pair(data, sim.gid_ball, sim.gid_face)
        table_contact = has_pair(data, sim.gid_ball, sim.gid_table)
        net_contact = has_pair(data, sim.gid_ball, sim.gid_net)
        if paddle_contact and not contacted:
            contacted = True
            log['contact'] = (float(now), ball.copy())

        if not contacted:
            if table_contact and not touching_table:
                log['serve_bounces'].append((float(now), ball.copy()))
            if net_contact:
                log['serve_net_contact'] = True
            if previous_x > 0 >= ball[0] and log['serve_net_cross'] is None:
                log['serve_net_cross'] = (float(now), ball.copy())
        else:
            if net_contact and not return_landed:
                log['return_net_contact'] = True
            if previous_x <= 0 < ball[0] and log['return_net_cross'] is None:
                log['return_net_cross'] = (float(now), ball.copy())
            if table_contact and not touching_table and not return_landed:
                return_landed = True
                if 0.02 < ball[0] < 1.35 and abs(ball[1]) < 0.74:
                    log['landing'] = (float(ball[0]), float(ball[1]))
        previous_x = ball[0]
        touching_table = table_contact

        if step % 4 == 0:
            log['ball'].append((float(now), ball.copy()))
            mujoco.mj_fwdPosition(model, data)
            log['paddle'].append((float(now), data.site_xpos[sim.sid_paddle].copy()))
            state = controller.estimator.state()
            if state is not None:
                log['estimate'].append((float(now), state[1].copy()))
            # ``prediction`` is refreshed by OnlineController.choose() only.
            # Do not rerun the ball rollout from this 500 Hz data logger.
            log['prediction'] = controller.prediction
            site_velocity = np.zeros(6)
            mujoco.mj_fwdVelocity(model, data)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE,
                                     sim.sid_paddle, site_velocity, 0)
            peak_paddle_speed = max(peak_paddle_speed,
                                    float(np.linalg.norm(site_velocity[3:])))
        peak_joint_speed = max(peak_joint_speed,
                               float(np.max(np.abs(data.qvel[sim.motor_dofs]))))
        angles = data.qpos[sim.iface.qadr]
        min_joint_margin = min(min_joint_margin, float(np.min(angles-kin.Q_PHYS_MIN)),
                               float(np.min(kin.Q_PHYS_MAX-angles)))

        if viewer is not None and step % 40 == 0:
            if not viewer.is_running():
                viewer_closed = True
                break
            # The reference trajectory comes only from the latest online prediction.
            overlay.reference = np.asarray(controller.prediction, dtype=float)
            overlay._revision += 1
            overlay.append(ball)
            sync_trajectory(viewer, overlay)
            if realtime:
                time.sleep(max(0.0, now-(time.perf_counter()-wall_start)))

        if return_landed and now > (log['contact'][0] if log['contact'] else 0) + 0.60:
            break
        if contacted and ball[2] < 0.10:
            break
        if now > EVAL_TIMEOUT:
            break

    bounces = log['serve_bounces']
    net_cross = log['serve_net_cross']
    actual_serve_valid = bool(len(bounces) == 2 and net_cross is not None and
        not log['serve_net_contact'] and 0.08 < bounces[0][1][0] < 1.32 and
        -1.32 < bounces[1][1][0] < -0.08 and
        all(abs(item[1][1]) < 0.70 for item in bounces) and
        bounces[0][0] < net_cross[0] < bounces[1][0] and
        net_cross[1][2] > NET_TOP + BALL_R + 0.015)
    return_net = log['return_net_cross']
    legal_return = bool(actual_serve_valid and contacted and log['landing'] is not None
        and return_net is not None and return_net[1][2] > NET_TOP + BALL_R + 0.012
        and not log['return_net_contact'])
    compute = [event['compute_s'] for event in controller.events]
    result = dict(ball=ball_number, launch_pos=position, launch_vel=velocity,
                  launch_service_valid=int(service['valid']), actual_serve_valid=int(actual_serve_valid),
                  reachable_by_geometry=1, paddle_contact=int(contacted),
                  legal_return=int(legal_return), landing=log['landing'],
                  serve_bounces=bounces, serve_net_cross=net_cross,
                  return_net_cross=return_net, return_net_contact=int(log['return_net_contact']),
                  contact=log['contact'], measurement_count=controller.estimator.count,
                  prediction_update_count=len(controller.events), controller_events=controller.events,
                  controller_compute_mean_s=float(np.mean(compute)) if compute else None,
                  controller_compute_max_s=float(np.max(compute)) if compute else None,
                  controller_deadline_misses=sum(e['status'] == 'deadline-miss' for e in controller.events),
                  controller_too_late=sum(e['status'] == 'too-late' for e in controller.events),
                  controller_plan_activations=controller.activations,
                  controller_command_faults=controller.command_faults,
                  estimator_position_rmse_m=float(np.sqrt(np.mean(np.square(error_samples)))) if error_samples else None,
                  actual_peak_joint_speed_radps=peak_joint_speed,
                  actual_peak_paddle_speed_mps=peak_paddle_speed,
                  actual_min_joint_margin_rad=min_joint_margin,
                  log=log, outcome='viewer-closed' if viewer_closed else
                      ('success' if legal_return else ('contact-no-return' if contacted else 'missed')))
    return result, viewer_closed


def write_outputs(sim, results, out, nballs, seed, measurement_delay,
                  measurement_noise, control_period):
    rows = []
    for result in results:
        bounces = result['serve_bounces']
        net = result['serve_net_cross']
        ret = result['return_net_cross']
        landing = result['landing']
        rows.append(dict(
            ball=result['ball'], launch_x_m=result['launch_pos'][0],
            launch_y_m=result['launch_pos'][1], launch_z_m=result['launch_pos'][2],
            launch_vx_mps=result['launch_vel'][0], launch_vy_mps=result['launch_vel'][1],
            launch_vz_mps=result['launch_vel'][2],
            serve_bounce_1_x_m=bounces[0][1][0] if len(bounces) > 0 else None,
            serve_bounce_2_x_m=bounces[1][1][0] if len(bounces) > 1 else None,
            serve_net_z_m=net[1][2] if net else None,
            actual_serve_valid=result['actual_serve_valid'],
            paddle_contact=result['paddle_contact'], legal_return=result['legal_return'],
            return_net_contact=result['return_net_contact'],
            return_net_z_m=ret[1][2] if ret else None,
            return_landing_x_m=landing[0] if landing else None,
            return_landing_y_m=landing[1] if landing else None,
            measurement_count=result['measurement_count'],
            prediction_update_count=result['prediction_update_count'],
            controller_compute_mean_s=result['controller_compute_mean_s'],
            controller_compute_max_s=result['controller_compute_max_s'],
            controller_deadline_misses=result['controller_deadline_misses'],
            controller_too_late=result['controller_too_late'],
            controller_command_faults=result['controller_command_faults'],
            estimator_position_rmse_m=result['estimator_position_rmse_m'],
            actual_peak_paddle_speed_mps=result['actual_peak_paddle_speed_mps'],
            actual_peak_joint_speed_radps=result['actual_peak_joint_speed_radps'],
            actual_min_joint_margin_rad=result['actual_min_joint_margin_rad'],
            outcome=result['outcome']))
    stem = f'online_{nballs}'
    with (out / f'{stem}_balls.csv').open('w', newline='', encoding='utf-8') as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    with (out / f'{stem}_trajectories.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['ball', 'time_s', 'ball_x_m', 'ball_y_m', 'ball_z_m',
                         'paddle_x_m', 'paddle_y_m', 'paddle_z_m'])
        for result in results:
            for (t, ball), (_, paddle) in zip(result['log']['ball'], result['log']['paddle']):
                writer.writerow([result['ball'], t, *ball, *paddle])
    (out / f'{stem}_controller.json').write_text(json.dumps(
        [dict(ball=r['ball'], events=r['controller_events'],
              activation_times_s=r['controller_plan_activations']) for r in results], indent=2), encoding='utf-8')
    summary = dict(mode='causal_online_receding_horizon', seed=seed,
                   requested_balls=nballs, simulated_balls=len(results),
                   measurement_delay_s=measurement_delay,
                   measurement_noise_std_m=measurement_noise,
                   control_period_s=control_period, compute_budget_s=COMPUTE_BUDGET,
                   paddle_collision_model="planar_elliptical_prism", mujoco_version=mujoco.__version__,
                   legal_serves=sum(r['actual_serve_valid'] for r in results),
                   reachable_serves=sum(r['reachable_by_geometry'] for r in results),
                   paddle_contacts=sum(r['paddle_contact'] for r in results),
                   legal_returns=sum(r['legal_return'] for r in results),
                   controller_deadline_misses=sum(r['controller_deadline_misses'] for r in results),
                   controller_too_late=sum(r['controller_too_late'] for r in results),
                   max_paddle_speed_mps=max((r['actual_peak_paddle_speed_mps'] for r in results), default=0),
                   max_joint_speed_radps=max((r['actual_peak_joint_speed_radps'] for r in results), default=0),
                   min_joint_margin_rad=min((r['actual_min_joint_margin_rad'] for r in results), default=None),
                   motion_within_limits=sum(r['actual_peak_paddle_speed_mps'] <= V_PAD_MAX
                       and r['actual_peak_joint_speed_radps'] <= QDOT_MAX
                       and r['actual_min_joint_margin_rad'] >= kin.Q_MARGIN
                       and r['controller_command_faults'] == 0 for r in results),
                   notes=['No return trajectory was screened before launch.',
                          'Ground-truth ball velocity is not an online controller input.',
                          'Estimator error is offline evaluation only.'])
    (out / f'{stem}_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    try:
        import matplotlib.pyplot as plt
        renderer = mujoco.Renderer(sim.model, height=720, width=1280)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = (0.0, 0.0, 0.85)
        camera.distance, camera.azimuth, camera.elevation = 3.1, 90, -28
        renderer.update_scene(sim.data, camera=camera)
        if results:
            overlay = TrajectoryOverlay([])
            for _, point in results[-1]['log']['ball']:
                overlay.append(point)
            overlay.draw(renderer.scene)
        plt.imsave(out / f'{stem}_final.png', renderer.render())
        renderer.close()
    except Exception as exc:
        print(f'Final frame unavailable: {exc}')
    return summary


def run(nballs=10, headless=False, seed=20261007, realtime=True, output=None,
        auto_close=False, measurement_delay=0.08, measurement_noise_mm=2.0,
        control_period=0.06):
    if nballs < 1:
        raise ValueError('--nballs must be positive')
    if measurement_delay < 0 or measurement_noise_mm < 0 or control_period <= 0:
        raise ValueError('delay/noise must be non-negative and control period positive')
    viewer = None
    if not headless:
        try:
            import mujoco.viewer as mj_viewer
        except ImportError as exc:
            raise RuntimeError('Activate the mujoco-sim-win environment for the viewer') from exc
    sim = HittingSim()
    sim.model = mujoco.MjModel.from_xml_string(ET.tostring(flat_paddle_model(), encoding='unicode'),
        assets={str(p):p.read_bytes() for p in Path('meshes').glob('*.STL')})
    sim.model.body_pos[sim.model.body('delta_base').id] = BASE_POS
    sim.data = mujoco.MjData(sim.model)
    mujoco.mj_setConst(sim.model, sim.data)
    sim.iface = kin.DeltaMujocoInterface(sim.model, sim.data)
    sim.qadr, sim.vadr = ball_qv(sim.model)
    sim.gid_ball = sim.model.geom('ball_geom').id
    sim.gid_face = sim.model.geom('paddle_face').id
    sim.gid_table = sim.model.geom('table_top').id
    sim.gid_net = sim.model.geom('net').id
    sim.sid_paddle = sim.model.site('paddle_center').id
    sim.motor_dofs = sim.model.jnt_dofadr[sim.iface.jids]
    rng = np.random.default_rng(seed)
    out = Path(output) if output else Path(__file__).resolve().parent / 'results'
    out.mkdir(parents=True, exist_ok=True)
    if not headless:
        viewer = mj_viewer.launch_passive(sim.model, sim.data)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = (0.0, 0.0, 0.85)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.1, 90, -28
        viewer.opt.geomgroup[5] = 0
        viewer.sync()
    results = []
    try:
        for index in range(nballs):
            if viewer is not None and not viewer.is_running():
                break
            position, velocity, service, attempts = sample_legal_reachable_serve(sim, rng)
            place_launcher(sim.model, position)
            print(f'Ball {index+1:02d}/{nballs}: legal launch selected after {attempts} draw(s).', flush=True)
            observation_rng = np.random.default_rng(np.random.SeedSequence([seed, index, 1]))
            result, stopped = run_ball(sim, position, velocity, service, index+1, observation_rng, viewer,
                realtime, measurement_delay, measurement_noise_mm / 1000.0, control_period)
            results.append(result)
            print(f"  contact={bool(result['paddle_contact'])} return={bool(result['legal_return'])} "
                  f"events={result['prediction_update_count']} "
                  f"deadline_miss={result['controller_deadline_misses']} "
                  f"landing={result['landing']}", flush=True)
            if stopped:
                break
    finally:
        summary = write_outputs(sim, results, out, nballs, seed, measurement_delay,
                                measurement_noise_mm / 1000.0, control_period)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nballs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20261007)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--no-realtime', action='store_true')
    parser.add_argument('--auto-close', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--measurement-delay-ms', type=float, default=80.0)
    parser.add_argument('--measurement-noise-mm', type=float, default=2.0)
    parser.add_argument('--control-period-ms', type=float, default=60.0)
    args = parser.parse_args()
    run(args.nballs, args.headless, args.seed, not args.no_realtime, args.output,
        args.auto_close, args.measurement_delay_ms / 1000.0,
        args.measurement_noise_mm, args.control_period_ms / 1000.0)
