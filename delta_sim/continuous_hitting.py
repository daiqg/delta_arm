"""连续 10 球 MuJoCo 击球演示。

默认打开 MuJoCo viewer：
    python continuous_hitting.py
无图形环境：
    python continuous_hitting.py --headless

球以固定间隔从发球机抛出。第一球使用完整的碰撞复演求解拍速，后续球
复用已经标定的拍速并重新计算拦截时刻，因此能够连续运行而不会每球等待
几十秒的候选搜索。每一球仍然经过 MuJoCo 的真实球-拍面、球-球台碰撞。
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import mujoco
try:
    import mujoco.viewer as mj_viewer
except ImportError:
    mj_viewer = None
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_hitting import HittingSim


def run(nballs=10, headless=False, seed=20261007, realtime=True):
    if not headless and mj_viewer is None:
        raise RuntimeError(
            'MuJoCo viewer is unavailable in this Python environment. '
            'Run with the mujoco-sim-win environment: '
            'conda run -n mujoco-sim-win python continuous_hitting.py')
    sim = HittingSim()
    sim.calibrate_impact()
    target = np.array([0.58, 0.0])  # central repeatable target on opponent side
    results = []
    cached_vpad = None
    viewer = None

    if not headless:
        try:
            viewer = mj_viewer.launch_passive(sim.model, sim.data)
            viewer.sync()
        except Exception as exc:
            raise RuntimeError(
                f'Unable to open MuJoCo viewer: {exc}. '
                'Check that this command runs in mujoco-sim-win and that an '
                'OpenGL display is available.') from exc

    try:
        for i in range(nballs):
            if viewer is not None and not viewer.is_running():
                break
            result = sim.run_trial(
                seed + i, target, record=True, viewer=viewer,
                realtime=realtime, v_pad_override=cached_vpad,
                launch_jitter=0.0)
            if cached_vpad is None and 'v_pad' in result:
                cached_vpad = result['v_pad']
            results.append(result)
            landing = result.get('landing')
            err = result.get('err')
            print(f'ball {i+1:02d}/{nballs}: {result["outcome"]:>12s}  '
                  f'contact={result.get("contact") is not None}  '
                  f'landing={None if landing is None else np.round(landing, 3)}  '
                  f'error={None if err is None else round(err*1000, 1)} mm')
    finally:
        if viewer is not None:
            viewer.sync()
            # Leave the final scene visible until the user closes the viewer.
            while viewer.is_running() and not headless:
                viewer.sync()
                time.sleep(0.02)

    landed = [r for r in results if r.get('landing') is not None]
    contacted = [r for r in results if r.get('contact') is not None]
    errors = [r['err'] for r in results if r.get('err') is not None]
    summary = {
        'requested_balls': nballs,
        'simulated_balls': len(results),
        'contacts': len(contacted),
        'landings': len(landed),
        'success_100mm': sum(r.get('outcome') == 'success' for r in results),
        'mean_landing_error_m': float(np.mean(errors)) if errors else None,
        'max_landing_error_m': float(np.max(errors)) if errors else None,
    }
    out = Path(__file__).resolve().parent / 'results'
    out.mkdir(exist_ok=True)
    rows = []
    for i, r in enumerate(results, 1):
        landing = r.get('landing') or (np.nan, np.nan)
        contact = r.get('contact')
        rows.append([i, float(contact[0]) if contact else np.nan,
                     landing[0], landing[1], r.get('err') or np.nan,
                     float(r.get('outcome') == 'success')])
    np.savetxt(out / 'continuous_10_landing.csv', np.asarray(rows), delimiter=',',
               header='ball,contact_time_s,landing_x_m,landing_y_m,error_m,success',
               comments='')
    try:
        renderer = mujoco.Renderer(sim.model, height=720, width=1280)
        renderer.update_scene(sim.data, camera=-1)
        plt.imsave(out / 'continuous_final.png', renderer.render())
        renderer.close()
    except Exception as exc:
        print(f'Final MuJoCo frame was not saved: {exc}')
    (out / 'continuous_10_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nballs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20261007)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--no-realtime', action='store_true')
    args = parser.parse_args()
    run(args.nballs, args.headless, args.seed, not args.no_realtime)
