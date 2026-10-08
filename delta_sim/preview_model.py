"""Inspect the actual simulation model in a close-up MuJoCo view."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt
import mujoco
import numpy as np

import kinematics as kin


ROOT = Path(__file__).resolve().parent


def load_preview(paddle=False):
    if paddle:
        from run_hitting import HittingSim, HOME_PLAT

        sim = HittingSim()
        model, data, iface = sim.model, sim.data, sim.iface
        target = HOME_PLAT
    else:
        from run_trajectory import load_model

        model, data = load_model()
        iface = kin.DeltaMujocoInterface(model, data)
        target = np.array([0.0, 0.0, -0.21])

    q = kin.ik_safe(target)
    if q is None:
        raise RuntimeError("Preview pose is outside the safe workspace")
    iface.set_config(q)
    iface.ctrl_set(q)
    for _ in range(round(0.3 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    base = data.body("delta_base").xpos.copy()
    camera = mujoco.MjvCamera()
    camera.lookat[:] = base + [0.0, 0.0, -0.13 if paddle else -0.10]
    camera.distance = 0.85 if paddle else 0.65
    camera.azimuth = 145
    camera.elevation = -18
    return model, data, camera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paddle", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--show-fixtures", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model, data, camera = load_preview(args.paddle)
    # Hide only the visual mounting frame when inspecting small CAD components.
    if not args.show_fixtures:
        for i in range(model.ngeom):
            if model.geom(i).name.startswith(("frame_post", "gantry_")):
                model.geom_group[i] = 5
    options = mujoco.MjvOption()
    options.geomgroup[5] = int(args.show_fixtures)
    output = args.output or ROOT / "results" / (
        "model_paddle.png" if args.paddle else "model_delta.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=900, width=1200) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=options)
        plt.imsave(output, renderer.render())
    print(f"MuJoCo model preview: {output.resolve()}")

    if not args.headless:
        from mujoco import viewer as mj_viewer

        with mj_viewer.launch_passive(model, data) as viewer:
            with viewer.lock():
                viewer.opt.geomgroup[5] = options.geomgroup[5]
                viewer.cam.lookat[:] = camera.lookat
                viewer.cam.distance = camera.distance
                viewer.cam.azimuth = camera.azimuth
                viewer.cam.elevation = camera.elevation
            while viewer.is_running():
                viewer.sync()
                time.sleep(1 / 60)


if __name__ == "__main__":
    main()
