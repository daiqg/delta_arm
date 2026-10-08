"""Check paddle mounting transforms in compiled MuJoCo poses (metres)."""
import numpy as np
import mujoco

import kinematics as kin
from run_hitting import HittingSim, P_OFF, N


def main():
    sim = HittingSim()
    poses = [(-0.055, 0.0, -0.215), (0.0, 0.0, -0.21),
             (0.025, 0.02, -0.22), (-0.025, -0.02, -0.22)]
    for pose in poses:
        q = kin.ik_safe(np.asarray(pose))
        if q is None:
            raise AssertionError(f"Unreachable verification pose: {pose}")
        sim.iface.set_config(q)
        mujoco.mj_forward(sim.model, sim.data)
        data = sim.data
        ee = data.site('ee_center').xpos
        mount = data.site('paddle_mount_center').xpos
        centre = data.site('paddle_center').xpos
        top = data.site('paddle_handle_top').xpos
        bottom = data.site('paddle_handle_bottom').xpos
        rotation = data.body('paddle').xmat.reshape(3, 3)
        np.testing.assert_allclose(mount, ee, atol=1e-9)
        np.testing.assert_allclose(centre - ee, P_OFF, atol=1e-6)
        np.testing.assert_allclose(bottom - top, [0, 0, -0.09], atol=1e-6)
        np.testing.assert_allclose(rotation[:, 0], [0, 0, -1], atol=1e-6)
        np.testing.assert_allclose(rotation[:, 2], N, atol=1e-6)
        print(f"pose={pose}  mount_xy={mount[:2].round(4)}  blade_z={centre[2]:.4f}")
    print("Paddle mount aligned and vertical in all verification poses.")


if __name__ == '__main__':
    main()
