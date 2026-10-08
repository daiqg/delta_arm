"""Render-only trajectory geometry shared by the viewer and saved frames."""
import mujoco
import numpy as np


def sample_polyline(points, max_points):
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return points.copy()
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    points = points[np.r_[True, lengths > 1e-10]]
    if len(points) < 2:
        return points
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    count = min(max_points, max(2, int(np.ceil(distance[-1] / 0.001)) + 1))
    samples = np.linspace(0.0, distance[-1], count)
    return np.column_stack([np.interp(samples, distance, points[:, i]) for i in range(3)])


def trajectory_camera(base_world):
    camera = mujoco.MjvCamera()
    camera.lookat[:] = np.asarray(base_world) + [0.0, 0.0, -0.13]
    camera.distance = 0.68
    camera.azimuth = 135
    camera.elevation = -28
    return camera


class TrajectoryOverlay:
    def __init__(self, desired_world):
        self.reference = sample_polyline(desired_world, 801)
        self.actual = []
        self.tip = None
        self.spacing = 0.0003
        self._revision = 0
        self._live_scene = None
        self._live_revision = -1
        self._live_segments = 0
        self._live_end = 0

    def append(self, position_world):
        position = np.asarray(position_world, dtype=float).copy()
        self.tip = position
        if not self.actual or np.linalg.norm(position - self.actual[-1]) >= self.spacing:
            self.actual.append(position)
        # Only the display is simplified. Physics and CSV keep every 1 ms sample.
        if len(self.actual) > 1200:
            self.actual = self.actual[::2]
            self.spacing *= 2
            self._revision += 1

    @staticmethod
    def _segment(scene, start, end, actual=False):
        if np.linalg.norm(end - start) < 1e-10:
            return
        kind = mujoco.mjtGeom.mjGEOM_CAPSULE if actual else mujoco.mjtGeom.mjGEOM_LINE
        color = [1.0, 0.38, 0.04, 1.0] if actual else [0.10, 0.70, 1.0, 1.0]
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, kind, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                           np.asarray(color, dtype=np.float32))
        mujoco.mjv_connector(geom, kind, 0.0006 if actual else 3.0, start, end)
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        geom.emission = 0.7
        scene.ngeom += 1

    def _marker(self, scene):
        if self.tip is None:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                           np.full(3, 0.002), self.tip, np.eye(3).ravel(),
                           np.array([1.0, 0.38, 0.04, 1.0], dtype=np.float32))
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        geom.emission = 0.7
        scene.ngeom += 1

    def draw_live(self, scene):
        if len(self.reference)//2 + len(self.actual) + 2 > scene.maxgeom:
            raise RuntimeError("MuJoCo scene has insufficient capacity for trajectory geometry")
        if self._live_scene is not scene or self._live_revision != self._revision:
            scene.ngeom = 0
            for start, end in zip(self.reference[:-1:2], self.reference[1::2]):
                self._segment(scene, start, end)
            self._live_scene = scene
            self._live_revision = self._revision
            self._live_segments = 0
            self._live_end = scene.ngeom

        # Reuse completed segments; only the moving tip and its short tail change.
        scene.ngeom = self._live_end
        count = max(0, len(self.actual) - 1)
        for i in range(self._live_segments, count):
            self._segment(scene, self.actual[i], self.actual[i+1], actual=True)
        self._live_segments = count
        self._live_end = scene.ngeom
        if self.actual and self.tip is not None:
            self._segment(scene, self.actual[-1], self.tip, actual=True)
        self._marker(scene)

    def draw(self, scene, clear=False):
        if clear:
            scene.ngeom = 0
        reference = list(zip(self.reference[:-1:2], self.reference[1::2]))
        actual = list(zip(self.actual[:-1], self.actual[1:]))
        if self.actual and self.tip is not None:
            actual.append((self.actual[-1], self.tip))
        needed = len(reference) + len(actual) + int(self.tip is not None)
        if scene.ngeom + needed > scene.maxgeom:
            raise RuntimeError("MuJoCo scene has insufficient capacity for trajectory geometry")

        for segments, is_actual in ((reference, False), (actual, True)):
            for start, end in segments:
                self._segment(scene, start, end, actual=is_actual)
        self._marker(scene)


def configure_viewer(viewer, base_world):
    camera = trajectory_camera(base_world)
    with viewer.lock():
        viewer.cam.lookat[:] = camera.lookat
        viewer.cam.distance = camera.distance
        viewer.cam.azimuth = camera.azimuth
        viewer.cam.elevation = camera.elevation
        viewer.opt.geomgroup[5] = 0


def sync_trajectory(viewer, overlay):
    with viewer.lock():
        overlay.draw_live(viewer.user_scn)
    viewer.sync()
