"""Generate a vertical paddle rigidly mounted at the XML ee_center site.

SI units. Paddle local x points down, y across the blade, z towards world +X.
Masses and materials are estimates pending hardware identification.
"""
from pathlib import Path
import math
import re
import xml.etree.ElementTree as ET

PADDLE_OFFSET = (0.0, 0.0, -0.170)  # ee_center -> blade centre
PADDLE_NORMAL = (1.0, 0.0, 0.0)
PADDLE_HALF_THICK = 0.006
PADDLE_HALF_LENGTH = 0.085
PADDLE_HALF_WIDTH = 0.075
PADDLE_BOTTOM_OFFSET = PADDLE_OFFSET[2] - PADDLE_HALF_LENGTH
MOUNT_MASS = 0.020
PADDLE_MASS = 0.147


def paddle_block(centre):
    pos = ' '.join(f'{v:.9g}' for v in centre)
    return f'''
            <!-- Centred rigid mount: platform underside z=-4mm, plate -4..-8mm. -->
            <body name="paddle_mount" pos="{pos}">
              <site name="paddle_mount_center" pos="0 0 0" size="0.002" rgba="0 0 0 0" />
              <geom name="paddle_mount_plate" type="cylinder" size="0.018 0.002" pos="0 0 -0.006"
                    mass="0.008" rgba="0.40 0.42 0.45 1" contype="0" conaffinity="0" />
              <!-- Symmetric jaws grip the 10mm-thick vertical handle. -->
              <geom name="paddle_clamp_front" type="box" size="0.003 0.016 0.017" pos="0.008 0 -0.025"
                    mass="0.005" rgba="0.32 0.34 0.38 1" contype="0" conaffinity="0" />
              <geom name="paddle_clamp_back" type="box" size="0.003 0.016 0.017" pos="-0.008 0 -0.025"
                    mass="0.005" rgba="0.32 0.34 0.38 1" contype="0" conaffinity="0" />
              <geom name="paddle_clamp_bolt1" type="cylinder" size="0.002 0.013" pos="0 -0.011 -0.022"
                    euler="0 {math.pi/2} 0" mass="0.001" rgba="0.65 0.66 0.68 1" contype="0" conaffinity="0" />
              <geom name="paddle_clamp_bolt2" type="cylinder" size="0.002 0.013" pos="0 0.011 -0.033"
                    euler="0 {math.pi/2} 0" mass="0.001" rgba="0.65 0.66 0.68 1" contype="0" conaffinity="0" />
              <body name="paddle" pos="0 0 {PADDLE_OFFSET[2]}" euler="0 {math.pi/2} 0">
                <geom name="paddle_face" type="ellipsoid" size="0.085 0.075 0.006" mass="0.075"
                      rgba="0 0 0 0" contype="1" conaffinity="1" condim="3"
                      friction="1.8 0.02 0.005" solref="0.0015 0.07" margin="0.001" />
                <geom name="paddle_blade_vis" type="ellipsoid" size="0.085 0.075 0.003" mass="0"
                      rgba="0.62 0.42 0.20 1" contype="0" conaffinity="0" />
                <geom name="paddle_rubber_red" type="ellipsoid" size="0.083 0.073 0.0015" pos="0 0 0.0045"
                      mass="0.016" rgba="0.80 0.09 0.09 1" contype="0" conaffinity="0" />
                <geom name="paddle_rubber_black" type="ellipsoid" size="0.083 0.073 0.0015" pos="0 0 -0.0045"
                      mass="0.016" rgba="0.10 0.10 0.12 1" contype="0" conaffinity="0" />
                <!-- Handle z=-10..-100mm relative to platform; 15mm overlaps blade neck. -->
                <geom name="paddle_handle" type="box" size="0.045 0.014 0.005" pos="-0.115 0 0"
                      mass="0.040" rgba="0.55 0.38 0.18 1" contype="1" conaffinity="1" condim="3"
                      friction="0.9 0.02 0.005" solref="0.002 0.1" />
                <site name="paddle_handle_top" pos="-0.160 0 0" size="0.002" rgba="0 0 0 0" />
                <site name="paddle_handle_bottom" pos="-0.070 0 0" size="0.002" rgba="0 0 0 0" />
                <site name="paddle_center" pos="0 0 0" size="0.002" rgba="0 0 0 0" />
              </body>
            </body>'''


def ensure_paddle_robot(base_dir=None):
    root = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    xml = (root / 'delta_robot.xml').read_text(encoding='utf-8')
    tree = ET.fromstring(xml)
    platform = tree.find(".//body[@name='end_platform']")
    if platform is None or tree.find(".//body[@name='paddle']") is not None:
        raise ValueError('Expected a base Delta model with one end_platform and no paddle')
    ee = platform.find("site[@name='ee_center']")
    if ee is None:
        raise ValueError('Missing platform centre reference')
    centre = tuple(float(x) for x in ee.attrib['pos'].split())
    anchor = re.search(r'<site\s+name="plat_joint_1b"[^/]*/>', xml)
    if anchor is None:
        raise ValueError('Missing platform insertion anchor')
    out = xml[:anchor.end()] + paddle_block(centre) + xml[anchor.end():]
    out = out.replace('model="delta_robot"', 'model="delta_robot_paddle"', 1)
    dst = root / 'delta_robot_paddle.xml'
    if not dst.exists() or dst.read_text(encoding='utf-8') != out:
        dst.write_text(out, encoding='utf-8')
    return str(dst)


if __name__ == '__main__':
    print(ensure_paddle_robot())
