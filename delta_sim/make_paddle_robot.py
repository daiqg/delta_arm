"""
make_paddle_robot.py — 从 delta_robot.xml 自动生成 delta_robot_paddle.xml
==========================================================================
在动平台 (end_platform) 上刚性固定一只真实尺寸的乒乓球拍 (无关节 = 同一刚体),
生成文件供 scene_pingpong.xml <include>。保持与 delta_robot.xml 单一来源同步。

球拍参数 (与真实一致):
  - 拍板: 椭圆 170 x 150 mm, 木材+胶皮总厚 12 mm
  - 胶皮: 正手红 / 反手黑, 各 ~1.5mm, 覆盖拍面
  - 拍柄: 90 x 28 x 10 mm 直柄 (竖向朝上连接平台), 经夹具与平台刚性锁紧
  - 总质量 ~167 g (拍板 75g + 胶皮 2x16g + 柄 40g + 夹具 20g)
  - 安装姿态: 拍面法线 n = (cos30°, 0, sin30°) — 近乎垂直、前倾 30° (真实击球姿态,
    来球近水平时法向分量最大, 反弹可控); 拍长轴沿 (sin30°, 0, -cos30°) 斜向下前方;
    拍面中心位于平台中心前下方 (0.045, 0, -0.095) [平台系 = 基座系, 平台始终平行基座]
  - 拍面摩擦 (胶皮-球): mu ~ 1.8 (粘性反胶)
"""
import os
import re

PADDLE_BLOCK = """
      <!-- ===== 乒乓球拍: 刚性固定于动平台 (无关节, 同一刚体) ===== -->
      <!-- euler-y = 60° (自垂直起), 拍面法线 = (sin60, 0, cos60) = (cos30, 0, sin30) -->
      <body name="paddle" pos="0.045 0 -0.095" euler="0 1.0471975511965976 0">
        <!-- body 系: z = 拍面法线 (前倾30°), x = 拍长轴 (斜向下前方), y = 拍宽 -->
        <!-- 碰撞体 = 完整拍面含胶皮 (厚12mm, 摩擦=反胶) -->
        <geom name="paddle_face" type="ellipsoid" size="0.0835 0.0735 0.006" mass="0.075"
              rgba="0 0 0 0" contype="1" conaffinity="1" condim="3"
              friction="1.8 0.02 0.005" solref="0.0015 0.07" solimp="0.95 0.99 0.0001 0.5 2"
              margin="0.001" />
        <!-- 视觉: 木拍板 -->
        <geom name="paddle_blade_vis" type="ellipsoid" size="0.085 0.075 0.0035" mass="0.0"
              rgba="0.62 0.42 0.20 1" contype="0" conaffinity="0" />
        <!-- 视觉: 正手红胶皮 / 反手黑胶皮 -->
        <geom name="paddle_rubber_red" type="ellipsoid" size="0.082 0.072 0.0015" pos="0 0 0.0045"
              mass="0.016" rgba="0.80 0.09 0.09 1" contype="0" conaffinity="0" />
        <geom name="paddle_rubber_black" type="ellipsoid" size="0.082 0.072 0.0015" pos="0 0 -0.0045"
              mass="0.016" rgba="0.10 0.10 0.12 1" contype="0" conaffinity="0" />
        <!-- 拍柄: 90x28x10mm, 沿 -x (上后方) 伸向平台 -->
        <geom name="paddle_handle" type="box" size="0.045 0.014 0.005" pos="-0.0875 0 0" mass="0.040"
              rgba="0.55 0.38 0.18 1" contype="1" conaffinity="1" condim="3"
              friction="0.9 0.02 0.005" solref="0.002 0.1" />
        <!-- 柄-平台锁紧夹具 -->
        <geom name="paddle_clamp" type="box" size="0.020 0.024 0.016" pos="-0.055 0 0" mass="0.020"
              rgba="0.30 0.31 0.35 1" contype="0" conaffinity="0" />
        <site name="paddle_center" pos="0 0 0" size="0.004" rgba="1 0.5 0 1" />
      </body>
"""

# 拍面中心相对平台中心的偏移 (平台系 = 基座系, 米)
PADDLE_OFFSET = (0.045, 0.0, -0.095)
# 拍面法线 (基座系): 前倾 30°
PADDLE_NORMAL = (0.8660254037844387, 0.0, 0.5000000000000001)
# 拍面碰撞半厚 (含胶皮)
PADDLE_HALF_THICK = 0.006


def ensure_paddle_robot(base_dir=None):
    """生成/更新 delta_robot_paddle.xml (源 delta_robot.xml 变更时自动重建)"""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(base_dir, "delta_robot.xml")
    dst = os.path.join(base_dir, "delta_robot_paddle.xml")
    with open(src, "r", encoding="utf-8") as f:
        xml = f.read()

    if '<body name="paddle"' in xml:
        raise RuntimeError("delta_robot.xml 不应包含 paddle, 请检查源文件")

    # 在 end_platform body 的最后一个 site (plat_joint_1b) 之后插入球拍
    anchor = '<site name="plat_joint_1b"  pos="-0.0148 -0.0085 0" size="0.003" rgba="1 1 0.2 1" />'
    if anchor not in xml:
        # 宽松匹配
        m = re.search(r'<site\s+name="plat_joint_1b"[^/]*/>', xml)
        if not m:
            raise RuntimeError("未找到 end_platform 插入锚点 (plat_joint_1b site)")
        anchor = m.group(0)
    out = xml.replace(anchor, anchor + "\n" + PADDLE_BLOCK, 1)

    # 模型名
    out = out.replace('<mujoco model="delta_robot">', '<mujoco model="delta_robot_paddle">', 1)

    need = False
    if not os.path.exists(dst):
        need = True
    else:
        with open(dst, "r", encoding="utf-8") as f:
            old = f.read()
        if old != out:
            need = True
    if need:
        with open(dst, "w", encoding="utf-8") as f:
            f.write(out)
        print("[make_paddle_robot] 已生成 delta_robot_paddle.xml")
    return dst


if __name__ == "__main__":
    print(ensure_paddle_robot())
