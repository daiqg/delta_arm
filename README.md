# Delta 机械臂 MuJoCo 仿真项目

本项目提供 Delta 机械臂的末端轨迹绘制，以及末端刚性安装乒乓球拍的击球仿真，均使用 MuJoCo。机械臂采用参考 CAD 装配网格。本仓库已用独立新版替换旧项目，运行不依赖 `Project_uav/`；所需 STL 已包含在 `delta_sim/meshes/` 中。

- [详细使用指南](delta_sim/README.md)：环境、命令、可视化、输出与排错。
- [本次工作总结](WORK_SUMMARY.md)：仿真结构、参数、验证结果、问题与解决方案、sim2real 限制。

## 1. 进入运行环境

先克隆仓库：

```powershell
git clone https://github.com/daiqg/delta_arm.git
cd delta_arm
```

在 PowerShell 或 Anaconda Prompt 中激活环境，再进入模块目录：

```powershell
conda activate mujoco-sim-win
cd delta_sim
python -c "import sys, mujoco, mujoco.viewer; print(sys.executable); print(mujoco.__version__)"
```

已有的 `mujoco-sim-win` 环境已验证 MuJoCo 3.6.0。其他电脑可先创建环境：

```powershell
conda create -n mujoco-sim-win python=3.11 -y
conda activate mujoco-sim-win
python -m pip install -r requirements.txt
python -m pip install mujoco==3.6.0
```

已有环境缺少依赖时执行：

```powershell
python -m pip install -r requirements.txt
```

VS Code / PyCharm 也应选择 `mujoco-sim-win` 的 Python 解释器。

## 2. 常用命令

以下命令均在 `delta_sim/` 目录中运行：

| 目的 | 命令 | 行为 |
| --- | --- | --- |
| 查看 CAD 机械臂 | `python preview_model.py` | 保存近景截图并打开静态窗口 |
| 查看机械臂与球拍 | `python preview_model.py --paddle` | 显示末端刚性安装的球拍 |
| 绘制全部五种轨迹 | `python run_trajectory.py --shape all` | 默认打开 MuJoCo 窗口，每种轨迹执行两周期 |
| 绘制两圈螺旋线 | `python run_trajectory.py --shape helix --cycles 2` | 半径 40 mm，总 Z 行程 40 mm |
| 连续演示 10 球 | `python continuous_hitting.py --nballs 10` | 默认打开窗口，第一球需要求解拍速 |
| 检查模型与运动学 | `python verify_kinematics.py` | 检查 CAD 类型、FK/IK、闭环、重力保持及工作空间 |

轨迹窗口中，蓝色虚线是期望路径，橙色实线是已走过的实际路径。轨迹全部完成后默认保留画面，关闭窗口即可退出；添加 `--auto-close` 可自动退出。

击球窗口在最后一球后也保留画面。**关闭击球窗口后才保存本轮 CSV、汇总 JSON 和最终截图**；提前关闭会得到不完整的球数记录。第一球的标定与搜索可能耗时数十秒。

## 3. 无窗口运行

```powershell
python run_trajectory.py --shape all --headless
python continuous_hitting.py --nballs 10 --headless --no-realtime
python run_hitting.py --n 1 --seed 1000
```

`--headless` 表示不打开窗口，截图仍可能需要 OpenGL。结果写入 `delta_sim/results/`，再次运行会覆盖同名文件。击球结果应检查 `simulated_balls`、`contacts`、`landings` 和 `success_100mm`，不能只看文件名。

当前连续演示采用固定来球、逐球复位，是可重复的仿真流程验证。随机来球与实机准确率，以及材料、驱动器和坐标变换的标定要求，见[工作总结](WORK_SUMMARY.md)。
