# Delta 机械臂 MuJoCo 仿真

本项目使用 MuJoCo 实现 Delta 机械臂末端轨迹绘制，以及末端刚性安装乒乓球拍后的击球仿真。仓库包含运行所需的代码、场景和 CAD 网格，无需下载其他项目。

## 当前在线击球入口

```powershell
cd delta_arm\delta_sim
conda activate mujoco-sim-win
python continuous_hitting.py --nballs 10
```

`continuous_hitting.py` 是默认的因果在线方案。发球只经过乒乓规则与静态工作空间检查；发球后控制器只接收延迟位置观测和关节反馈，不读取真实球速度、未来轨迹，也不在主机器人模型中预演回球。默认设置为 80 ms 观测延迟、2 mm 位置噪声和 120 ms 重规划周期，结果写入 `results/online_10_*`。蓝色虚线为当前在线预测，橙色实线为实际球路。

`continuous_hitting_rehearsal.py` 保留旧的离线预演筛选实现，仅作接触模型回归基准；它的成功率不能代表在线控制性能。

下面按顺序完成环境安装、模型检查、轨迹运行和连续击球。命令面向 **Windows 64 位 + PowerShell**。

## 1. 准备工具

安装 [Git](https://git-scm.com/downloads/win) 和 [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)。打开支持 Conda 的 PowerShell 终端，确认以下命令可用：

```powershell
git --version
conda --version
```

如果普通 PowerShell 不识别 `conda`，在 Miniconda Prompt 中执行 `conda init powershell`，关闭并重新打开 PowerShell 后再继续。图形窗口和离屏截图均需要可用的 OpenGL 驱动及图形环境。

## 2. 克隆仓库并创建环境

在希望保存项目的目录打开终端，按顺序执行：

```powershell
git clone https://github.com/daiqg/delta_arm.git
cd delta_arm
cd delta_sim
conda env create -f environment_win.yml
conda activate mujoco-sim-win
```

创建成功后会得到名为 `mujoco-sim-win` 的环境，使用 Python 3.11；环境文件直接安装固定版本的 MuJoCo 3.6.0，便于对比验证结果。

[environment_win.yml](delta_sim/environment_win.yml) 仅包含仿真需要的 NumPy、Matplotlib、MuJoCo 和安装工具。配置没有锁定全部间接依赖，不能保证不同时间安装的环境逐包相同。

**后续命令均在同一个终端、`delta_arm/delta_sim/` 目录中执行。** 每一步成功后再继续。如果环境已存在，跳过创建步骤，激活后确认依赖可用；不要重复创建同名环境。IDE 运行时也应选择该 Conda 环境的解释器。

## 3. 检查模型

```powershell
python verify_kinematics.py
python verify_paddle_mount.py
```

第一条命令会输出 `CAD geometry check: 36 mesh components`，以及 FK/IK 误差、闭环残差、3 秒重力保持和工作空间扫描结果。第二条命令在四个位姿核对球拍与末端中心的安装关系：拍柄沿世界 -Z，拍面竖直、法向世界 +X，拍心位于末端中心正下方 170 mm。

查看带球拍的模型：

```powershell
python preview_model.py --paddle
```

程序保存 `results/model_paddle.png` 并打开静态模型窗口。**关闭窗口后再执行下一步。** 仅查看机械臂可改用 `python preview_model.py`。

## 4. 运行五种轨迹

```powershell
python run_trajectory.py --shape all --cycles 1 --auto-close
```

同一窗口依次显示圆、正方形、8 字、五角星和螺旋线，每种执行一个周期。蓝色虚线是期望路径，橙色实线是实际路径；全部完成后自动关窗。

查看螺旋线统计：

```powershell
Get-Content results/traj_helix_metrics.json
```

确认 `completed` 为 `true`。长度误差单位为米，乘以 1000 才是毫米；`results/traj_helix.png` 是误差图，`results/traj_helix_mujoco.png` 是场景截图。其余形状使用相同命名规则。

轨迹主要 XY 尺寸约为 80～110 mm；螺旋线半径 40 mm，总 Z 行程 40 mm。调整形状、周期数及显示方式见[详细指南](delta_sim/README.md)。

## 5. 运行连续 10 球

```powershell
python continuous_hitting.py --nballs 10
```

程序随机生成候选来球，先在 MuJoCo 中筛出符合双方落台、过网且可被机械臂安全拦截的发球，再试算若干拍速，选择能真实过网并落到发球方台面的回球。完整准备 10 球后才打开窗口，以正常时间节奏连续执行；终端的 `Preparing...` 表示正在筛选，请等待。蓝色虚线为预测来球路径，橙色为实际球路，包含两次发球落台、击球、回球落台弹起及机械臂回位。

第 10 球结束后立即保存汇总、CSV 和截图，并保留窗口供查看；关闭窗口退出。也可使用 `python continuous_hitting.py --nballs 10 --auto-close` 自动结束。提前关窗会保存不完整的球数记录。随后查看：

```powershell
Get-Content results/random_10_summary.json
```

检查 `requested_balls` 和 `simulated_balls` 是否均为 10，再查看 `legal_serves`、`reachable_serves`、`paddle_contacts`、`legal_returns` 和 `motion_within_limits`。`target_hits_100mm` 另表示落点距目标小于 100 mm 的球数。逐球初态、两次台面接触、球网通过高度、实际峰值速度和回球落点在 `results/random_10_balls.csv`；2 ms 采样的球及拍心轨迹在 `results/random_10_trajectories.csv`。

使用 MuJoCo 3.6.0、默认种子的一轮验证中，规范发球、可达、拍面接触、合法回球和全程运动约束均为 10/10，筛选做了 85 次完整试打。实际最大拍速 1.191 m/s，最大关节速度 11.145 rad/s，最小关节限位余量 2.72°；GUI 与无窗口结果一致。回球落点 X 为 0.189～0.370 m，均在对方台面内，但距 (0.58, 0) m 的目标 100 mm 内为 0/10，尚未优化定点落球。每球复位仿真状态，且回球在显示前经过离线物理试算，因此此演示不能代表在线控制或实机成功率。这里只模拟发球机出球后的单打球路，不包含人的手掌与抛球动作。运行会覆盖同名随机结果文件；可用 `--output results/my_run` 指定独立目录。

## 6. 可选：无窗口运行

需要批量输出时，可以用以下命令替代上述可视化运行：

```powershell
python run_trajectory.py --shape all --cycles 1 --headless
python continuous_hitting.py --nballs 10 --headless --no-realtime
```

`--headless` 只关闭交互窗口，离屏截图仍需要 OpenGL。发球筛选和数据字段见详细指南。

- [详细使用指南](delta_sim/README.md)：参数、坐标系、输出字段与排错。
- [工作总结](WORK_SUMMARY.md)：结构设计、验证记录、已知问题与 sim2real 限制。
