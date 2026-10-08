# Delta 机械臂 MuJoCo 仿真

本项目使用 MuJoCo 实现 Delta 机械臂末端轨迹绘制，以及末端刚性安装乒乓球拍后的击球仿真。仓库包含运行所需的代码、场景和 CAD 网格，无需下载其他项目。

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
python -m pip install mujoco==3.6.0
```

创建成功后会得到名为 `mujoco-sim-win` 的环境，使用 Python 3.11。最后一条命令固定 MuJoCo 3.6.0，便于对比验证结果。

[environment_win.yml](delta_sim/environment_win.yml) 包含 NumPy <2、Matplotlib、SciPy、MuJoCo、PyTorch CPU 等依赖；部分依赖用于兼容原有环境，并非这两个仿真的必需项。配置没有锁定全部依赖，不能保证不同时间安装的环境逐包相同。

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

程序使用已验证的固定拍速，在 MuJoCo 窗口中逐球演示发球、挥拍与落点，终端输出每球结果。

**等待终端输出第 10 球后，正常关闭窗口，程序才保存本轮汇总、CSV 和最终截图。** 提前关窗会得到不完整的球数记录。随后查看：

```powershell
Get-Content results/continuous_10_summary.json
```

检查 `requested_balls` 和 `simulated_balls` 是否均为 10，再查看 `contacts`、`landings`、`success_100mm`。成功判据包含触球、过网检查、对方台面落点和目标误差小于 100 mm；不能只凭文件名判断完成情况。

当前竖直拍面模型的固定来球 10 球验证结果为拍面接触 10/10、对方落台 10/10、目标成功 10/10，落点误差约 38.5 mm。演示每球复位仿真状态，且来球与拍速相同，不代表随机来球或实机成功率。运行会覆盖同名结果文件，以本次生成的数据为准。

## 6. 可选：无窗口运行

需要批量输出时，可以用以下命令替代上述可视化运行：

```powershell
python run_trajectory.py --shape all --cycles 1 --headless
python continuous_hitting.py --nballs 10 --headless --no-realtime
```

`--headless` 只关闭交互窗口，离屏截图仍需要 OpenGL。随机来球离线评估使用 `python run_hitting.py --n 1 --seed 1000`，其参数和输出见详细指南。

- [详细使用指南](delta_sim/README.md)：参数、坐标系、输出字段与排错。
- [工作总结](WORK_SUMMARY.md)：结构设计、验证记录、已知问题与 sim2real 限制。
