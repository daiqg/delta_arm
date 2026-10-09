# Delta 机械臂 MuJoCo 仿真

本项目使用 MuJoCo 实现 Delta 机械臂末端轨迹绘制，以及末端刚性安装乒乓球拍后的击球仿真。仓库包含运行所需的代码、场景和 CAD 网格，无需下载其他项目。

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

## 5. 在线连续击球（30 球）

在安装依赖、激活环境并进入 `delta_sim/` 后执行：

```powershell
python continuous_hitting.py --nballs 30
```

默认打开 MuJoCo 窗口：蓝色为最近一次在线预测，橙色为实际球路。每球复位后重新发球；结束后保留窗口，关闭窗口退出。加 `--auto-close` 可自动关闭。

```powershell
python continuous_hitting.py --nballs 30 --headless --no-realtime --output results/my_run
Get-Content results/my_run/online_30_summary.json
python verify_online_control.py
```

发球只按双方落台、过网及工作空间相交筛选，**不会试打或预先筛选回球**。控制器只接收已送达位置观测，初态真值、真值球速和未来球路不进入控制器。发球范围保持旧在线版的小幅随机族，不扩大或缩小范围来调整本轮成功率；该范围并不代表任意随机来球。

| 参数 | 设置 |
| --- | --- |
| 位置采样 | 100 Hz |
| 观测延迟 / 噪声 | 80 ms / 2 mm 标准差高斯噪声 |
| 规划检查间隔 | 60 ms；已有待发布计划时不重复启动 |
| 计算预算 | 100 ms，包括状态估计、预测和全轨迹安全检查；超时丢弃 |
| 计划生效 | 开始计算后的 100 ms；此前继续旧轨迹，因此实际重规划间隔可能超过 60 ms |
| 物理积分 / 伺服 | 2 kHz / 500 Hz |
| 挥拍 | 30 ms 加速、40 ms 击球窗口、30 ms 减速、0.5 s 回位 |
| 拦截策略 | 由实时预测选时间和横向位置；拦截平台高度 -217 mm |
| 挥拍速度模板 | (0.97, 横向修正, 0.20) m/s；这是保守模板，不是每球全局最优拍速搜索 |
| 实测限制 | 拍速 ≤1.2 m/s，关节速度 ≤20 rad/s，物理关节限位余量 ≥2° |

可用 `--seed`、`--measurement-delay-ms`、`--measurement-noise-mm` 和 `--control-period-ms` 修改实验设置。`--no-realtime` 只取消显示等待，不取消计算预算检查。

`online_<n>_summary.json` 汇总合法发球、接触、严格回球、规划超时和运动约束；`online_<n>_balls.csv` 保存逐球指标，`online_<n>_trajectories.csv` 保存球与拍心轨迹，`online_<n>_controller.json` 保存每次规划的观测年龄、耗时、候选状态及发布时间，`online_<n>_final.png` 保存最终场景。逐球估计 RMSE 使用同一采集时刻的真值，仅供评估。

旧预演版保持不变：`python continuous_hitting_rehearsal.py --nballs 10`，输出仍为 `random_10_*`。它的回球经过离线筛选，不能当作在线性能。

### 30 球参考结果

MuJoCo 3.6.0、种子 `20261007`、80 ms 观测延迟、2 mm 测量噪声下，一轮连续 30 球获得 **26/30（86.7%）严格规范回球**；合法且可达发球、球拍接触、运动约束合格均为 30/30，规划超时为 0。第 3、4、6、13 球触网，未记录到对方台面有效落点。峰值拍速 1.1384 m/s，峰值关节速度 9.2106 rad/s。

该指标要求不触网并留有过网余量，比正式规则严格；每球重置后再发球，不是双方连续对拉。单轮结果不代表任意来球成功率，也不保证不同机器负载下结果完全相同。完整记录见 [30 球结果](delta_sim/results/online_30_validation/online_30_summary.json)及同目录逐球 CSV、轨迹、控制日志和截图。它与此前三个种子各 10 球的统计不同，且默认种子的前 10 球重叠，不能视为独立测试相加。

### 文件用途

- `run_trajectory.py`、`trajectory_view.py`：五种末端轨迹与 MuJoCo 可视化。
- `continuous_hitting.py`、`online_control.py`：在线预测、滚动规划及连续击球。
- `run_hitting.py`、`kinematics.py`、XML 和 `meshes/`：共享物理场景、运动学和 CAD 资源，不应删除。
- `continuous_hitting_rehearsal.py`：保留的离线预演对照，在线入口还复用其发球机定位函数。
- `make_paddle_robot.py`、`preview_model.py`、`verify_*.py`：模型生成、查看和必要校验。
- `results/`：参考结果；新增临时实验默认不加入 Git，避免调参输出混入发布版本。

本仓库可独立运行，不依赖旁边的旧项目目录。详细设计和历史问题记录仍保留在工作总结与在线优化记录中。

## 6. 可选：无窗口运行

需要批量输出时，可以用以下命令替代上述可视化运行：

```powershell
python run_trajectory.py --shape all --cycles 1 --headless
python continuous_hitting.py --nballs 10 --headless --no-realtime
```

`--headless` 只关闭交互窗口，离屏截图仍需要 OpenGL。发球筛选和数据字段见详细指南。

- [详细使用指南](delta_sim/README.md)：参数、坐标系、输出字段与排错。
- [工作总结](WORK_SUMMARY.md)：结构设计、验证记录、已知问题与 sim2real 限制。
