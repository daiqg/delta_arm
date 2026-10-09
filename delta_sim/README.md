# Delta 仿真使用指南

本目录是独立的 Delta 机械臂 MuJoCo 仿真模块。长度、质量、时间、速度、力矩分别使用 m、kg、s、m/s、N·m；关节角在代码和 XML 中使用 rad，文档中的角度会标明单位。完整设计与验证记录见[工作总结](../WORK_SUMMARY.md)。

## 1. 环境与启动

以下步骤面向 Windows 64 位和 PowerShell。先安装 Git、Miniconda，并确认 `git --version`、`conda --version` 可执行。若 PowerShell 不识别 Conda，在 Miniconda Prompt 中执行 `conda init powershell` 后重开 PowerShell。

从尚未克隆项目的目录开始，按顺序执行：

```powershell
git clone https://github.com/daiqg/delta_arm.git
cd delta_arm
cd delta_sim
conda env create -f environment_win.yml
conda activate mujoco-sim-win
```

若已完成[根目录 README](../README.md) 的安装步骤，无需重复执行本节。`environment_win.yml` 必须在当前目录存在；已克隆仓库时直接进入其 `delta_sim/` 子目录，已创建同名环境时跳过创建步骤。

[environment_win.yml](environment_win.yml) 仅包含 Python 3.11、NumPy >=1.24 且 <2、Matplotlib >=3.7、pip 和 MuJoCo 3.6.0。MuJoCo 在环境创建时直接固定版本，便于对比现有接触结果；其余依赖及间接依赖并未全部锁定。

后文所有命令均在已激活的环境和 `delta_arm/delta_sim/` 目录中执行。激活环境本身不要求输出版本信息；第 2 节的模型预览用于查看模型和图形功能。图形窗口和离屏截图需要可用的 OpenGL 驱动/上下文。IDE 应选择 `mujoco-sim-win` 的 Python 解释器。

### 已有环境与后续运行

重新打开终端后，进入仓库的 `delta_sim/` 目录，再执行 `conda activate mujoco-sim-win` 即可运行，无需重新安装。

已有同名环境若缺少依赖，可以在本目录补齐：

```powershell
conda env update -n mujoco-sim-win -f environment_win.yml
conda activate mujoco-sim-win
```

`requirements.txt` 包含相同的三个仿真库及版本约束，供已有 Python 3.11 环境用 `python -m pip install -r requirements.txt` 安装。Windows 的推荐安装流程以上述 YAML 为准。更新已有环境不会主动移除原有的其他包。版本变化可能影响碰撞落点，比较实验时应记录实际依赖版本。

不使用终端激活时，可以在本目录用以下命令替代对应的运行命令：

```powershell
conda run --no-capture-output -n mujoco-sim-win python run_trajectory.py --shape all
conda run --no-capture-output -n mujoco-sim-win python continuous_hitting.py --nballs 10
```

## 2. 模型预览

```powershell
python preview_model.py
python preview_model.py --paddle
```

夹具中心与 `ee_center` 重合，拍心相对其为 (0, 0, -0.170) m，拍柄沿世界 -Z，拍面法向为世界 +X。每个预览命令先保存截图，再打开静态 MuJoCo 窗口；关闭当前窗口后再执行下一条命令。预览不执行轨迹或击球，默认隐藏遮挡机械臂的外部安装架，仅影响显示。

| 预览选项 | 用途 |
| --- | --- |
| `--paddle` | 显示机械臂、球拍和击球场景的近景 |
| `--show-fixtures` | 显示外部安装架 |
| `--headless` | 只保存截图 |
| `--output results/custom.png` | 指定截图路径 |

默认截图：[机械臂](results/model_delta.png)、[机械臂与球拍](results/model_paddle.png)。

模型是小型 CAD Delta：上臂 100 mm、前臂 200 mm、基座肩部半径 74.5 mm。在基座系 Z = -210 mm 处，安全 IK 扫描得到约 74 mm 的内切半径。现有轨迹约为 80～110 mm 级，尺寸来自模型可达范围；更大的轨迹需要匹配实际硬件尺寸，不能只放大显示。

## 3. 轨迹动态可视化

```powershell
python run_trajectory.py --shape all
python run_trajectory.py --shape circle
python run_trajectory.py --shape helix --cycles 2
```

默认打开 MuJoCo 窗口。`all` 在同一窗口依次执行圆、正方形、8 字、五角星和螺旋线，切换形状时清空上一条显示路径。

- 蓝色虚线：完整期望路径。
- 橙色实线：末端已经走过的实际路径。
- 橙色小点：当前末端位置。

轨迹叠加只参与渲染，不产生质量或碰撞。显示路径按距离抽稀；CSV 与误差统计保留全部 1 ms 原始采样。物理积分与控制均为 1 kHz，窗口按约 20 ms 的间隔同步，实际运行速度取决于电脑性能。

### 轨迹尺寸

尺寸以基座坐标系为准；四种平面轨迹的 Z 均为 -210 mm。

| `--shape` | 几何尺寸 | 单周期时间 |
| --- | --- | ---: |
| `circle` | 半径 55 mm，直径 110 mm | 4 s |
| `square` | 边长 80 mm，角点平滑启停 | 5 s |
| `figure8` | 总宽 90 mm、总高 45 mm | 5 s |
| `star` | 外半径 50 mm，外接圆直径 100 mm；内半径 21 mm | 6 s |
| `helix` | 半径 40 mm，Z 从 -230 mm 单调升至 -190 mm | 每圈 4 s |

五角星的旧终端/图表标签写作“外径50mm”，实际实现使用的是**外半径 50 mm**。螺旋线的总 Z 行程始终为 40 mm，增加圈数会减小螺距，不会增加总高度。

### 常用选项

| 选项 | 说明 |
| --- | --- |
| `--shape all` | 执行全部形状；默认值 |
| `--cycles 2` | 每种形状的周期数，至少 1；默认 2 |
| `--auto-close` | 全部完成后自动关闭窗口 |
| `--headless` | 不打开窗口，保存数据、图表和场景截图 |
| `--view` | 显式打开窗口，兼容旧命令；不能与 `--headless` 同用 |
| `--no-ff` | 关闭速度前馈，用于对比实验 |

默认完成后保留最终画面，关闭窗口才退出。提前关闭会停止当前形状及后续形状；已记录的当前形状会输出 `completed=false`，不应当作全程统计。每种形状的文件在该形状结束时保存，不必等所有形状完成。

```powershell
python run_trajectory.py --shape all --cycles 1 --auto-close
python run_trajectory.py --shape all --headless
```

## 4. 在线连续十球

在安装依赖、激活环境并进入 `delta_sim/` 后执行：

```powershell
python continuous_hitting.py --nballs 10
```

默认打开 MuJoCo 窗口：蓝色为最近一次在线预测，橙色为实际球路。每球复位后重新发球；结束后保留窗口，关闭窗口退出。加 `--auto-close` 可自动关闭。

```powershell
python continuous_hitting.py --nballs 10 --headless --no-realtime --output results/my_run
Get-Content results/my_run/online_10_summary.json
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

## 5. 结果文件

所有默认输出在 `results/`：

| 文件 | 内容 |
| --- | --- |
| `model_delta.png` / `model_paddle.png` | 模型近景 |
| `traj_<shape>.png` | XY、XYZ、Z 位置及未滤波误差图 |
| `traj_<shape>_mujoco.png` | 实际 MuJoCo 场景及轨迹叠加截图 |
| `traj_<shape>.csv` | 时间、期望/实际 XYZ、三维误差 |
| `traj_<shape>_metrics.json` | 全程 RMS、最大误差、Z 误差、闭环残差、力矩峰值及完成状态 |
| `online_<n>_balls.csv` | 随机发球、拦截、回球的逐球 SI 数据及合法性结果 |
| `online_<n>_trajectories.csv` | 2 ms 采样的球位置与拍心世界坐标轨迹 |
| `online_<n>_summary.json` | 连续随机击球汇总与预演次数 |
| `online_<n>_final.png` | 最终 MuJoCo 帧，离屏渲染成功时保存 |

轨迹 CSV 的 XYZ 坐标相对基座；击球落点相对球台世界坐标系。CSV 和 JSON 长度单位均为米，图表会标注 mm 或 cm；不能直接混合两种坐标系。

再次运行会覆盖同名文件，未被本次命令生成的旧文件可能仍然保留。建议每轮单独归档 `results/`，同时记录命令、时间、Python/MuJoCo 版本及模型参数。随机击球 JSON 已记录种子和 MuJoCo 版本，但尚未记录完整环境和模型哈希。

## 6. 常见问题

| 现象 | 处理 |
| --- | --- |
| `module 'mujoco' has no attribute 'viewer'` | viewer 是子模块；现有脚本已显式导入。激活按第 1 节创建的环境，运行仓库中的当前脚本；自编脚本应显式使用 `import mujoco.viewer` |
| 看不到窗口 / IDE 使用了其他环境 | 在终端激活 `mujoco-sim-win`，或在 IDE 选择该环境；可使用第 1 节的 `conda run --no-capture-output` |
| `conda env create` 提示环境已存在 | 跳过创建步骤并激活同名环境；缺依赖时按第 1 节更新环境 |
| 找不到脚本或 `environment_win.yml` | 确认当前目录为克隆仓库中的 `delta_sim/` |
| 第一球迟迟未动 | 合法来球及可行拍速筛选会运行多次完整 MuJoCo 试打；查看终端是否报错，必要时先用 `--nballs 1` 验证环境 |
| 全部运行结束但终端未退出 | 最终窗口按设计保留；轨迹和击球均可用 `--auto-close` |
| 击球 JSON 还是上次结果 | 准备过程不更新结果；等待播放完成，检查 JSON 的实际球数及输出目录 |
| 请求 10 球但实际球数不足 | 核对 `simulated_balls` 与 CSV；不能仅凭文件名认为完整运行，也不能仅凭汇总确定中断原因 |
| `--headless` 仍有 OpenGL 错误 | 无窗口运行仍使用离屏渲染；需要可用的 OpenGL 驱动/上下文 |
| 预览模型没有运动 | `preview_model.py` 是静态检查；动态运行使用轨迹或连续击球脚本 |
| 五角星标签尺寸与图形不一致 | 以实际外半径 50 mm、外接圆直径 100 mm 为准 |
| 修改球拍 XML 后被覆盖 | `delta_robot_paddle.xml` 自动生成；修改 `make_paddle_robot.py`，基础机械臂修改 `delta_robot.xml` |

`run_hitting.py` 是连续击球、模型预览及安装校验共用的仿真核心模块；运行击球请使用 `continuous_hitting.py`。旧固定球结果和独立评估入口已移除。实机迁移前仍需测量安装公差、质量惯量、碰撞与空气参数、驱动器动态；离线试算不是在线击球控制器。
