# Delta 仿真使用指南

本目录是独立的 Delta 机械臂 MuJoCo 仿真模块。长度、质量、时间、速度、力矩分别使用 m、kg、s、m/s、N·m；关节角在代码和 XML 中使用 rad，文档中的角度会标明单位。完整设计与验证记录见[工作总结](../WORK_SUMMARY.md)。

## 在线连续击球

```powershell
python continuous_hitting.py --nballs 10
```

默认入口执行因果在线控制，结果为 `results/online_10_balls.csv`、`online_10_trajectories.csv`、`online_10_summary.json` 与 `online_10_final.png`。随机发球只检查规范的两次落台、过网和静态工作空间相交；回球不会在发球前预演筛选。控制器以 80 ms 延迟、2 mm 高斯位置噪声的观测更新状态，按 120 ms 周期重规划；完整 Python 计算时间计入截止判断。可通过 `--measurement-delay-ms`、`--measurement-noise-mm` 和 `--control-period-ms` 修改这些量。

`continuous_hitting_rehearsal.py` 是保留的离线预演基准，不能用于报告在线成功率。

建议先按[快速复现流程](../README.md)完成模型检查、轨迹绘制和连续击球，再查阅本指南中的参数和输出说明。

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

后文所有命令均在已激活的环境和 `delta_arm/delta_sim/` 目录中执行。激活环境本身不要求输出版本信息；第 2 节的校验和模型预览用于检查计算与图形功能。图形窗口和离屏截图需要可用的 OpenGL 驱动/上下文。IDE 应选择 `mujoco-sim-win` 的 Python 解释器。

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

## 2. 模型预览与校验

```powershell
python verify_kinematics.py
python verify_paddle_mount.py
python preview_model.py
python preview_model.py --paddle
```

第一条校验应输出 `CAD geometry check: 36 mesh components`，随后给出 FK/IK、闭环与重力保持结果。第二条在四个位姿验证安装变换：夹具中心与 `ee_center` 重合，拍心相对其为 (0, 0, -0.170) m，拍柄沿世界 -Z，拍面法向为世界 +X。每个预览命令先保存截图，再打开静态 MuJoCo 窗口；关闭当前窗口后再执行下一条命令。预览不执行轨迹或击球，默认隐藏遮挡机械臂的外部安装架，仅影响显示。

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

## 4. 连续 10 球可视化

```powershell
python continuous_hitting.py --nballs 10
```

场景包含标准尺寸球台、球网、发球机外观、40 mm 球及末端刚性安装的球拍。击球场景的 Delta 基座位于 (-0.69, 0, 1.25) m；球拍安装尺寸不变。安装高度来自球拍安全工作空间与第二次落台后球高的交集，实际设备须按可达性重新测量。发球口约在 (1.40, 0, 1.02～1.22) m，球速中心族见 `continuous_hitting.py`；位置和速度有种子可复现的小幅随机变化。

每个候选发球先在 MuJoCo 中确认发球方台面接触、无碰网越网、接球方台面接触，且第二次弹跳后有满足关节安全余量的拍心拦截点。对合格来球，脚本逐一预演若干拍速，要求球拍真实接触后再次越网、无碰网并在对方台面发生接触，才把该球放进可视化 10 球序列。CSV 保留发球初态、两次落台、过网高度、拦截位置、拍速、回球落台和试算次数。这是预先验证可行性的演示，**不能用 10/10 的演示率宣称在线感知与实时控制成功率**。

先在终端完成整组球的筛选，再打开同一窗口连续执行 10 球；每球开始前复位物理状态，相当于发球机依次发出 10 个球。蓝色虚线为预测来球路径，橙色实线为实际球路；每球显示回球落台后的弹起，并让机械臂回位。关闭窗口可提前中止。准备耗时取决于候选试算次数；`Preparing...` 期间未开窗属于正常行为。

轨迹采用平滑接近、20 ms 加速、40 ms 匀速击球、20 ms 减速及 0.5 s 平滑回位，避免原 Hermite 接近段因指定终点高速而出现位置过冲。按 2 ms 检查全程安全 IK、球拍最低点离台至少 10 mm、含速度前馈的伺服指令范围、拍速 ≤1.2 m/s 与关节速度 ≤20 rad/s；完整预演另验证实际峰值速度及关节物理限位至少 2° 余量。上述速度上限是当前仿真设计值，实机需按驱动器规格设定。

这里只复现发球机出球后的单打球路规则：先落发球方、越网、再落接球方，接球方一次落台后击球并直接落到对方台面。采用更严格的无碰网和台内余量筛选，不模拟人的手掌、16 cm 抛球及裁判判罚；也不作为双打斜线发球演示。空气密度 1.2 kg/m³、黏度 1.8e-5 Pa·s；球使用 MuJoCo 椭球气动模型，`fluidcoef="0.175 0.0875 0.525 0.35 0.35"`。这些系数及胶皮、台面接触参数是待实测标定的近似值，不应将其视为已完成 sim2real 标定。

| 选项 | 默认值 / 作用 |
| --- | --- |
| `--nballs 10` | 请求球数，建议使用正整数 |
| `--seed 20261007` | 随机种子，控制发球族选择及小幅位置/速度变化 |
| `--headless` | 不打开窗口 |
| `--no-realtime` | 取消演示中的等待，适合批量运行；不保证严格实时 |
| `--auto-close` | 结束后自动关闭 viewer |
| `--output results/my_run` | 自定义输出目录，避免覆盖先前结果 |

运行结束后先写入本轮汇总、CSV 和最终截图，再保留最后一帧；可用 `--auto-close` 自动关窗。提前关窗会停止后续球，保存已经执行的记录，汇总中的实际球数可能小于请求球数。

```powershell
python continuous_hitting.py --nballs 10 --headless --no-realtime --output results/my_run
Get-Content results\my_run\random_10_summary.json
```

汇总应检查：

| 字段 | 含义 |
| --- | --- |
| `requested_balls` | 请求球数 |
| `simulated_balls` | 返回结果的试验数，可能包含关窗中止的试验 |
| `legal_serves` | 发球双方台面接触且无碰网过网的球数 |
| `reachable_serves` | 第二次弹跳后存在安全 IK 拦截点的球数 |
| `paddle_contacts` | 检测到球拍面接触的球数 |
| `legal_returns` | 击球后过网、无碰网且接触对方台面的球数 |
| `motion_within_limits` | 全程规划与实际速度、关节余量均合格的球数 |
| `target_hits_100mm` | 合法回球且落点距 (0.58, 0) m 小于 100 mm 的球数 |
| `preview_attempts` | 为选出该序列所做的完整 MuJoCo 击球试算总数 |

即使指定其他球数，输出文件名仍为 `random_10_*`，应以 JSON 内的实际计数为准。不同 MuJoCo 版本的接触结果可能不同；脚本在找不到可行回球时会明确报错，而不会把漏接算成合法回球。

## 5. 结果文件

所有默认输出在 `results/`：

| 文件 | 内容 |
| --- | --- |
| `model_delta.png` / `model_paddle.png` | 模型近景 |
| `traj_<shape>.png` | XY、XYZ、Z 位置及未滤波误差图 |
| `traj_<shape>_mujoco.png` | 实际 MuJoCo 场景及轨迹叠加截图 |
| `traj_<shape>.csv` | 时间、期望/实际 XYZ、三维误差 |
| `traj_<shape>_metrics.json` | 全程 RMS、最大误差、Z 误差、闭环残差、力矩峰值及完成状态 |
| `random_10_balls.csv` | 随机发球、拦截、回球的逐球 SI 数据及合法性结果 |
| `random_10_trajectories.csv` | 2 ms 采样的球位置与拍心世界坐标轨迹 |
| `random_10_summary.json` | 连续随机击球汇总与预演次数 |
| `random_10_final.png` | 最终 MuJoCo 帧，离屏渲染成功时保存 |

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
