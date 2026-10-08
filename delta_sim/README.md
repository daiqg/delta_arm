# Delta 仿真使用指南

本目录是独立的 Delta 机械臂 MuJoCo 仿真模块。长度、质量、时间、速度、力矩分别使用 m、kg、s、m/s、N·m；关节角在代码和 XML 中使用 rad，文档中的角度会标明单位。完整设计与验证记录见[工作总结](../WORK_SUMMARY.md)。

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
python -m pip install mujoco==3.6.0
```

若已完成[根目录 README](../README.md) 的安装步骤，无需重复执行本节。`environment_win.yml` 必须在当前目录存在；已克隆仓库时直接进入其 `delta_sim/` 子目录，已创建同名环境时跳过创建步骤。

[environment_win.yml](environment_win.yml) 包含 Python 3.11、NumPy <2、SciPy、Matplotlib、quaternion、MKL、MuJoCo、PyTorch CPU 及 pygame。其中部分依赖用于兼容原有环境，并非新版仿真必需。该配置不是全部版本锁定的环境快照，因此安装后单独固定 MuJoCo 3.6.0，便于对比现有接触结果。

后文所有命令均在已激活的环境和 `delta_arm/delta_sim/` 目录中执行。激活环境本身不要求输出版本信息；第 2 节的校验和模型预览用于检查计算与图形功能。图形窗口和离屏截图需要可用的 OpenGL 驱动/上下文。IDE 应选择 `mujoco-sim-win` 的 Python 解释器。

### 已有环境与后续运行

重新打开终端后，进入仓库的 `delta_sim/` 目录，再执行 `conda activate mujoco-sim-win` 即可运行，无需重新安装。

已有同名环境若缺少依赖，可以在本目录补齐：

```powershell
conda env update -n mujoco-sim-win -f environment_win.yml
conda activate mujoco-sim-win
python -m pip install mujoco==3.6.0
```

`requirements.txt` 仅列出新版仿真的最低 Python 依赖；完整 Windows 安装流程以上述 YAML 为准。版本变化可能影响碰撞落点，比较实验时应记录实际依赖版本。

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

场景包含标准尺寸球台、球网、发球机外观、40 mm 球及末端刚性安装的球拍。默认来球位置为 (1.40, 0, 1.02) m，速度为 (-6.5, 0, -1) m/s，目标落点为 (0.58, 0) m。来球、球拍和台面的相互作用由 MuJoCo 计算。

流程为：每球预测拦截时刻 → 使用固定拍速 (1.10, 0, 0.45) m/s 挥拍 → 触球和落点判定 → 下一球。该拍速仅针对当前固定来球、竖直球拍和 MuJoCo 3.6.0 验证；终端逐球打印是否接触、落点与误差。

**当前连续演示是同一窗口内的逐球实验，每球都会复位整个仿真状态，且关闭来球随机抖动。** 它不是保持多个球同时存在的连续物理世界，不能据此评估真实发球机的固定节拍或随机来球成功率。

| 选项 | 默认值 / 作用 |
| --- | --- |
| `--nballs 10` | 请求球数，建议使用正整数 |
| `--seed 20261007` | 随机种子；默认演示关闭来球抖动 |
| `--headless` | 不打开窗口 |
| `--no-realtime` | 取消演示中的等待，适合批量运行；不保证严格实时 |

运行结束后窗口会保留最后一帧。**正常关闭窗口后，才写入本轮汇总、CSV 和最终截图。** 该脚本没有 `--auto-close` 参数。提前关窗会停止后续球，汇总中的实际球数可能小于请求球数。

```powershell
python continuous_hitting.py --nballs 10 --headless --no-realtime
Get-Content results\continuous_10_summary.json
```

汇总应检查：

| 字段 | 含义 |
| --- | --- |
| `requested_balls` | 请求球数 |
| `simulated_balls` | 返回结果的试验数，可能包含关窗中止的试验 |
| `contacts` | 检测到球拍面接触的试验数 |
| `landings` | 击球后满足对方台面落点判定的试验数 |
| `success_100mm` | 接触后满足过网检查、落台且目标误差小于 100 mm 的试验数 |
| `mean_landing_error_m` / `max_landing_error_m` | 有落点试验的平均 / 最大误差，不含无落点试验 |

即使指定其他球数，当前输出文件名仍为 `continuous_10_*`，应以 JSON 内的实际计数为准。

## 5. 随机来球离线评估

```powershell
python run_hitting.py --n 1 --seed 1000
python run_hitting.py --n 20 --seed 1000
```

此脚本不打开 viewer，默认 20 次试验；每球加入来球抖动并选择目标落点，单独搜索拍速。它通常比固定来球连续演示耗时更长。输出终端统计、落点散点图、误差图和最多三个示例图。当前没有充分验证的大样本随机成功率，不能把固定 10 球结果替代随机评估。

## 6. 结果文件

所有默认输出在 `results/`：

| 文件 | 内容 |
| --- | --- |
| `model_delta.png` / `model_paddle.png` | 模型近景 |
| `traj_<shape>.png` | XY、XYZ、Z 位置及未滤波误差图 |
| `traj_<shape>_mujoco.png` | 实际 MuJoCo 场景及轨迹叠加截图 |
| `traj_<shape>.csv` | 时间、期望/实际 XYZ、三维误差 |
| `traj_<shape>_metrics.json` | 全程 RMS、最大误差、Z 误差、闭环残差、力矩峰值及完成状态 |
| `continuous_10_landing.csv` | 球序号、触球时间、落点、误差、成功标记 |
| `continuous_10_summary.json` | 连续击球计数及落点误差汇总 |
| `continuous_final.png` | 连续击球最终 MuJoCo 帧，渲染成功时保存 |
| `hit_landing_scatter.png` / `hit_error_series.png` | 随机评估落点与误差图 |
| `hit_example_<trial>.png` | 部分随机试验轨迹示例 |

轨迹 CSV 的 XYZ 坐标相对基座；击球落点相对球台世界坐标系。CSV 和 JSON 长度单位均为米，图表会标注 mm 或 cm；不能直接混合两种坐标系。

再次运行会覆盖同名文件，未被本次命令生成的旧文件可能仍然保留。建议每轮单独归档 `results/`，同时记录命令、时间、Python/MuJoCo 版本及模型参数；现有结果 JSON 未自动记录这些环境信息。

## 7. 常见问题

| 现象 | 处理 |
| --- | --- |
| `module 'mujoco' has no attribute 'viewer'` | viewer 是子模块；现有脚本已显式导入。激活按第 1 节创建的环境，运行仓库中的当前脚本；自编脚本应显式使用 `import mujoco.viewer` |
| 看不到窗口 / IDE 使用了其他环境 | 在终端激活 `mujoco-sim-win`，或在 IDE 选择该环境；可使用第 1 节的 `conda run --no-capture-output` |
| `conda env create` 提示环境已存在 | 跳过创建步骤并激活同名环境；缺依赖时按第 1 节更新环境 |
| 找不到脚本或 `environment_win.yml` | 确认当前目录为克隆仓库中的 `delta_sim/` |
| 第一球迟迟未动 | 检查终端异常和 MuJoCo 窗口；固定来球演示不执行首球拍速搜索 |
| 全部运行结束但终端未退出 | 最终窗口按设计保留；轨迹可用 `--auto-close`，击球需关闭窗口 |
| 击球 JSON 还是上次结果 | 本轮需关闭窗口后才保存；检查 JSON 的实际球数 |
| 请求 10 球但实际球数不足 | 核对 `simulated_balls` 与 CSV；不能仅凭文件名认为完整运行，也不能仅凭汇总确定中断原因 |
| `--headless` 仍有 OpenGL 错误 | 无窗口运行仍使用离屏渲染；需要可用的 OpenGL 驱动/上下文 |
| 预览模型没有运动 | `preview_model.py` 是静态检查；动态运行使用轨迹或连续击球脚本 |
| 五角星标签尺寸与图形不一致 | 以实际外半径 50 mm、外接圆直径 100 mm 为准 |
| 修改球拍 XML 后被覆盖 | `delta_robot_paddle.xml` 自动生成；修改 `make_paddle_robot.py`，基础机械臂修改 `delta_robot.xml` |

当前竖直拍面模型的固定来球 10 球验证为拍面接触 10/10、落台 10/10、目标成功 10/10，落点误差约 38.5 mm。来球到达拍面时仍处于上升段；此结果不能推广到下落来球、随机来球或实机。实机迁移前仍需测量安装公差、质量惯量、碰撞与空气参数、驱动器动态，见[工作总结的限制说明](../WORK_SUMMARY.md#6-当前限制与-sim2real-准备)。
