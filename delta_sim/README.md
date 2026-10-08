# Delta 仿真使用指南

本目录是独立的 Delta 机械臂 MuJoCo 仿真模块。长度、质量、时间、速度、力矩分别使用 m、kg、s、m/s、N·m；关节角在代码和 XML 中使用 rad，文档中的角度会标明单位。完整设计与验证记录见[工作总结](../WORK_SUMMARY.md)。

已清理开发期 `debug_*.py` 和独立 `calibrate_*.py` 扫描脚本。日常使用入口为模型预览、运动学校验、轨迹绘制和两种击球脚本；击球所需的运行时碰撞标定仍在 `run_hitting.py` 中自动执行。

## 1. 环境与启动

在仓库根目录使用已有的 Conda 环境：

```powershell
conda activate mujoco-sim-win
cd delta_sim
```

开发机器的该环境解释器为 `D:\anaconda3\envs\mujoco-sim-win\python.exe`，已验证 MuJoCo 3.6.0；其他电脑路径不同，不需要使用这个绝对路径。本机激活环境时会自动检查 MuJoCo/viewer 并显示路径和版本。

没有此环境的 Windows 电脑，在本目录执行：

```powershell
conda env create -f environment_win.yml
conda activate mujoco-sim-win
python -m pip install mujoco==3.6.0
```

[environment_win.yml](environment_win.yml) 从原项目原样迁入，包含 Python 3.11、NumPy <2、SciPy、Matplotlib、quaternion、MKL、MuJoCo、PyTorch CPU 及 pygame。它保留了原环境的较完整依赖，其中部分依赖不是新版仿真的必需项；文件并未锁定所有版本，也不是现有环境的精确导出。创建后固定 MuJoCo 3.6.0，便于对比现有碰撞结果。

已有环境不必重新创建。确需按 YAML 补齐依赖时，可执行 `conda env update -n mujoco-sim-win -f environment_win.yml`，随后重新固定 MuJoCo 版本。本机自动检查由环境内 `etc/conda/activate.d` 中的脚本提供，YAML 不包含这些脚本，其他电脑不会自动获得该提示。

只需补齐新版仿真的最低依赖时：

```powershell
python -m pip install -r requirements.txt
```

要复现开发机器的 MuJoCo 接触版本，再执行 `python -m pip install mujoco==3.6.0`。其余依赖尚未锁定，完整环境复现仍需记录版本。

`requirements.txt` 指定最低依赖版本，并未锁定版本。MuJoCo 版本变化会影响接触与落点；对比实验时应固定并记录实际版本。

不能在当前终端激活 Conda 时，可从本目录执行：

```powershell
conda run --no-capture-output -n mujoco-sim-win python run_trajectory.py --shape all
conda run --no-capture-output -n mujoco-sim-win python continuous_hitting.py --nballs 10
```

后文命令均假设已激活环境且位于本目录。

## 2. 模型预览与校验

```powershell
python preview_model.py
python preview_model.py --paddle
python verify_kinematics.py
```

预览先保存截图，再打开静态 MuJoCo 窗口；它不执行轨迹或击球。默认隐藏遮挡机械臂的外部安装架，仅影响显示。

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

场景包含标准尺寸球台、球网、发球机外观、40 mm 球及末端刚性安装的球拍。默认来球位置为 (1.58, 0, 1.02) m，速度为 (-5, 0, -1) m/s，目标落点为 (0.58, 0) m。来球、球拍和台面的相互作用由 MuJoCo 计算。

流程为：碰撞标定 → 第一球完整预测与拍速搜索 → 挥拍、触球及落点判定 → 下一球。第一球前可能等待数十秒，后续球复用第一球拍速并重新计算拦截时刻。终端逐球打印是否接触、落点与误差。

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
| `contacts` | 检测到球拍面或拍柄接触的试验数 |
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
| `module 'mujoco' has no attribute 'viewer'` | viewer 是子模块；现有脚本已显式导入。本机可重新激活环境查看自动检查结果，并运行当前脚本 |
| 看不到窗口 / IDE 使用了其他环境 | 检查 `sys.executable`，切换到 `mujoco-sim-win`；可使用上面的 `conda run --no-capture-output` |
| 第一球迟迟未动 | 首球先做标定、预测与拍速搜索；查看终端。不能仅凭短暂等待判定卡死 |
| 全部运行结束但终端未退出 | 最终窗口按设计保留；轨迹可用 `--auto-close`，击球需关闭窗口 |
| 击球 JSON 还是上次结果 | 本轮需关闭窗口后才保存；检查 JSON 的实际球数 |
| 请求 10 球但实际球数不足 | 核对 `simulated_balls` 与 CSV；不能仅凭文件名认为完整运行，也不能仅凭汇总确定中断原因 |
| `--headless` 仍有 OpenGL 错误 | 无窗口运行仍使用离屏渲染；需要可用的 OpenGL 驱动/上下文 |
| 预览模型没有运动 | `preview_model.py` 是静态检查；动态运行使用轨迹或连续击球脚本 |
| 五角星标签尺寸与图形不一致 | 以实际外半径 50 mm、外接圆直径 100 mm 为准 |
| 修改球拍 XML 后被覆盖 | `delta_robot_paddle.xml` 自动生成；修改 `make_paddle_robot.py`，基础机械臂修改 `delta_robot.xml` |

实机迁移前尚需处理球拍安装坐标、质量惯量、碰撞与空气参数、驱动器动态等问题，见[工作总结的限制说明](../WORK_SUMMARY.md#6-当前限制与-sim2real-准备)。
