# Delta 实机与 sim2real

当前阶段（2026-10-10）：已接收真实机械臂 STEP 并完成装配及连杆几何提取，继续推进电机资料评审和 ROS 2 规划。实物标定、材料/质量、电路及完整驱动协议仍待确认；本目录尚无可驱动电机的程序，也未进行通电或实机验证。`../delta_sim/` 仍是旧尺寸独立仿真。

建议阅读顺序：

1. [CAD 评审](CAD_REVIEW.md)：真实尺寸、与旧仿真的差异、坐标系、材料及装配疑点。
2. [电机手册解读](MOTOR_MANUAL_NOTES.md)：确认了什么、协议有哪些歧义、还需哪些资料。
3. [实施 pipeline](PIPELINE.md)：从硬件接口冻结到 ROS 2 在线击球的分阶段任务及验收。
4. [待确认参数表](config/hardware_pending.yaml)：记录后续需要实测或由厂家确认的参数。它是资料模板，不是 ROS 2 可加载配置，更不是上电配置。

```text
delta_real/
├── README.md                     # 当前状态、阅读顺序与文件用途
├── MOTOR_MANUAL_NOTES.md          # 九页电机手册的技术摘要、页码和疑点
├── CAD_REVIEW.md                  # 新 STEP 的尺寸、坐标与问题评审
├── PIPELINE.md                   # ROS 2 / ros2_control 架构、实施步骤与验收
├── import_step.py                # CAD 离线工具：装配、几何特征提取和预览
├── derive_geometry.py            # 从解析几何提取肩轴/球心、中心距及基座变换
├── cad/
│   ├── assembly.json             # 装配层级、位姿、几何体积、圆柱/球面特征（mm）
│   ├── parts.csv                 # 按实例编号展开的零件清单
│   ├── geometry.json             # 六杆连接点、臂长与坐标变换（m、rad）
│   ├── assembly_preview.png      # 实际 CAD 外形预览
│   └── linkage_dimensions.png    # 基座坐标系内的连杆中心示意
└── config/
    └── hardware_pending.yaml     # 已知规格与未确认硬件参数；禁止直接用于驱动
```

当前提供文档、参数模板和已运行的 CAD 提取工具；工具复现步骤见 CAD 评审。ROS 2 工作空间、驱动、launch、URDF/Xacro 和实机控制器将在相应阶段建立。

资料来源为用户提供的《DM-J4310-2ECV1.1 减速电机使用说明书》，V1.0，2023-11-16，共 9 页。原始 `达妙电机.pdf` 留在本地 robot 目录，未修改，也未复制上传到仓库。本目录保存自行整理的技术摘要，读者须核对实际电机铭牌、固件及对应版本官方资料。

本轮提取的主动臂约 200 mm、从动杆球心距约 393.988 mm；原仿真为 100/200 mm。源 STEP 中材料统一设为钢，需要另行确认实际材料和质量。下一步优先确认装配偏置、关节限位/零位、球拍夹具，同时补齐完整驱动协议、电路和 CAN 方案，详见 pipeline。
