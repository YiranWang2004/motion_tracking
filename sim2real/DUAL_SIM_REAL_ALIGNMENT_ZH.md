# 双 G1：sim2sim 与 sim2real 流程对齐

> 最新默认动作已改为 `motion_000029`，已通过两次完整 UDP 闭环；此前 `motion_000000` 的失败记录保留为历史。详见 [默认 motion 筛选报告](DUAL_DEFAULT_MOTION_SELECTION_ZH.md)。

更新：2026-09-05。适用于 `deploy_dual_scalebfm_residual.py` 和两个
`dual_scalebfm_*.yaml`。单机器人 OmniContact 入口不在这次重构范围内。

## 检查结论

整合前只是共用 ScaleBFM + residual 推理组件，完整运行流程并未对齐：
仿真等待 s/b/a，实机自动进入 reference；两端目标变化限幅不同；
实机 bridge 转发实际 PD 增益，而仿真对 enable=0 强制使用固定阻尼。
因此，不能把仿真通过简单理解为实机全流程通过。

现在两端入口都实例化 `dual_runtime.interactive_control.InteractiveDualCoordinator`。
`sim_control.py` 只保留兼容导入。旧 `DualPolicyCoordinator` 留作基础实现及兼容测试，
正式部署入口不再选择其自动启动流程。

| 项目 | 整合后的共同行为 |
| --- | --- |
| 启动 | ZERO TORQUE，等待新的 Start/s 按键沿；不因启动时按键已按住而使能 |
| DefaultPose | 从各自测量关节角插值，50 Hz 下用 100 tick 完成两秒过渡，再持续保持 |
| 站立 | DefaultPose ready 后重新按 B/b，独立初始化两份 LocoMode 循环状态 |
| 搬运 | A/a 时执行三物体几何预检，以当时机器人 A 位姿对齐 reference，再运行耦合策略 |
| 完成 | reference 完成后回到独立 LocoMode；本进程不重复播放任务 |
| 停止 | 任一侧 Stop/x 对两侧发送 kp=0、kd=8 的阻尼命令并退出 |
| 目标限制 | 关节范围限制及各阶段 max_target_delta 共用；残差阶段保留估计力矩限制 |
| 状态故障 | 缺失、过期、双桥状态偏斜超限均终止，尝试向两侧发送阻尼 |
| 运行保护 | 仅 LocoMode 站立阶段使用绝对倾角限制；任务阶段检查箱子相对已执行 reference 的高度、位置误差 |
| 实时检查 | 两端检查控制处理耗时，包含 LocoMode；扣除等待输入的时间；连续慢 tick 才终止 |
| 记录/回放 | 共用状态、原因、实际目标及 PD 增益、enable、按钮、输入时间戳、计算耗时、有效配置和 A 时刻对齐位姿 |

`ZERO TORQUE` 显式发送 kp=0、kd=0。`enable=0` 的非零位置增益命令在
`RobotSession` 中转换为阻尼，避免 bridge 转发后继续位置控制。
仿真禁用状态按命令的 kd 执行，不再自行替换为另一个固定值。
这次修正在双机 Python 发送端完成，不改变 C++ bridge 对其他调用方的协议。

`--act-robot none` **仍发送 UDP 命令**：等待阶段零力矩，进入有目标的阶段则发阻尼。
它不是被动监听，也不保证无电机力矩。纯离线策略评估应使用离线评估工具；
只观察实物应使用不创建命令 socket 的 viewer。A-only/B-only 的另一侧同样使用禁用语义。

## 共用参数

两个配置文件的 `control` 段决定控制行为，不能再通过 `simulation.robots` 私自调整限幅。
机器人后端配置只选择 UDP 端点。

```yaml
default_pose_duration_s: 2.0
control:
  standing_asset_dir: omnicontact
  damping_kd: 8.0
  max_tilt_rad: 0.7
  require_button_release: true
  phase_target_delta:
    default_pose: 0.02
    loco_standing: 1.0
    executing: 1.0
task_safety:
  object_position_z_error_m: 0.1
  object_position_xyz_error_m: 0.3
```

目标变化值单位是 rad/控制 tick，不是 rad/s。任务阶段恢复原仿真值 1.0；
上次整合误改为 0.005，破坏了动态目标跟踪，不能因为数值更小就认为更安全。
这是共用配置，两端任务限幅都会变为 1.0；本次只在仿真中验证，未运行实机。
DefaultPose/LocoMode 沿用单 G1 目标路径，不做基于 50 Hz 状态估计的逆力矩目标重写；
仿真每个物理子步仍裁剪实际 PD 力矩，实机实际执行受硬件控制环约束。
两端软件路径相同并不意味着电机侧实际力矩完全相同。

仿真不再独立执行另一套任务误差退出分支；部署控制器负责两端共同的任务检查。
仿真内部的 `task_termination_reason()` 保留用于诊断和测试。

## 当前启动命令

### sim2sim：终端 1

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
bash scripts/run_dual_scalebfm_simulator.sh \
  --config config/g1/dual_scalebfm_contact_v2_8192.yaml
```

无窗口时在该命令后添加 `--headless`，按键在终端 1 输入，每键后回车。

### sim2sim：终端 2

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
bash scripts/run_dual_scalebfm_residual.sh \
  --config config/g1/dual_scalebfm_contact_v2_8192.yaml \
  --pose-source sim \
  --act-robot both \
  --confirm-actuation ENABLE_MOTORS \
  --no-visualization
```

顺序：等待 ZERO TORQUE → s → 等待 DefaultPose ready → b → 确认站立稳定 → a。
reference 结束自动回站立，x 退出。提前按 b 不会排队等待进入站立；ready 后需要重新按。
`--duration` 包含等待按键时间，人工调试建议省略。

一键 headless 启动同样支持以上流程：

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
bash scripts/run_dual_scalebfm_sim2sim.sh \
  --config config/g1/dual_scalebfm_contact_v2_8192.yaml \
  --headless
```

### sim2real：桥接与持续 Vive 输入

网络配置、网口隔离和 Tracker 标定使用 [部署说明](DUAL_SCALEBFM_DEPLOY_ZH.md)
第 3–4 节。启动 SteamVR 并确认三 Tracker 可用后，终端 1：

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
bash scripts/run_dual_bridges.sh
```

终端 2 的接口形式如下。当前默认 motion 已通过仿真搬运验证，但没有实机验证；下列命令说明入口形式，
不是该模型已获实机搬运验证的结论。

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
bash scripts/run_dual_scalebfm_residual.sh \
  --config config/g1/dual_scalebfm_contact_v2_8192.yaml \
  --pose-source vive \
  --vive-config /home/bcj/wyr/motion_tracking/sim2real/config/g1/omnicontact_vive_dual.json \
  --act-robot both \
  --confirm-actuation ENABLE_MOTORS \
  --no-visualization
```

`--vive-config` 使用绝对路径；若写相对路径，则相对于部署 YAML 所在目录解析。
实机按钮由 bridge 状态上报，使用 Start/B/A/Stop；不是在策略终端输入 s/b/a。
wrapper 已同时声明 `dual-policy` 和 `vive` 依赖。

## 仍然存在的仿真实机差异

- 仿真两份关节状态和三物体位姿属于同一个精确快照，以时间戳配对；重传只重发缓存命令，
  不重复推进网络历史。实机两 bridge 和 Vive 异步，按本机到达时间、偏斜与位姿新鲜度检查，
  不能要求两个机器人时间戳完全相同。当前没有插值/外推到统一采样时刻。
- 仿真 lockstep 等待控制命令期间物理不会继续前进。实机持续运动，因此相同的计算耗时保护
  不能让 lockstep 覆盖网络抖动、丢包和连续失联时的真实动态。当前没有网络故障注入或实时物理模式。
- 仿真启动已在 DefaultPose，并在 ZERO TORQUE/DefaultPose 锁定两个 base 和箱子；B 后同步释放。
  实机从测量关节角过渡，没有软件浮动基座锁定，支撑条件不同。合成测试覆盖插值逻辑，但默认场景
  不能证明任意实机初始关节角都能安全完成过渡。
- 仿真位姿为 MuJoCo 真值；实机持续读取三个 Tracker，并应用标定外参，受噪声、遮挡、延迟影响。
  双机当前没有单机器人那种启动时读取一次 Vive 并导入场景的选项。
- 当前仿真为了让单 G1 LocoMode 稳定，使用单机的关节 armature/阻尼/摩擦，保留双机训练碰撞几何。
  这不是原训练物理模型的完整复制，也不代表已辨识实机动力学。
- 当前 Vive JSON 箱子尺寸仍须按实物标定，与 reference 的半尺寸约 `[0.48190537, 0.15, 0.15]`
  一致；不能用现有 `[0.15, 0.15, 0.15]` 直接认为通过。几何检查在 A 时执行。
- 双机任务仍按 reference 执行，没有单机器人的 `--goal-position` 接口；统一运行流程不等于统一策略模型。

## 历史验证：motion_000000（默认已更换）

共享流程、禁用命令、异步 bridge 时间戳、传感器失效、双侧停止、MuJoCo 物理、
原有双机策略组件与回放上次共 57 项、本次修复后共 64 项相关回归测试通过；部署 wrapper 的 `--help`、shell 语法和 Python 编译检查通过。实际模型另做了两次 headless UDP 闭环：

1. **站立通过**：s → 两秒过渡 → b → 约 15 秒站立 → x，751 个站立 tick；
   两台最终 pelvis 高度约 0.76775 m，最大倾角约 8.36°，站立控制处理 P95 约 1.14 ms。
   日志：`logs/dual_scalebfm_contact_v2_8192/20260905_204204_954372772/`。
2. **上次“倾角失败”结论更正**：当时第 68 帧后的 0.754 rad 退出由新增站立倾角门限触发，
   不能据此认定 checkpoint 失效。reference 本身 pelvis 最大倾角为 1.254/1.218 rad，
   第 68 帧约 1.243/1.190 rad。这是有意俯身的动作，站立用的 0.7 rad 门限不适用。
   原始日志保留：`logs/dual_scalebfm_contact_v2_8192/20260905_204238_314224583/`。
3. **本次恢复验证**：恢复任务限幅 1.0，仅在 LocoMode 检查绝对倾角；A 切换保留前一条
   PD 命令作为限幅器锚点；箱子误差检查对齐上一条已执行命令，而不是下一条尚未发送的目标。
   仿真通过俯身和抬箱，之后在已执行 reference 第 143 帧触发箱子高度误差 0.103 m > 0.1 m。
   该次复测没有完成整段 motion，不能宣称已恢复完整跟踪，也不应把该次结果归因于 checkpoint。
   日志：`logs/dual_scalebfm_contact_v2_8192/20260905_205627_270683896/`。

本次新增了动态目标不被过度限幅、motion 俯身不会触发站立保护、站立倾角保护仍有效、
任务误差按已执行参考检查、A 切换保持限幅器连续性的回归测试。
没有运行实机或实际 Vive 测试。当前证据证明上次整合引入了不适当的限幅和倾角退出，
这两项回退已修正。上述 motion_000000 复测未完整通过；后续选中的 motion_000029
已通过两次完整闭环，详见本文顶部的默认 motion 筛选报告。
