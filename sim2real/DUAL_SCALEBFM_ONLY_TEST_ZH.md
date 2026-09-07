# 双机器人纯 ScaleBFM 基线测试：sim2sim 与实机

> 2026-09-06：双机 B 阶段现统一使用 **ScaleBFM 跟踪静态 DefaultPose**，任务完成后也回到该阶段；正式入口不再加载走路策略 `LocoMode.onnx`。启动命令不变，sim2sim 与实机共用实现。机制与最新验证见 [ScaleBFM 站立说明](DUAL_SCALEBFM_STANDING_ZH.md)。

更新：2026-09-06。

在运行 ScaleBFM + residual actor 前，可通过同一个部署入口添加
`--scalebfm-only`，测试 ScaleBFM 本身的空载 reference 跟踪能力。
该选项同时支持 `--pose-source sim` 和 `--pose-source vive`，不复制另一套控制器。

## 测试边界与共用流程

| 环节 | 纯 ScaleBFM 模式 |
| --- | --- |
| DefaultPose / ScaleBFM 站立 | 两秒 PD 过渡后，ScaleBFM 批量跟踪静态 DefaultPose |
| A 后的 ScaleBFM | 保留双机器人 batch=2 推理 |
| Residual checkpoint | 不要求存在，不加载，不检查其文件 checksum |
| Residual actor / 归一化 / observation 构造 | 不执行 |
| 策略目标 | 等于 ScaleBFM 原始关节目标 |
| ScaleBFM 历史动作 | 只写入 ScaleBFM raw action，不添加 residual |
| 关节限制、PD 和估计力矩限制 | 继续使用现有 RobotSession 路径，因此最终 command_target 可能与策略原始目标不同 |
| Reference 与机器人检查 | 保留机器人几何预检、状态超时、站立阶段倾角和停止处理 |
| 箱体与箱子检查 | 删除 MuJoCo 箱体几何；跳过箱子位置/尺寸/朝向预检和箱子误差终止 |
| 实机 bridge/Vive 与仿真状态输入 | 继续通过原有后端接入 |
| 两机帧同步 | 同一个 coordinator / reference 帧计数器 |
| 日志 / 回放 | 共用记录格式，metadata 新增 policy_mode |

本测试为**空载跟踪**：MuJoCo 中没有可见箱体或箱体碰撞，实机只需两台机器人的
Tracker，不读取箱子 Tracker。Reference 仍沿用现有 NPZ，包括原动作中的机器人轨迹。
为兼容现有日志/数据协议，object 字段保留为占位数据，不参与预检或任务终止；
`metadata.json` 记录 `object_enabled=false`。
`control_mode`、`future_step` 和限幅都沿用 YAML，不另行改变。

`--scalebfm-only` 与把 `residual_scale` 设置为零不同：该标志会跳过 checkpoint
依赖和整个 residual 推理分支。仍要求其他五类资源及 manifest 有效。
默认不加该选项，行为仍为 ScaleBFM + residual。

## 1. 单元测试：无需 bridge、Vive 或 residual checkpoint

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
PYTHONPATH=src uv run --extra dual-policy --with pytest python -m pytest \
  tests/test_dual_scalebfm_only.py -q
```

五个测试用例覆盖：

- 禁止加载 actor 和构造 residual observation 时，基线仍可执行。
- 输出等于 ScaleBFM 目标，residual 为零，下一周期历史动作没有 residual 污染。
- Reference 连续推进至末帧。
- 原有 residual 模式仍按原公式叠加目标与历史动作。
- 基线不依赖 residual 文件，但仍校验基础资源；sim/vive 两端均接受同一个标志。

这些单元测试使用合成 reference 和受控网络输出，验证软件逻辑，不能代替物理闭环。

## 2. sim2sim：一键启动

使用当前默认 cfgen_box1m_lift_drop（原地抬放）：

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
bash scripts/run_dual_scalebfm_sim2sim.sh \
  --config config/g1/dual_scalebfm_contact_v2_8192.yaml \
  --scalebfm-only
```

无窗口时加 `--headless`，在启动终端逐个输入 s/b/a/x，每次回车。
有窗口时在 MuJoCo 窗口按键。

使用 1 米箱子的横移动作：

```bash
bash scripts/run_dual_scalebfm_sim2sim.sh \
  --config config/g1/dual_scalebfm_contact_v2_box1m.yaml \
  --reference-bundle /home/bcj/wyr/Dual_G1_MJ/results/cfgen_box1m_lateral_0p5m.npz \
  --scalebfm-only
```

纯 ScaleBFM 模式不检查仿真箱体尺寸与 NPZ 的一致性，因为没有物理箱体。
恢复完整 residual 模式时，箱子尺寸仍必须匹配。
原地抬放时将 reference 路径替换为 `cfgen_box1m_lift_drop.npz`。

### 分两个终端启动

终端 1，MuJoCo 也必须收到空载标志：

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
bash scripts/run_dual_scalebfm_simulator.sh \
  --config config/g1/dual_scalebfm_contact_v2_8192.yaml \
  --scalebfm-only
```

终端 2，部署控制器：

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
bash scripts/run_dual_scalebfm_residual.sh \
  --config config/g1/dual_scalebfm_contact_v2_8192.yaml \
  --pose-source sim \
  --scalebfm-only \
  --act-robot both \
  --confirm-actuation ENABLE_MOTORS \
  --no-visualization
```

若覆盖 reference，两个终端必须传相同的 `--reference-bundle`；一键脚本自动转发给两端。

## 3. 实机：双 bridge + Vive

先按 [双机部署说明](DUAL_SCALEBFM_DEPLOY_ZH.md) 配置网络和两台机器人 Tracker 标定，
启动 SteamVR。纯 ScaleBFM 模式也会实际发送 PD 命令；两台机器人的支撑、
站立确认和停止流程与完整策略相同。

下面以 1 米箱子横移 reference 中的机器人动作为例，实际按空载执行。
Vive JSON 只需提供机器人 A/B 的 Tracker、世界外参、两个 Tracker 到 pelvis 的外参，
以及 `calibration_confirmed`；`object_tracker_serial`、`object_tracker_to_object` 和
`object_half_extents_m` 在此模式下可省略，存在时也不会作为箱子输入使用。
完整 residual 模式仍要求有效箱子标定和三个 Tracker。
本次没有自动修改用户标定 JSON，也没有运行实机。

终端 1，启动两个 bridge：

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
bash scripts/run_dual_bridges.sh
```

终端 2，纯 ScaleBFM 部署：

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
bash scripts/run_dual_scalebfm_residual.sh \
  --config config/g1/dual_scalebfm_contact_v2_box1m.yaml \
  --reference-bundle /home/bcj/wyr/Dual_G1_MJ/results/cfgen_box1m_lateral_0p5m.npz \
  --pose-source vive \
  --vive-config /home/bcj/wyr/motion_tracking/sim2real/config/g1/omnicontact_vive_dual.json \
  --scalebfm-only \
  --act-robot both \
  --confirm-actuation ENABLE_MOTORS \
  --no-visualization
```

需要 A-only 或 B-only 接口检查时，将 `--act-robot both` 改为 `a` 或 `b`；
该模式不能用于验证双机器人搬运。`--act-robot none` 仍发送零力矩/阻尼命令，
不是被动监听。不要用 4 秒 duration 期待自动走完人工门控。

实机阶段按钮来自 bridge 状态：任意一台遥控器的 Start/B/A/Stop 统一控制两台。
推理终端不读取 s/b/a/x。前台单独运行 bridge 时，可使用 bridge 自身的键盘输入；
一键运行两个 bridge 的终端不转发键盘。无需两人同时按 A，两个机器人使用同一帧 reference。

## 4. 两端共同操作与日志

```text
WAITING FOR BRIDGES
→ ZERO TORQUE（首帧有效）
→ Start/s
→ 两秒 DefaultPose 过渡
→ DefaultPose ready
→ B/b
→ 确认两台 ScaleBFM DefaultPose standing 稳定
→ A/a
→ ScaleBFM-only reference
→ 正常完成后回到 ScaleBFM DefaultPose standing
→ Stop/x
```

启动时应看到 `policy_mode=scalebfm_only`；按 A 后显示
`A accepted: executing dual ScaleBFM-only reference`。

日志目录沿用 YAML 的 `log_directory`，也可以用 `--log-dir` 指定绝对路径。
`metadata.json` 中 `policy_mode` 为 `scalebfm_only`。
任务帧的 `residual` 全为零；`observation` 为未计算 residual observation 的零占位。
`scalebfm_target` 记录基础策略目标，`command_target` 记录 PD 限制后实际发送的目标。
`inference_time_s` 是整段策略处理耗时；`processing_time_s` 还包含控制阶段处理。

要切回完整带箱策略，从两端移除 `--scalebfm-only`，并恢复与 reference 匹配的
物理箱体及实机箱子 Tracker。两端模式不一致会明确报错；一键脚本会自动同步标志。

## 5. 可重复的自动物理检查

此命令通过进程内传输运行真实模型和 MuJoCo，不打开硬件 socket；自动执行
DefaultPose → ScaleBFM 站立 → reference → 完成后站立 3 秒，并将是否通过写入 JSONL：

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
PYTHONPATH=src uv run --extra dual-policy python scripts/screen_dual_sim_motions.py \
  --config config/g1/dual_scalebfm_contact_v2_8192.yaml \
  --motions /home/bcj/wyr/Dual_G1_MJ/results/cfgen_batch_128 \
  --start 29 --count 1 \
  --scalebfm-only \
  --output logs/scalebfm_only_motion29.jsonl
```

筛选脚本正常退出只表示结果已写入；验收应检查 JSONL 的 `passed` 和 `reason`。
它不验证实际 UDP 延迟或墙钟实时预算。GUI/UDP 及实机使用前面的正式入口。

## 6. 历史验证：此前保留箱体的基线

新增 5 项基线测试通过；本次相关回归共 75 项通过。
另使用独立本机端口、真实基础模型和当时默认 motion_000029 跑了实际 UDP headless 闭环：

- 正常进入 DefaultPose、LocoMode 和 ScaleBFM-only。
- 执行 69 个任务帧，任务 residual 最大绝对值为 0。
- 控制处理 P95 约 7.03 ms。
- 在已执行第 69 帧触发箱子高度误差 0.106 m > 0.1 m，发送双侧阻尼并退出。
- **这次纯 ScaleBFM 没有完成整个带箱动作**。代码与开关验证通过，不等于物理任务通过。
  未放宽任务误差阈值来制造通过结果，也未执行实机测试。

日志：`logs/dual_scalebfm_only_validation/20260906_205356_852994327/`，
包含 `rollout.npz`、`metadata.json`、`validation.json`；终端输出在其上级 `console.log`。

## 7. 历史空载模式验证（2026-09-06，替换站立策略之前）

- 相关回归 80 项通过，新增箱体几何不存在、箱子误差不终止、两 Tracker 无箱子配置、
  机器人位置预检仍有效等检查。
- 使用真实 ScaleBFM、当前默认 `cfgen_box1m_lift_drop.npz` 和实际 UDP 的 headless
  空载闭环通过：连续执行第 1～429 帧，residual 全零。
- Reference 完成后回到 LocoMode，继续站立 572 tick（11.44 秒），按 x 正常退出。
- 最终两台 pelvis 高度约 0.7673 m；任务处理耗时 P95 约 9.24 ms。
- 没有执行实机测试。

日志：`logs/dual_scalebfm_only_validation/20260906_210716_473568192/`，含
`rollout.npz`、`metadata.json`、`console.log` 和 `validation.json`。

## 8. 当前 ScaleBFM 站立闭环验证

82 项相关测试通过。真实模型与 UDP 空载闭环完成默认原地抬放动作第 1～429 帧，
随后回到 ScaleBFM DefaultPose 站立并正常停止。站立阶段最大倾角约 3.22°。
日志与详细统计见 [当前站立验证](DUAL_SCALEBFM_STANDING_ZH.md)。未执行实机测试。

## 从实测布局启动 sim2sim

现支持 `--initial-scene-source vive` 一次性导入双机 pelvis 和箱子完整位姿，
之后关闭 OpenVR，由 MuJoCo 独立演化。空载模式只读取两台机器人。
完整命令、标定要求和验证范围见 [Vive 实测场景启动指南](DUAL_VIVE_INITIAL_SCENE_ZH.md)。
