# 双机 sim2sim 按 B 后倒地：原因与修复

日期：2026-09-05。使用 `config/g1/dual_scalebfm_contact_v2_8192.yaml` 和真实
`motion_000000.npz` 布局复现。本次仅验证 DefaultPose → LocoMode → Stop，不执行搬箱。

## 结论

上次重构对齐了 `s/b/a/x` 状态机，但未完整对齐单机的目标处理路径及关节被动动力学。
双机 LocoMode 使用的 ONNX、观测顺序、DefaultPose 和 PD 增益均沿用单机；
以下两项差异导致同一模型不能正常站立。

## 原因 1：额外的估算力矩限制改写了 LocoMode 目标

单机路径：

```text
LocoMode target → 关节位置/变化量限制 → 发出 PD 目标
→ 每个 200 Hz 子步用最新 q/dq 计算 PD → 裁剪实际力矩
```

原双机路径在发出目标前还经过 `RobotSession` 的估算力矩限制：根据 50 Hz 状态估计
PD 力矩，裁剪后反算新的关节目标。这个新目标被保持到下一次策略更新，改变了随后
物理子步的恢复力和速度阻尼效果，不能等价于单机的逐物理步力矩裁剪。

已修复：仅仿真 DefaultPose/LocoMode 跳过这一步目标反算。关节位置/变化量限制和
仿真每个物理子步的实际力矩上限保留；residual 执行阶段和实机仍保留原估算力矩限制。

## 原因 2：训练 XML 的关节被动参数不适合直接复用单机 PD

双机物理模型使用 `g1_scalebfm_training.xml`，单机使用 `g1_29dof.xml`。
单机关节 armature 为 0.01，阻尼为 0.001，摩擦损失为 0.1；训练 XML 使用不同的
电机折算惯量，例如部分腕关节 armature 为 0.003609725。
同一套 LocoMode PD 在这些不同参数下会产生不同的闭环响应。

对照实验保留原碰撞几何、质量、关节限位和力矩上限：

| 处理 | 结果 |
| --- | --- |
| 原目标反算 + 原训练被动参数 | 释放 base 后约 2 秒倒地 |
| 去掉 LocoMode 目标反算，保留训练被动参数 | 30 秒未倒地，但高频抖动及大幅水平位移仍存在 |
| 去掉目标反算，并对齐单机关节被动参数 | 30 秒站立，最终水平位移约 4 cm |

当前两个双机配置都显式加入：

```yaml
simulation:
  joint_dynamics_xml: assets/g1_29dof.xml
```

仿真器按关节名称读取该 XML 的 `dof_armature`、`dof_damping`、`dof_frictionloss`，
应用到 A/B 两台关节。保留双机训练碰撞几何、刚体质量及力矩上限。
这些参数对整个仿真固定生效，不会在按 A 时切换。当前仿真因此不再严格等价于原训练
XML 的被动动力学，搬箱策略需要在这套配置下重新验收。

## 其他运行修复

- UDP 两侧可能分别读到重传旧帧和下一新帧。现在在超时范围内补读旧的一侧，配成同一
  快照后才计算命令；持续无法匹配仍停止，不使用混合状态计算。
- 同一状态快照重传只重发缓存命令，不重复推进 LocoMode 历史。
- 一键脚本正常停止时等待部署进程保存轨迹，避免仿真先退出后脚本立刻杀掉部署进程、
  造成 `rollout.npz` 写入中断。

## 完整进程验证

使用真实模型产物、真实 reference、正式关节限位和力矩上限，经本机 UDP 两进程链路运行。
按键为 `s → b → 站立 30 秒 → x`，没有按 A。

| 指标 | Robot A | Robot B |
| --- | --- | --- |
| LocoMode 帧数 | 1500 | 1500 |
| LocoMode 日志时长 | 29.976 s | 29.976 s |
| Pelvis 最低高度 | 0.7517 m | 0.7517 m |
| Pelvis 最高高度 | 0.7930 m | 0.7930 m |
| 最终 pelvis 高度 | 0.7689 m | 0.7689 m |
| 最终水平位移 | 0.0404 m | 0.0403 m |
| 最大倾斜角（含初始过渡） | 8.36° | 8.36° |

两进程正常退出，最终状态为 `stopped`，轨迹完整保存：

- [终端日志](logs/dual_scalebfm_contact_v2_8192/20260905_183102/console.log)
- [结构化轨迹](logs/dual_scalebfm_contact_v2_8192/20260905_183102/rollout.npz)
- [统计结果](logs/dual_scalebfm_contact_v2_8192/20260905_183102/standing_validation.json)

相关回归测试 83 项通过，另有 5 个 subtest 通过。新增/增强验证包括：真实关节限位和
力矩上限下的 10 秒站立及水平位移检查、两台被动参数与单机一致、LocoMode 不改写目标、
residual 仍保留估算力矩限制、UDP 重传配对。
先前站立测试用了宽泛关节限位和统一 88 Nm 力矩上限，因此没有暴露正式配置下的问题；
现在已移除这两项不符合实际的简化。

## 重新启动

退出已有仿真和部署进程后，仍使用原来的两条命令。

终端 1：

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
bash scripts/run_dual_scalebfm_simulator.sh \
  --config config/g1/dual_scalebfm_contact_v2_8192.yaml
```

终端 2：

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
bash scripts/run_dual_scalebfm_residual.sh \
  --config config/g1/dual_scalebfm_contact_v2_8192.yaml \
  --pose-source sim \
  --act-robot both \
  --confirm-actuation ENABLE_MOTORS \
  --no-visualization
```

等待 `ZERO TORQUE` → `s` → 确认 DefaultPose → `b` → 确认站立 → `a`。
这次不需要重新生成模型产物。当前结论仅覆盖按 B 后的站立问题，不代表整段搬箱已通过。
