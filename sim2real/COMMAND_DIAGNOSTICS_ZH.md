# 双机部署日志：站立与限幅诊断

正常启用录制时，新启动的 `deploy_dual_scalebfm_residual.py` 自动将这些字段写入同目录的
`rollout.npz`，无需额外参数；`--no-record` 会关闭录制。旧日志无法补回这些中间值。
`metadata.json` 中 `command_diagnostics_schema_version=1` 表示包含此诊断格式。

关节数组形状为 `(采样数, 2, 29)`，机器人顺序 A、B，关节顺序与 `q` 相同；
位置单位 rad，估算力矩单位 N·m。每机器人标志形状为 `(采样数, 2)`。

| 字段 | 含义 |
| --- | --- |
| `scalebfm_target` | 站立阶段为基础 ScaleBFM 原始目标；任务阶段保留原来的基础模型目标含义，未加入 residual。其他阶段为 NaN。 |
| `command_requested_target` | 调用发送接口的原始目标，尚未做禁用机器人替换或限幅；任务阶段包含 residual。 |
| `command_pre_limit_target` | 禁用机器人替换之后、关节位置/逐周期变化限幅之前的目标。 |
| `command_post_target_limit` | 关节位置和逐周期变化限幅之后、估算力矩限幅之前的目标。 |
| `command_target` | 原有字段，全部处理后最终发送的目标，包括限扭反算目标后的关节位置裁剪。 |
| `torque_limit_enabled` | 此命令是否启用了估算力矩限幅。 |
| `torque_limit_triggered` | 逐关节标志：限扭前估算力矩绝对值严格超过配置阈值。 |
| `estimated_torque_pre_limit` | 根据限扭前目标计算的 `Kp*(target-q)-Kd*dq`。 |
| `estimated_torque_clipped` | 上述估算值裁剪到配置的正负力矩阈值之后的值；不是实测力矩，也不保证与最终位置裁剪后的估算值相等。 |
| `command_diagnostics_valid` | 此机器人最后成功发送的命令是否有这些诊断数据。 |

故障/停止的 hold 命令会清除旧诊断：valid 为 false，新增浮点字段为 NaN，触发标志为 false。
未启用限扭时，两个估算力矩字段为 NaN，限扭标志为 false；通过 `torque_limit_enabled`
区分“未检查”和“已检查但未超限”。诊断对应最后成功发送的命令，模拟器重发缓存命令时保留该命令的诊断。

排查顺序：先比较原始目标与 `command_post_target_limit`，再比较后者与 `command_target`，
结合逐关节限扭标志和 `q/dq` 判断变化发生在哪一步。站立阶段仍没有 residual observation，
因此 `residual`、`observation` 保持 NaN。这些记录不会更改控制目标、增益或限幅配置。
