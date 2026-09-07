# 双 G1 默认 motion 筛选与替换

> 2026-09-06 默认动作更新：现已改为 `cfgen_box1m_lift_drop.npz`（430 帧），箱子半尺寸 `[0.5, 0.15, 0.15]`。下文 motion_000029 的筛选与闭环结果保留为历史，不是新默认动作的验证结果。

2026-09-05：用户使用的 `config/g1/dual_scalebfm_contact_v2_8192.yaml` 默认
reference 已由 `motion_000000.npz` 替换为 **`motion_000029.npz`**。

来源：`/home/bcj/wyr/Dual_G1_MJ/results/cfgen_batch_128/motion_000029.npz`。
这是物理闭环完成验证，不是只检查 NPZ 或离线推理成功。

## 当前默认值

| 项目 | 数值 |
| --- | --- |
| Reference 总帧数 | 598 |
| 控制频率 | 50 Hz |
| 初始化 / 执行 | `start_frame=1`，连续执行第 1～597 帧 |
| 动作执行时长 | 约 11.94 秒；不包括等待按键和站立 |
| 箱子半尺寸 | `[0.48190537, 0.15, 0.15]` m |
| 箱子全尺寸 | 约 `0.96381 × 0.3 × 0.3` m |
| 箱子质量 | 0.5 kg，与原场景一致 |
| Checkpoint | 仍为 `contact_v2_ctrl_2_8192/checkpoints/best_agent.pt` |
| Reference SHA256 | `eac65af4a72e34633005b2ed2ba7de408a42ffc86cf8ab6ee1ece9ab3d52a154` |

通过产物准备脚本重新生成了 reference 与 manifest，并核对其他五个产物 SHA256
未改变。只更新用户指定的 contact_v2_8192 配置对应产物；其他 checkpoint 的配置
没有被改成使用此动作。

## 筛选与复测

筛选使用实际 ScaleBFM、residual、LocoMode、PD 限幅和 MuJoCo 物理。
每条 motion 的箱体尺寸与 reference 一致；在编译 MuJoCo 模型之前设置碰撞几何，
按保留的质量重新计算长方体惯量，使碰撞边界和动力学参数一致。

筛选没有放宽退出条件：箱子高度误差 0.1 m、位置误差 0.3 m；LocoMode 倾角
0.7 rad。首先采用进程内传输筛选，要求 reference 全部完成、回到 LocoMode 后再站立
3 秒。进程内筛选不验证 UDP 或墙钟实时预算，因此通过候选再用正式启动脚本复测。

`motion_000000` 在筛选中复现第 143 帧箱子高度误差退出。多个其他候选虽然能执行
数百帧，但没有完整结束，未被选用。选中 `motion_000029` 后，实际 UDP 复测结果如下：

| 验证 | B 后约 5 秒按 A | B 后约 2 秒按 A |
| --- | --- | --- |
| Reference 加载 | 临时候选覆盖参数 | 已更新的默认配置，无覆盖参数 |
| 执行帧 | 1～597，全部连续 | 1～597，全部连续 |
| 完成后站立 | 404 tick，约 8.08 秒 | 404 tick，约 8.08 秒 |
| 最大箱子高度误差 | 0.07370 m | 0.08053 m |
| 最大箱子位置误差 | 0.18911 m | 0.23691 m |
| 任务控制处理 P95 | 8.20 ms | 8.17 ms |
| 最终 pelvis 高度 A/B | 0.76855 / 0.76853 m | 0.76858 / 0.76802 m |
| 退出 | x，正常结束 | x，正常结束 |

误差按输入状态对应的上一条已执行参考计算，与运行检查一致。
两次复测都出现 `Reference complete; returning to LocoMode standing`，没有 FAULT。
这些是当前本机配置下的仿真结果，不代表实机验证或任意干扰下的成功保证。

记录位置：

- 批量筛选：`logs/motion_screen/*results.jsonl`，找到通过项后停止，没有声称测完全部 128 条。
- 汇总：`logs/motion_screen/selected_motion_validation.json`。
- 第一次 UDP：`logs/dual_scalebfm_contact_v2_8192/20260905_210545_321051996/`。
- 默认配置 UDP：`logs/dual_scalebfm_contact_v2_8192/20260905_210629_570680739/`。

两次 UDP 目录都包含 `rollout.npz`、`metadata.json`、`console.log`、`validation.json`。
相关回归测试共 65 项通过，包含新增的箱体尺寸/惯量检查。

## 启动命令

命令不变，重启进程后即使用新默认动作：

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
bash scripts/run_dual_scalebfm_sim2sim.sh \
  --config config/g1/dual_scalebfm_contact_v2_8192.yaml
```

操作：等待 ZERO TORQUE → s → 等待 DefaultPose ready → b → 确认站立稳定 → a。
动作约 12 秒后结束，自动返回 LocoMode，按 x 退出。正常完成与提前 FAULT 不同。
仿真日志在非任务阶段显示 `frame=-、object_error=n/a`，避免完成后把箱子当前位置
与 reference 第 1 帧比较，误报一个没有任务意义的大误差。

## 再次筛选或重建产物

以下筛选不连接实机，不自动改写默认产物：

```bash
PYTHONPATH=src uv run --extra dual-policy python scripts/screen_dual_sim_motions.py \
  --config config/g1/dual_scalebfm_contact_v2_8192.yaml \
  --motions /home/bcj/wyr/Dual_G1_MJ/results/cfgen_batch_128 \
  --output logs/motion_screen/recheck.jsonl \
  --start 29 --count 1
```

重建当前默认 artifact：

```bash
uv run --extra dual-policy python scripts/prepare_dual_scalebfm_artifacts.py \
  --source-root /home/bcj/wyr/Dual_G1_MJ \
  --residual-checkpoint /home/bcj/wyr/Dual_G1_MJ/saved_checkpoints/contact_v2_ctrl_2_8192/checkpoints/best_agent.pt \
  --reference-bundle /home/bcj/wyr/Dual_G1_MJ/results/cfgen_batch_128/motion_000029.npz \
  --output config/g1/dual_policy_artifacts_contact_v2_8192
```

不要只复制 reference 而忽略 manifest 和箱子尺寸。回退旧动作时，原始
`motion_000000.npz` 仍在源目录，旧 manifest 留存在
`logs/motion_screen/previous_manifest.json`；还需将 contact 配置箱子半尺寸恢复为
`[0.40829325, 0.15, 0.15]`。
