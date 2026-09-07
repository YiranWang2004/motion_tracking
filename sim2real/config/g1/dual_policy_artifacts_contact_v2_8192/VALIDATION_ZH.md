# contact_v2_ctrl_2_8192 候选部署产物

> 2026-09-07 默认 reference 已替换为原地抬放动作的世界 Z 轴 −90°版本，双机沿 Y 排列；间距仍为 1.8 m。详见 [旋转与验证记录](../../../DUAL_REFERENCE_ROTATION_ZH.md)。此前完整闭环数据属于旋转前动作。

> 2026-09-06 默认动作更新：现已改为 `cfgen_box1m_lift_drop.npz`（430 帧），箱子半尺寸 `[0.5, 0.15, 0.15]`。下文 motion_000029 的筛选与闭环结果保留为历史，不是新默认动作的验证结果。

> 当前默认 reference 已改为 `cfgen_batch_128/motion_000029.npz`（598 帧）。
> 已通过完整 sim2sim 跟踪、返回 LocoMode 及继续站立验证。
> 箱子半尺寸为 `[0.48190537, 0.15, 0.15]` m；模型及 checkpoint 未更换。
> 详细筛选记录见 [默认 motion 筛选报告](../../../DUAL_DEFAULT_MOTION_SELECTION_ZH.md)。
> 下文旧的倒地及 motion_000000 测试只保留为历史记录，不是当前验收结论。

来源：`/home/bcj/wyr/Dual_G1_MJ`。使用
`saved_checkpoints/contact_v2_ctrl_2_8192/checkpoints/best_agent.pt`；
它不是原默认配置指定的 `omnicontact-hand-1.5kg` checkpoint。

六个部署资源已复制并生成 SHA256 manifest。配套配置：
`config/g1/dual_scalebfm_contact_v2_8192.yaml`。
`control_mode=2` 根据目录名及仓库训练脚本推定；`residual_scale=0.10` 沿用仓库默认值。
原始训练命令尚未确认，TensorBoard 仅含 scalar，没有完整超参数记录。

## 历史验证结果（motion_000000，首次准备时）

- A/B actor 输入 201 维，输出 29 维；两侧 policy/preprocessor 相同，符合共享 actor 结构。
- 当时 reference 含全部所需 training 字段，共 753 帧；箱子半尺寸
  `[0.40829325, 0.15, 0.15]` 与当时仿真配置一致。
- 500 步 CPU 离线验证通过，平均 4.534 ms、P95 4.626 ms、最大 4.826 ms；预算 18 ms。
- 真实 reference 的 headless 物理测试失败：`s` 保持 DefaultPose 正常，
  `b` 后 LocoMode 站立失稳，两台 pelvis 高度从 0.793 m 降至约 0.10 m。
  `a` 时预检报出 `robot B initial position differs from reference by 4.748 m`。
  失败发生在 residual 策略启动之前，不能归因于该 residual checkpoint。
- 因此本包仅通过资源和离线推理验证，未通过站立及搬箱动力学验收。

终端输出见同目录 `sim_validation.log`，轨迹见
`logs/dual_scalebfm_contact_v2_8192/20260905_174620/rollout.npz`。

## 复现

在 `/home/bcj/wyr/motion_tracking/sim2real` 执行：

```bash
uv run --extra dual-policy python scripts/validate_dual_scalebfm_policy.py \
  --config config/g1/dual_scalebfm_contact_v2_8192.yaml --steps 500

bash scripts/run_dual_scalebfm_sim2sim.sh \
  --config config/g1/dual_scalebfm_contact_v2_8192.yaml
```

仿真按 `s → b`，观察是否稳定；若失稳则按 `x` 停止，不要继续按 `a`。
当前操作：s → 等待 DefaultPose ready → b → 确认稳定 → a；完成后返回站立，按 x 退出。
