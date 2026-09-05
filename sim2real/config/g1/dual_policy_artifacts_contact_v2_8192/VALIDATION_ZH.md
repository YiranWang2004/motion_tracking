# contact_v2_ctrl_2_8192 候选部署产物

来源：`/home/bcj/wyr/Dual_G1_MJ`。使用
`saved_checkpoints/contact_v2_ctrl_2_8192/checkpoints/best_agent.pt`；
它不是原默认配置指定的 `omnicontact-hand-1.5kg` checkpoint。

六个部署资源已复制并生成 SHA256 manifest。配套配置：
`config/g1/dual_scalebfm_contact_v2_8192.yaml`。
`control_mode=2` 根据目录名及仓库训练脚本推定；`residual_scale=0.10` 沿用仓库默认值。
原始训练命令尚未确认，TensorBoard 仅含 scalar，没有完整超参数记录。

## 验证结果

- A/B actor 输入 201 维，输出 29 维；两侧 policy/preprocessor 相同，符合共享 actor 结构。
- 默认 reference 含全部所需 training 字段，共 753 帧；箱子半尺寸
  `[0.40829325, 0.15, 0.15]` 与当前仿真配置一致。
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
当前应先排查真实配置下的 LocoMode/PD/限位衔接，再验证搬箱。
