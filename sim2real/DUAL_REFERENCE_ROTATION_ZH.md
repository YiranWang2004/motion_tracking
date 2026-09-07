# 默认原地抬放参考：世界 Z 轴顺时针旋转 90°

2026-09-07 已将 `dual_scalebfm_contact_v2_8192.yaml` 使用的默认 artifact 替换。
源动作保留于 `/home/bcj/wyr/Dual_G1_MJ/results/cfgen_box1m_lift_drop.npz`。
新动作：`/home/bcj/wyr/Dual_G1_MJ/results/cfgen_box1m_lift_drop_yaw_minus90.npz`。

变换绕世界原点，`(x,y,z) → (y,-x,z)`。所有帧、两个机器人、箱子、身体/末端世界位置、
朝向及世界线速度和角速度统一变换。关节角、关节速度、接触标志、箱子局部接触点及局部尺寸
保持不变。参考仍为 430 帧、50 Hz，`start_frame=1`，`reference_alignment: none`。
箱子半尺寸仍为 `[0.5,0.15,0.15]`，通过箱子 yaw=-90° 将长边转向世界 Y。

新起点 A `[0,-0.9,0.77]`，yaw=90°；B `[0,0.9,0.77]`，yaw=-90°；
箱子 `[0,0,0.15]`，yaw=-90°。双机间距保持 1.8 m。
上次实际采样间距约 1.336 m，因此此版本仅完成精确 90°旋转，并非完全匹配实测起点。
按上次采样计算 A/B 位置误差约 0.326/0.196 m，可能仍触发预检；未放宽保护门限。

生成及安装命令：

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
uv run python scripts/rotate_dual_reference.py \
  --source /home/bcj/wyr/Dual_G1_MJ/results/cfgen_box1m_lift_drop.npz \
  --output /home/bcj/wyr/Dual_G1_MJ/results/cfgen_box1m_lift_drop_yaw_minus90.npz \
  --yaw-degrees -90
# 生成脚本拒绝覆盖已有输出；上面的生成步骤已完成。
uv run --extra dual-policy python scripts/prepare_dual_scalebfm_artifacts.py \
  --source-root /home/bcj/wyr/Dual_G1_MJ \
  --residual-checkpoint /home/bcj/wyr/Dual_G1_MJ/saved_checkpoints/contact_v2_ctrl_2_8192/checkpoints/best_agent.pt \
  --reference-bundle /home/bcj/wyr/Dual_G1_MJ/results/cfgen_box1m_lift_drop_yaw_minus90.npz \
  --output config/g1/dual_policy_artifacts_contact_v2_8192
```

原来的 sim2sim 启动命令不变；重启即可使用新动作。源 NPZ 未覆盖，可以用相同准备脚本指定
原始源 NPZ 恢复。其他五个模型/FK artifact 的 SHA256 均保持不变。
新 reference SHA256：`10e4322667bdfb0410bc5a039574b06ba90cbbe5fdad5725504ed69ba82ace8a`。

已验证全帧身体位置的精确 90°变换、四元数单位长度、关节与局部接触数据不变、全部六个
artifact 的 manifest 校验。未执行新动作的动态闭环或实机控制。
