# 双机器人 sim2sim 当前启动命令速查

> 最新默认动作已改为 `motion_000029`，已通过两次完整 UDP 闭环；此前 `motion_000000` 的失败记录保留为历史。详见 [默认 motion 筛选报告](DUAL_DEFAULT_MOTION_SELECTION_ZH.md)。

更新日期：2026-09-05。适用于当前 `contact_v2_ctrl_2_8192/best_agent.pt` 候选部署包。
本文所有路径均使用当前机器的 `/home/bcj/wyr`，所有闭环命令均选择本机 MuJoCo 仿真。

> 当前验证状态：组件准备和 500 步离线推理已通过。2026-09-05 已修复按 `b` 后倒地：
> 仿真 DefaultPose/LocoMode 不再经过双机 residual 的估算力矩反算目标，
> 并通过 `simulation.joint_dynamics_xml` 对齐单机的关节 armature、阻尼及摩擦。
> 已完成站立回归，当前默认 motion_000029 另通过两次完整搬箱及回站立复测。
> 仍按 `s → b → 确认稳定 → a` 操作。训练参数 `control_mode=2`、`residual_scale=0.10`
> 的原始训练命令尚未确认。原因和验证记录见 [LocoMode 站立修复](DUAL_LOCOMODE_FIX_ZH.md)。

## 1. 当前使用的路径与组件

| 项目 | 路径或设置 |
| --- | --- |
| 运行目录 | `/home/bcj/wyr/motion_tracking/sim2real` |
| 训练资源仓库 | `/home/bcj/wyr/Dual_G1_MJ` |
| **本次必须显式指定的配置** | `config/g1/dual_scalebfm_contact_v2_8192.yaml` |
| 已准备的部署产物目录 | `config/g1/dual_policy_artifacts_contact_v2_8192` |
| Residual checkpoint 来源 | `/home/bcj/wyr/Dual_G1_MJ/saved_checkpoints/contact_v2_ctrl_2_8192/checkpoints/best_agent.pt` |
| Reference 来源 | `/home/bcj/wyr/Dual_G1_MJ/results/cfgen_batch_128/motion_000029.npz` |
| Reference 帧数 / 起始帧 | 598 / `start_frame=1` |
| 箱子半尺寸 | `[0.48190537, 0.15, 0.15]` 米 |
| 当前仿真箱子质量 | 0.5 kg，由仿真 XML 定义 |
| 物理 / 策略频率 | 200 Hz / 50 Hz |
| 推理设备 | CPU |

**不要省略下面命令的 `--config`。** 脚本默认读取旧的 `dual_scalebfm_residual.yaml`，
其产物路径是 `dual_policy_artifacts`，不是本次已准备的候选包。

## 2. 首次安装依赖

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
uv sync --extra dual-policy
```

纯 sim2sim 不需要启动 SteamVR、真实 G1 bridge 或双机网络 namespace。
`dual-policy` extra 包含模型推理所需的 PyTorch；安装 OpenVR 依赖不代表会访问 Tracker。

## 3. 产物准备（当前已经完成）

日常启动可跳过本节。只有首次生成或明确更换源模型/reference 时，才重新运行：

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
uv run --extra dual-policy python scripts/prepare_dual_scalebfm_artifacts.py \
  --source-root /home/bcj/wyr/Dual_G1_MJ \
  --residual-checkpoint /home/bcj/wyr/Dual_G1_MJ/saved_checkpoints/contact_v2_ctrl_2_8192/checkpoints/best_agent.pt \
  --reference-bundle /home/bcj/wyr/Dual_G1_MJ/results/cfgen_batch_128/motion_000029.npz \
  --output config/g1/dual_policy_artifacts_contact_v2_8192
```

命令会覆盖指定输出目录中的六个部署资源及 `manifest.json`：

```text
scalebfm_model.pt
scalebfm_metadata.json
scalebfm_mode_table.pt
residual_actor.pt
reference_bundle.npz
g1_29dof_scalebfm.xml
manifest.json
```

重新生成后必须再次运行离线验证；已有验证报告只代表当时使用的资源。
当前配套 YAML 已存在，不需要再从旧默认配置复制。不要单独覆盖权重文件而保留旧 checksum。

## 4. 先做离线验证

此命令不启动仿真、bridge 或电机输出：

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
uv run --extra dual-policy python scripts/validate_dual_scalebfm_policy.py \
  --config config/g1/dual_scalebfm_contact_v2_8192.yaml \
  --steps 500
```

通过时打印：

```text
offline dual policy validation passed
```

本次实测平均 4.534 ms、P95 4.626 ms、最大 4.826 ms，低于 18 ms 预算。
该验证检查资源校验和、FK、数值和推理耗时，不代表物理搬运成功。

## 5. 推荐：一条命令启动 GUI

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
bash scripts/run_dual_scalebfm_sim2sim.sh \
  --config config/g1/dual_scalebfm_contact_v2_8192.yaml
```

脚本同时启动 MuJoCo 和部署进程，并自动为部署进程添加：

```text
--pose-source sim
--act-robot both
--confirm-actuation ENABLE_MOTORS
--no-visualization
```

这里的 `--no-visualization` 只关闭额外的策略孪生数据流，不关闭 MuJoCo 窗口。
`--act-robot both` 控制的是本机仿真，两路地址均为 loopback。

### 5.1 按键操作顺序

按键在 **MuJoCo 窗口** 输入，不是在部署终端输入。

| 顺序 | 操作 | 预期行为 |
| --- | --- | --- |
| 1 | 等待 `ZERO TORQUE` | 两台从原版 DefaultPose 初始化，物理暂不推进 |
| 2 | 按 `s` | 两秒过渡至 DefaultPose，等待 `DefaultPose ready`；两个 base 和箱子保持锁定 |
| 3 | 确认姿态稳定后按 `b` | 两条同帧 LocoMode 命令到齐后，同步释放 base 和箱子 |
| 4 | 等待 `LocoMode standing`，观察两台是否稳定 | 两台独立运行 LocoMode；此时不推进搬箱 reference |
| 5 | 仅在稳定后按 `a` | 对齐 reference、执行几何预检；通过后开始双机搬箱，头顶红色圆柱消失 |
| 6 | Reference 结束 | 自动回到两台独立的 LocoMode standing |
| 7 | 按 `x` | 对两台发送禁用阻尼命令并退出，一键脚本清理两个进程 |

提前按 `a` 或 `b` 不会跳过前置阶段。任务完成后再次按 `a` 不会重放，需要重启场景。
终端 `Ctrl+C` 也可结束。

此前第 4 步的倒地已修复。根因是双机额外改写了 LocoMode 目标，且训练 XML 的
关节惯量与单机不同；两者叠加使原版 PD 控制失稳。现在使用单机的目标处理和关节被动
参数，仍保留真实关节限位、物理子步力矩上限及双机碰撞几何。
`joint_dynamics_xml` 对整个仿真固定生效，不会在按 `a` 时切换惯量；因此当前仿真不再
严格等价于原训练 XML 的被动动力学，搬箱效果需要重新验证。

## 6. 分两个终端启动（便于分别查看日志）

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

按键顺序与第 5 节相同。两终端必须使用同一配置；不要再启动一键脚本形成重复进程。
仿真默认等待部署连接最多 60 秒，因此先后启动两个终端时不要间隔太久。

## 7. 无窗口 headless

一条命令启动：

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
bash scripts/run_dual_scalebfm_sim2sim.sh \
  --config config/g1/dual_scalebfm_contact_v2_8192.yaml \
  --headless
```

在启动命令的终端逐次输入 `s`、`b`、`a`、`x`，**每次按回车**。
不要一次性输入完整序列；`b` 后需要根据状态日志确认站立是否稳定。headless 不会自动启动任务。

需要分两个终端时，终端 1 改为：

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
bash scripts/run_dual_scalebfm_simulator.sh \
  --config config/g1/dual_scalebfm_contact_v2_8192.yaml \
  --headless
```

终端 2 使用第 6 节的部署命令；按键在仿真器所在的终端 1 输入。

### 7.1 可选：限制运行时间

```bash
bash scripts/run_dual_scalebfm_sim2sim.sh \
  --config config/g1/dual_scalebfm_contact_v2_8192.yaml \
  --headless \
  --duration 30
```

`--duration` 是部署进程墙钟时间，**包含等待按键阶段**，不是按 `a` 后才开始计时。
人工调试建议不设置。设置时长不会自动输入任何按键。

## 8. 日志与回放

默认结构化日志目录：

```text
/home/bcj/wyr/motion_tracking/sim2real/logs/dual_scalebfm_contact_v2_8192/<启动时间>/
  rollout.npz
  metadata.json
```

查看已有日志：

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
ls -lt logs/dual_scalebfm_contact_v2_8192
```

同时保留完整终端输出，可使用：

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
mkdir -p logs/dual_scalebfm_contact_v2_8192
set -o pipefail
bash scripts/run_dual_scalebfm_sim2sim.sh \
  --config config/g1/dual_scalebfm_contact_v2_8192.yaml \
  2>&1 | tee "logs/dual_scalebfm_contact_v2_8192/console_$(date +%Y%m%d_%H%M%S).log"
```

回放修复前已保存的失稳记录，不需要 SteamVR、bridge 或策略进程：

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
bash scripts/run_dual_scalebfm_viewer.sh \
  --replay-log logs/dual_scalebfm_contact_v2_8192/20260905_174620/rollout.npz \
  --reference-bundle config/g1/dual_policy_artifacts_contact_v2_8192/reference_bundle.npz \
  --replay-speed 0.5 \
  --replay-paused
```

回放其他记录时，将 `--replay-log` 替换为对应 `rollout.npz` 路径。
回放按键：空格或 `P` 暂停/继续，`N/B` 前进/后退一帧，`A/D` 降低/提高速度，
`R` 从头播放，`G` 切换 ghost。这些是回放控制键，不是仿真任务启动键。

详细验证记录：
[候选包验证报告](config/g1/dual_policy_artifacts_contact_v2_8192/VALIDATION_ZH.md)。

## 9. 停止、端口检查和重启

正常停止用 `x`，也可在启动终端按 `Ctrl+C`。一键脚本会清理两个子进程；
分终端运行时，确认两个终端均已退出后再重启。

| 用途 | Robot A | Robot B |
| --- | --- | --- |
| 仿真状态端口 | `127.0.0.1:56001` | `127.0.0.1:56101` |
| 仿真命令端口 | `127.0.0.1:56002` | `127.0.0.1:56102` |

```bash
ss -lunp | rg ':(56001|56002|56101|56102)\b'
```

没有输出表示这四个 UDP 端口当前没有监听进程。不要同时运行两套使用相同端口的双机仿真。

## 10. 测试与帮助

交互控制、同步和部署回归：

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
PYTHONPATH=src uv run --with pytest python -m pytest -q \
  tests/test_dual_sim_control.py \
  tests/test_dual_scalebfm_sim2sim.py \
  tests/test_dual_scalebfm_deploy.py \
  tests/test_dual_scalebfm_visualization.py \
  tests/test_dual_runtime.py
```

查看入口参数：

```bash
uv run python src/dual_scalebfm_sim2sim.py --help
uv run --extra dual-policy python src/deploy_dual_scalebfm_residual.py --help
uv run --extra vive python scripts/view_dual_scalebfm_residual.py --help
```

## 11. 与单机器人命令的区别

- 当前所有启动命令必须显式选择本候选配置。
- 双机使用 `--act-robot both`，不是单机 `--act`。
- 双机使用 `--duration`，不是单机 `--run-seconds`。
- 初始关节角来自原版 DefaultPose，任务布局来自双机 reference；当前不支持
  `--initial-scene-source vive` 或 `--goal-position`，不会读取 Vive JSON 的 goal 来生成任务。
- `--act-robot none` 仍发送禁用命令，释放 base 后仅施加阻尼，不是完全不发包的评估模式。
  无输出检查请使用第 4 节离线 validator。
- 两台 LocoMode 与搬箱策略的控制门控已经实现；组件齐全和离线测试通过，
  不等于实际站立或搬箱验证通过。

## 仿真与实机共用流程（2026-09-05 更新）

两端已统一状态机、两秒 DefaultPose 过渡、分阶段目标限幅、停止语义和任务误差检查。
最新差异、验证范围及切换实机的命令见 [仿真实机流程对齐说明](DUAL_SIM_REAL_ALIGNMENT_ZH.md)。
原有站立验证记录使用旧版过渡/限幅配置，不能替代新版本的搬运验证。

2026-09-05 A 键回退修正：任务 `executing` 限幅恢复 1.0 rad/tick；0.7 rad 绝对倾角
检查仅用于 LocoMode，不能用于包含俯身的 motion。启动命令不变。上次以该倾角退出
推断 checkpoint 失效的结论撤回；验证情况见 [流程对齐说明](DUAL_SIM_REAL_ALIGNMENT_ZH.md)。
