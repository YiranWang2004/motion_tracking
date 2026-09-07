# 双机器人 sim2sim 当前启动命令速查

> 2026-09-07 默认 reference 已替换为原地抬放动作的世界 Z 轴 −90°版本，双机沿 Y 排列；间距仍为 1.8 m。详见 [旋转与验证记录](DUAL_REFERENCE_ROTATION_ZH.md)。此前完整闭环数据属于旋转前动作。

> 当前双机默认采用世界系参考：`reference_alignment: none`。实际 A、B、箱子分别与各自 reference 世界系位姿比较；虚影不随实际 A 平移或旋转。旧的 A 锚定描述仅适用于显式选择 `xyyaw` 的历史配置。位置门限暂沿用 `preflight.max_partner_position_error_m`，现在同时检查 A 和 B。实测标定世界系需与 motion 世界系一致；本次不自动重定位参考、不放宽门限。

> 2026-09-06：双机 B 阶段现统一使用 **ScaleBFM 跟踪静态 DefaultPose**，任务完成后也回到该阶段；正式入口不再加载走路策略 `LocoMode.onnx`。启动命令不变，sim2sim 与实机共用实现。机制与最新验证见 [ScaleBFM 站立说明](DUAL_SCALEBFM_STANDING_ZH.md)。

> 当前默认动作：`cfgen_box1m_lift_drop.npz`（原地抬放），2026-09-06 按用户要求替换。此前 motion_000029 的完整闭环验证为历史结果，不代表新默认动作已通过同样验证。

更新日期：2026-09-05。适用于当前 `contact_v2_ctrl_2_8192/best_agent.pt` 候选部署包。
本文所有路径均使用当前机器的 `/home/bcj/wyr`，所有闭环命令均选择本机 MuJoCo 仿真。

> 历史验证状态（旧 LocoMode 路径）：组件准备和 500 步离线推理已通过。2026-09-05 已修复按 `b` 后倒地：
> 仿真 DefaultPose/LocoMode 不再经过双机 residual 的估算力矩反算目标，
> 并通过 `simulation.joint_dynamics_xml` 对齐单机的关节 armature、阻尼及摩擦。
> 已完成站立回归，此前 motion_000029 另通过两次完整搬箱及回站立复测。
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
| Reference 来源 | `/home/bcj/wyr/Dual_G1_MJ/results/cfgen_box1m_lift_drop.npz` |
| Reference 帧数 / 起始帧 | 430 / `start_frame=1` |
| 箱子半尺寸 | `[0.5, 0.15, 0.15]` 米 |
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
  --reference-bundle /home/bcj/wyr/Dual_G1_MJ/results/cfgen_box1m_lift_drop.npz \
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
| 3 | 确认姿态稳定后按 `b` | 两条同帧 ScaleBFM 站立命令到齐后，同步释放 base 和箱子 |
| 4 | 等待 `ScaleBFM DefaultPose standing`，观察两台是否稳定 | 同一 ScaleBFM 模型批量推理，两台各自维护站立历史；此时不推进搬箱 reference |
| 5 | 仅在稳定后按 `a` | 对齐 reference、执行几何预检；通过后开始双机搬箱，头顶红色圆柱消失 |
| 6 | Reference 结束 | 自动回到 ScaleBFM DefaultPose standing |
| 7 | 按 `x` | 对两台发送禁用阻尼命令并退出，一键脚本清理两个进程 |

提前按 `a` 或 `b` 不会跳过前置阶段。任务完成后再次按 `a` 不会重放，需要重启场景。
终端 `Ctrl+C` 也可结束。

历史说明：旧 LocoMode 第 4 步的倒地曾修复。根因是双机额外改写了 LocoMode 目标，且训练 XML 的
关节惯量与单机不同；两者叠加使原版 PD 控制失稳。当时采用单机的目标处理和关节被动
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
- 双机 ScaleBFM 站立与搬箱策略的控制门控已经实现；组件齐全和离线测试通过，
  不等于实际站立或搬箱验证通过。

## 12. 双机器人：用箱子标定公共世界系

对应单机速查文档的“箱子上表面中心单点快速标定”。双机直接复用
`calibrate_vive_box_world.py`，输入和输出改为 **`config/g1/omnicontact_vive_dual.json`**。
两台机器人和箱子共用这一个 JSON 中的 `world_from_steamvr`，不要分别给 A/B 标定
两个不同的世界系。该步骤用于实际 Vive 感知；纯 MuJoCo sim2sim 不读取这份标定。

### 12.1 准备与摆放

启动 SteamVR，确认 Tracker 可见；此步骤不需要启动 G1 bridge 或策略。
如果不确定箱子 Tracker 的序列号：

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
uv run --extra vive python scripts/identify_vive_trackers.py
```

逐个移动 Tracker，确认对应关系后按 Ctrl+C 退出，检查双机 JSON 中三个序列号：

```bash
jq '{robot_a_tracker_serial, robot_b_tracker_serial, object_tracker_serial}' \
  config/g1/omnicontact_vive_dual.json
```

将箱子 Tracker 的原点放在箱子上表面中心，采样期间保持静止。脚本的方向假设为：
**Tracker +X 沿箱子/任务世界 +X，Tracker +Z 向下**；因此 Tracker +Y 与箱子/世界 +Y
相反，保持右手坐标系。它不是仅凭一个点自动推断地面和朝向，而是利用这个已知摆放姿态
建立世界系。若实际安装有水平偏移、底座高度或不同安装方向，需要修正箱子外参。

下面示例把箱子中心定义为任务世界 `[0, 0, 0.15]` m，上表面/Tracker 原点高度为
`0.30` m，适用于高 0.30 m 且 Tracker 原点按此假设放置的箱子。
`--box-center-world` 是指定的箱子中心世界坐标；`--box-top-z` 是上表面/Tracker 原点的
世界 Z 高度，必须高于中心 Z。有安装高度偏移时需按实际 Tracker 原点高度填写。
这两个参数不是箱子的长宽，不会自动设置箱子尺寸。

### 12.2 只计算、打印，不写 JSON

```bash
uv run --extra vive python scripts/calibrate_vive_box_world.py \
  --vive-config config/g1/omnicontact_vive_dual.json \
  --box-center-world 0.0 0.0 0.15 \
  --box-top-z 0.30 \
  --samples 150
```

默认读取双机 JSON 的 `object_tracker_serial`。需要显式指定时，在命令后添加
`--serial LHR-实际箱子Tracker序列号`。`--samples` 默认 150，最少 20。
没有 `--output` 就只输出计算结果，不会修改 JSON。

### 12.3 备份并保存标定

先备份，再执行写入：

```bash
cp config/g1/omnicontact_vive_dual.json \
  "config/g1/omnicontact_vive_dual.json.bak.$(date +%Y%m%d_%H%M%S)"

uv run --extra vive python scripts/calibrate_vive_box_world.py \
  --vive-config config/g1/omnicontact_vive_dual.json \
  --output config/g1/omnicontact_vive_dual.json \
  --box-center-world 0.0 0.0 0.15 \
  --box-top-z 0.30 \
  --samples 150
```

写入会覆盖 `world_from_steamvr` 和 `object_tracker_to_object`，保留其他字段，
包括 `robot_a_tracker_to_pelvis`、`robot_b_tracker_to_pelvis`、三个序列号、
`object_half_extents_m`、`goal_position_w` 和 `calibration_confirmed`。
因此它既不会代替两台 pelvis 外参标定，也不会自动确认标定有效；原先的
`calibration_confirmed: true` 也会被保留，需要重新核对本次结果。
箱子外参采用简化摆放假设，后续仍应按实际刚性安装关系检查。

### 12.4 检查结果、尺寸与双机预览

```bash
jq '{world_from_steamvr, robot_a_tracker_to_pelvis, robot_b_tracker_to_pelvis,
     object_tracker_to_object, object_half_extents_m, calibration_confirmed}' \
  config/g1/omnicontact_vive_dual.json
```

`object_half_extents_m` 要另行填写实测半尺寸：例如 1.0 × 0.3 × 0.3 m 箱子为
`[0.5, 0.15, 0.15]`；此前默认 motion_000029 对应约
`[0.48190537, 0.15, 0.15]`。不能因为上表面高度都是 0.30 m 就把不同箱子尺寸混用。
实际箱子、所选 reference 和仿真配置的尺寸必须匹配。

确认参数后，可用双机只读 viewer 检查 A/B pelvis、三个 Tracker 和箱子坐标：

```bash
bash scripts/run_dual_scalebfm_viewer.sh \
  --vive-config config/g1/omnicontact_vive_dual.json
```

viewer 不需要策略或 bridge 才能显示 Vive 位姿；没有 bridge 时关节不代表实测关节角。
当前双机配置加载器要求 `calibration_confirmed` 为 JSON 布尔值 `true`，不能直接套用
单机预览工具的 `--allow-unconfirmed`。先核对输出变换和安装参数，再设置确认字段并进行
只读空间检查。两台 pelvis 外参需分别检查，重新挂载任一 Tracker 后应重新核对相应外参。
后续 `--pose-source vive` 的双机部署读取同一份 JSON；修改后需重启已加载它的进程。


## 仿真与实机共用流程（2026-09-05 更新）

两端已统一状态机、两秒 DefaultPose 过渡、分阶段目标限幅、停止语义和任务误差检查。
最新差异、验证范围及切换实机的命令见 [仿真实机流程对齐说明](DUAL_SIM_REAL_ALIGNMENT_ZH.md)。
原有站立验证记录使用旧版过渡/限幅配置，不能替代新版本的搬运验证。

2026-09-05 A 键回退修正：任务 `executing` 限幅恢复 1.0 rad/tick；0.7 rad 绝对倾角
检查仅用于 LocoMode，不能用于包含俯身的 motion。启动命令不变。上次以该倾角退出
推断 checkpoint 失效的结论撤回；验证情况见 [流程对齐说明](DUAL_SIM_REAL_ALIGNMENT_ZH.md)。

## 双机只读孪生的默认箱子尺寸

`bash scripts/run_dual_scalebfm_viewer.sh` 默认显示全尺寸 **X=0.3、Y=1.0、Z=0.3 m**
的长方体（MuJoCo 半尺寸 `[0.15, 0.5, 0.15]`）。初始箱体坐标轴与世界系对齐时，
1 m 长边沿世界 Y 轴；收到实际位姿后仍随 Tracker 的箱体姿态旋转，不锁定世界朝向。Vive JSON 中的尺寸和实时 actual
数据包尺寸不再覆盖孪生显示尺寸；Tracker 仍按既有标定外参更新位姿。

明确指定 CFGen bundle 时，读取 `training_box_half_extents`；没有此字段时读取
全尺寸 `box_size` 并除以 2；两者都没有则保留默认尺寸。无显式 bundle 时，策略
可视化数据中的 reference 尺寸会用于显示当前任务的箱体。实物显示框和参考框使用
同一尺寸；这一显示规则也适用于日志回放。

```bash
bash scripts/run_dual_scalebfm_viewer.sh \
  --reference-bundle /home/bcj/wyr/Dual_G1_MJ/results/cfgen_box1m_lift_drop.npz
```

该参数在实时孪生中用于选择箱体尺寸，并不会单独启动 reference 动作播放。
这次调整只影响只读孪生显示，不更改 Vive 标定文件、部署的几何检查或物理 sim2sim 箱体配置。

默认双机 sim2sim 场景 `config/g1/assets/dual_scalebfm_sim.xml` 使用
`Dual_G1_MJ` 的 `omnicontact-hand` 全身碰撞几何：每只手包含 16 个 mesh
碰撞体，手部摩擦系数为 `(2.0, 0.01, 0.001)`。模型和所需 mesh 已保存在
本仓库，运行时无需引用 `Dual_G1_MJ`。关节动力学、力矩电机和箱子参数仍由
原有 sim2sim 配置控制，启动命令无需增加参数。

## 首帧启动等待（2026-09-06 修复）

启动先显示 `WAITING FOR BRIDGES`，收到两台 bridge 与有效位姿后才显示
`ZERO TORQUE: both bridges and pose ready`。GUI 窗口初始化慢于模型加载是正常的；
之前部署器直接用运行期 0.2 秒超时等待首帧，会偶发 `missing_bridge_state`。
现在首帧等待使用 `simulation.startup_timeout_s`（现有配置为 60 秒），等待期间不发送
使能命令，超时明确报 `startup_timeout: ...`。收到首帧后仍使用原 `state_timeout_s`，
不会给运行中的失联增加 60 秒容忍时间。命令无需修改。

验证：使用独立本机测试端口，先启动部署器，在其 WAITING 后再延迟 3 秒启动实际 headless 仿真器，
成功收到首帧并按 x 正常退出。日志在 `logs/startup_wait_validation/`。

## 纯 ScaleBFM 前置测试

现有入口添加 `--scalebfm-only` 即可跳过 residual actor 的加载和推理。
实机 Vive/双 bridge 和 sim2sim 共用该开关、状态机与控制路径。
单元测试、两端完整启动命令及实际基线结果见
[纯 ScaleBFM 基线测试说明](DUAL_SCALEBFM_ONLY_TEST_ZH.md)。

2026-09-06 更新：`--scalebfm-only` 现在是空载模式，移除仿真箱体和箱子相关检查，
实机只读取两台机器人 Tracker。分开启动时仿真器和部署器都需传该标志；一键脚本自动同步。
详见 [空载 ScaleBFM 测试](DUAL_SCALEBFM_ONLY_TEST_ZH.md)。

## 从实测布局启动 sim2sim

现支持 `--initial-scene-source vive` 一次性导入双机 pelvis 和箱子完整位姿，
之后关闭 OpenVR，由 MuJoCo 独立演化。空载模式只读取两台机器人。
完整命令、标定要求和验证范围见 [Vive 实测场景启动指南](DUAL_VIVE_INITIAL_SCENE_ZH.md)。

## MuJoCo 内置参考动作虚影

现在默认显示青色半透明双机器人 reference；有箱模式同时显示参考箱子。按 G 显示/隐藏。
A 前固定显示世界系任务起始姿态，A 后在同一世界系逐帧同步；空载无箱子虚影。
原命令及 `--no-visualization` 均不影响此内置显示。详见
[实测场景与参考虚影说明](DUAL_VIVE_INITIAL_SCENE_ZH.md)。
