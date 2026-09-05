# Dual G1 ScaleBFM Residual 实机部署

> 当前 `contact_v2_ctrl_2_8192` 候选包的完整启动命令、显式配置路径及已知问题，见 [双机 sim2sim 当前命令速查](DUAL_SIM2SIM_QUICK_REFERENCE_ZH.md)。

本入口沿用已经通过 A-only、B-only、both 实机测试的双 namespace 和双
`g1_udp_bridge`。正式策略进程只通过两组 UDP endpoint 向 bridge 发送 29 维绝对 PD
目标，不直接加入 Unitree DDS 网络。

## 1. 准备部署产物

下面的命令会复制 ScaleBFM、MAPPO Actor、CFGen/Kimodo reference 和训练使用的 FK
XML，并生成 SHA256 manifest。每次更换 checkpoint 或 reference 都必须重新执行。

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
uv run --extra dual-policy python scripts/prepare_dual_scalebfm_artifacts.py
```

当前默认 checkpoint 是：

```text
Dual_G1_MJ/results/ablation_collision_omnicontact_hand/
  omnicontact-hand-1.5kg/checkpoints/best_agent.pt
```

默认 reference 是训练集第一条 `motion_000000.npz`。reference 必须是带 `training_*`
字段的双机 bundle，并且必须与要执行的箱子尺寸和
初始 A/B/箱子几何一致。任务几何条件在按 A 进入搬运策略前检查；DefaultPose/LocoMode 在此之前运行。

## 2. 配置三台 Tracker

编辑：

```text
config/g1/omnicontact_vive_dual.json
```

必须填写三个不同的序列号、`world_from_steamvr`、两台机器人的
`tracker_to_pelvis`、`object_tracker_to_object` 和实测箱子尺寸。检查完成后才能设置：

```json
"calibration_confirmed": true
```

先运行以下工具确认三台 Tracker 都连续有效：

```bash
uv run --extra vive python scripts/list_vive_trackers.py --seconds 10
```

需要确认三台设备分别对应哪个序列号时，实时显示所有 Tracker 的原始 XYZ，然后每次只移动
一台，观察对应行：

```bash
uv run --extra vive python scripts/identify_vive_trackers.py
```

确认后把三个固定的 `LHR-...` 序列号分别写入双机配置；不要使用可能在重连后变化的
OpenVR index。按 `Ctrl-C` 结束实时显示。

`calibrate_vive_world.py` 用 O/+X/+Y 三点建立公共世界坐标；
`calibrate_vive_box_world.py` 可以用箱子 Tracker 建立/核查箱子坐标。两台机器人
Tracker 到 pelvis 的外参必须按实际刚性安装位置测量，不能保留零值占位。

## 3. 离线验证策略

此步骤不打开 bridge、不访问机器人、不发送 UDP 命令：

```bash
bash scripts/run_dual_scalebfm_residual.sh --help
uv run --extra dual-policy python scripts/validate_dual_scalebfm_policy.py --steps 500
```

必须满足：所有 checksum 通过、输出有限、shape 为 `(2, 29)`，并且 P95 推理时间低于
配置中的 18 ms。这台主机实测 CPU 比 CUDA 小 batch 更快，因此默认使用 CPU。

## 4. 启动双网络和 bridge

首次接线时，在 `config/g1/dual_network.yaml` 中写入固定的
`robot_a.interface`、`robot_a.expected_mac`、`robot_b.interface` 和
`robot_b.expected_mac`。该文件是双机日常启动的持久配置源，同时保存 namespace、物理口 IP、
veth 和 bridge 配置路径。后续所有命令都自动读取它，不要在命令行重复输入网口或
namespace；更换网卡或接线时也只修改该文件。

修改后先做只读预览，确认 A/B、接口名和 MAC 与现场接线一致：

```bash
bash scripts/setup_dual_network.sh
```

主机每次重启后执行：

```bash
sudo bash scripts/setup_dual_network.sh --apply
sudo bash scripts/setup_dual_network.sh --check
bash scripts/diagnostics_dual_network.sh
```

推荐一条命令同时启动两个 bridge：

```bash
bash scripts/run_dual_bridges.sh
```

任一 bridge 退出时脚本会停止另一侧。需要分开调试时仍可在两个终端执行
`run_dual_bridge_a.sh` 和 `run_dual_bridge_b.sh`，二者也自动读取同一个网络配置。

先重新执行原有最小路由测试，确认 A/B/both 行为仍正确。不要跳过 A-only 和 B-only。

## 5. 正式 policy 分阶段测试

### 5.1 启动只读 MuJoCo 孪生

先验证双机场景和本机 MuJoCo 资源，不打开窗口、不连接机器人：

```bash
bash scripts/run_dual_scalebfm_viewer.sh --check-model
```

启动 SteamVR 并确认三台 Tracker 在线后，即可在独立终端启动实时 viewer；不要求 bridge
或 policy 已经启动：

```bash
bash scripts/run_dual_scalebfm_viewer.sh
```

viewer 默认直接读取 `config/g1/omnicontact_vive_dual.json` 和 OpenVR。三台 Tracker 分别
决定 A pelvis、B pelvis 和箱子的实际空间姿态；两台机器人关节先使用与单机 viewer 相同的
`config/g1/omnicontact/OmniContact.yaml:default_angles_lab`。因此只有 Tracker 在线、两个
bridge 和 policy 都未启动时，也能显示两台 default-pose G1 和实测箱子。

此外，viewer 并行接收三路只读 UDP：

- `55003`：G1 A bridge state mirror；收到后只用 A 的 29 维实测关节角覆盖 default pose。
- `55103`：G1 B bridge state mirror；收到后只用 B 的 29 维实测关节角覆盖 default pose。
- `55204`：policy visualization，提供 CFGen reference、ScaleBFM target、residual 和最终
  PD target；在 OpenVR 直读模式下不会反向覆盖 viewer 自己读取的实测 pelvis/箱子姿态。

两台实测机器人都保留 G1 原生材质，半透明黄色和橙色模型分别是 A/B ghost；实心棕色箱子
是 Tracker 标定后的实测状态，半透明橙色箱子是 CFGen reference。蓝色、紫色和橙色的三个
金字塔及其 XYZ 轴分别表示 A、B 和箱子原始 Tracker 坐标；机器人 pelvis 和实测箱子已经应用
各自的 tracker-to-body 外参。某台 Tracker 短暂丢帧时保留它最后一次有效姿态，另外两台继续
更新。世界原点和实测箱子也显示红 X、绿 Y、蓝 Z 坐标轴，场景、箱子和坐标轴配色与单机
OmniContact 搬箱子 viewer 一致。

只测试 Tracker、明确不接收 bridge 关节镜像时可以执行：

```bash
bash scripts/run_dual_scalebfm_viewer.sh --state-port-a 0 --state-port-b 0
```

如果 viewer 在没有 SteamVR 的远端机器运行，可使用 `--no-vive` 恢复为完全依赖 policy
visualization 中实测姿态的模式：

```bash
bash scripts/run_dual_scalebfm_viewer.sh --no-vive
```

此时三个金字塔只能显示 visualization/replay 中的标定后策略姿态，而不是原始 Tracker 本体。

按 `G` 在“CFGen reference ghost”和“最终发送给 bridge 的限幅 PD target ghost”之间切换。
离线回放时，空格或 `P` 暂停/继续，`N/B` 前后单帧，`A/D` 降低/提高播放速度，`R` 从头播放。
需要只观察箱子、坐标系和 ghost 时可增加 `--no-robot` 隐藏两台实测 G1 网格。实时 visual
数据超过 `--stream-timeout` 后，reference/target ghost 会像单机 viewer 一样自动隐藏，收到
新包后自动恢复，避免把过期参考姿态误认为当前姿态。
viewer 不创建命令 socket、不向 bridge 或机器人发送任何数据，关闭或崩溃不会改变控制环。
同一组 mirror 端口只启动一个 viewer，避免 `SO_REUSEADDR` 下多个进程争用 UDP 数据。

若端口需要修改，policy 使用：

```bash
bash scripts/run_dual_scalebfm_residual.sh \
  --visualization-host 127.0.0.1 --visualization-port 55214
```

viewer 使用相同的 `--visualization-host/--visualization-port`。A/B mirror 端口可分别通过
`--state-port-a` 和 `--state-port-b` 修改，并必须与两个 bridge YAML 一致。

### 5.2 无电机和分阶段使能

第一阶段检查感知和控制接口，`--act-robot none` 仍发送零力矩/阻尼命令，不能当作被动监听；策略推理需按顺序按键进入：

```bash
bash scripts/run_dual_scalebfm_residual.sh \
  --duration 10 \
  --act-robot none
```

确认 preflight 误差、50 Hz 日志和推理耗时正常。日志保存在
`sim2real/logs/dual_scalebfm_residual/<时间>/rollout.npz`。

悬挂或可靠支撑两台机器人后，依次执行：

```bash
# 只使能 A
bash scripts/run_dual_scalebfm_residual.sh \
  --duration 4 --act-robot a --confirm-actuation ENABLE_MOTORS

# 只使能 B
bash scripts/run_dual_scalebfm_residual.sh \
  --duration 4 --act-robot b --confirm-actuation ENABLE_MOTORS

# 同时使能
bash scripts/run_dual_scalebfm_residual.sh \
  --act-robot both --confirm-actuation ENABLE_MOTORS
```

实机和仿真现在都等待人工门控：`ZERO TORQUE → Start/s → 两秒 DefaultPose
过渡 → DefaultPose ready → B/b → LocoMode standing → A/a → ScaleBFM + residual
→ reference 完成后回到 LocoMode → Stop/x`。Start/B/A 必须按顺序重新按下，启动时
已按住的键不会触发阶段切换；提前按 B 后需松开并在 ready 后重新按下。
实机没有浮动基座锁定，应按实际支撑条件完成 DefaultPose 过渡和站立确认。
四秒 `--duration` 示例只适合短时接口检查，包含等待按键时间，不会自动启动任务。
详见 [仿真实机流程对齐说明](DUAL_SIM_REAL_ALIGNMENT_ZH.md)。

任一 bridge 状态超时、双机 state skew 超限、任一 Tracker 丢失、几何预检失败、模型出现
非有限值或连续三帧控制处理超时，都会对两侧发送禁用 hold 并退出。

## 6. 更换 reference 或 checkpoint

不要直接覆盖 artifact 中的单个文件。重新运行准备脚本，使 manifest、模型、reference 和
FK XML 保持一个原子版本。`--reference-bundle` 运行参数只适合临时无电机分析；正式实机
运行应把 reference 写入带 checksum 的 artifact bundle。

## 7. 离线回放孪生状态

正式入口保存的每个 `rollout.npz` 都可以在没有机器人、bridge 和 SteamVR 的情况下回放：

```bash
bash scripts/run_dual_scalebfm_viewer.sh \
  --replay-log logs/dual_scalebfm_residual/<时间>/rollout.npz \
  --replay-speed 1.0
```

回放会使用同目录 `metadata.json` 记录的 reference 路径，按首帧 robot A 姿态重新执行与实机
一致的 `xyyaw` 对齐。reference 文件已移动时，增加
`--reference-bundle /绝对路径/reference_bundle.npz`。按空格或 `P` 暂停，`N/B` 前后单帧，
`A/D` 调整播放速度，`R` 从头播放，`G` 切换 reference/最终 PD target ghost。

实时终端每秒打印 A、B、visual、viveA、viveB 和 viveObj 数据年龄；超过默认 `500 ms` 会
显示 `!`。这是 viewer 告警，不代替 policy 中更严格的 `200 ms` bridge state 和 `100 ms`
Tracker fail-closed 检查。

## 8. 双机器人 sim2sim：与单机一致的按键控制

2026-09-05 站立修复：DefaultPose/LocoMode 使用与单机一致的关节目标限制流程，
实际力矩仍在每个 200 Hz 物理子步裁剪。`simulation.joint_dynamics_xml` 从单机
`assets/g1_29dof.xml` 读取关节 armature、阻尼和摩擦，对整个仿真固定生效。
保留双机碰撞几何，但不再声称与原训练 XML 的被动动力学完全相同。
详见 [倒地原因与修复验证](DUAL_LOCOMODE_FIX_ZH.md)。

双机 sim2sim 使用一个共享 MuJoCo 世界替换两台 bridge 和三 Tracker 输入，
搬箱仍运行 `deploy_dual_scalebfm_residual.py` 的 ScaleBFM + residual。
仿真和实机共用 `InteractiveDualCoordinator`，包括两秒 DefaultPose 过渡及 Start/B/A/Stop 门控。
仿真两路端口为 `127.0.0.1:56001/56002` 和 `127.0.0.1:56101/56102`。

先完成第 1 节策略产物准备；`config/g1/dual_policy_artifacts` 中必须有 manifest、
模型、metadata、reference 和 FK 资源。LocoMode 使用现有
`config/g1/omnicontact/LocoMode.onnx`，A/B 各自维护独立的循环网络状态。

一条命令启动 GUI 仿真和部署进程：

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
bash scripts/run_dual_scalebfm_sim2sim.sh
```

操作顺序与单机一致，按键在 MuJoCo 窗口输入：

1. 等待部署终端打印 `ZERO TORQUE`。两台机器人已在原版 DefaultPose，物理暂不推进。
2. 按 `s`：两秒过渡到原版 DefaultPose，等待 `DefaultPose ready`，两个 floating base 和箱子保持锁定。
3. 确认姿态稳定后按 `b`：两路同一状态帧的 LocoMode 命令到齐，才同步释放 base 和箱子。
4. 等待 `LocoMode standing` 并确认两台机器人稳定，按 `a`：使用新鲜双机位姿做
   reference 对齐、几何预检，随后运行搬箱策略。两台头顶的红色圆柱消失。
5. reference 完成后自动回到两台独立的 LocoMode standing，继续站立，等待 `x`。
6. 按 `x`：对两台发送禁用阻尼命令并退出；一键脚本清理两个进程。终端 `Ctrl+C` 也可结束。

提前按 `a` 或 `b` 不会跳过前置阶段；不再有自动进入搬箱的行为。完成后的 `a` 不会
重放已执行完的 reference，需要重启场景。`--duration N` 可限制部署进程运行时间，
这个墙钟时间包含等待按键阶段；人工操作建议不设置。

分两个终端运行时，终端 1：

```bash
bash scripts/run_dual_scalebfm_simulator.sh
```

终端 2：

```bash
bash scripts/run_dual_scalebfm_residual.sh \
  --pose-source sim \
  --act-robot both \
  --confirm-actuation ENABLE_MOTORS \
  --no-visualization
```

无窗口运行：

```bash
bash scripts/run_dual_scalebfm_sim2sim.sh --headless
```

在启动命令的终端逐次输入 `s`、`b`、`a`、`x`，每次按回车；仍须在 `b` 后确认站立
稳定才输入 `a`。分终端 headless 运行时，在仿真器终端输入按键。没有交互终端时可通过
标准输入提供按键，EOF 不会自动使能或自动启动任务。

每个 50 Hz 状态帧携带同一原子 pose 和按键快照。重传不会改变该帧的按键，两个机器人
收到相同按钮值。两条 PD 命令都必须确认同一状态时间戳、控制阶段和 reference 帧；
阶段不一致会终止仿真，不会让一台先释放 base。每个 200 Hz 物理子步重新计算 PD 力矩。
箱子误差检查只在搬箱执行阶段进行，比较的是控制器实际发出的 reference 帧及对齐后
的箱子位置；等待 `s/b/a` 和结束后的 LocoMode 不消耗任务帧，也不触发搬箱误差检查。

初始化布局仍来自双机 reference bundle：A/B 的全局 XY、朝向及箱子位姿来自
`start_frame`；机器人关节改用原版 `DefaultPose.yaml`，pelvis 高度默认 `0.793 m`，
可通过 `simulation.initial_root_height_m` 调整。这与旧版直接加载 reference 关节角的
训练 reset 不同。启动搬箱后仍需验证该 checkpoint 对站立切换到任务 reference 的适应性。
当前没有 Vive 一次快照导入和 `--goal-position`；双机 JSON 中的 goal 不驱动此任务。

`--act-robot none` 仍发送 `enable=0` 命令，释放 base 后仿真只施加阻尼，不能用于
验证站立/搬箱效果。完全不连接 bridge 的数值检查应使用第 3 节离线 validator。

测试交互流程、两路同步和仿真控制：

```bash
PYTHONPATH=src uv run --with pytest python -m pytest -q \
  tests/test_dual_sim_control.py \
  tests/test_dual_scalebfm_sim2sim.py \
  tests/test_dual_scalebfm_deploy.py
```

仿真单元测试使用临时 reference 和真实 MuJoCo 场景，不依赖未提交的训练 checkpoint；
测试通过不代表正式模型完成搬运。历史自动启动版本已记录默认 `best_agent.pt` 在箱子
高度误差阈值上失败；该记录不能视为新交互流程的动力学验证。正式验收仍须补齐产物，
按上述流程完成站立、搬箱和结束后站立的整段测试。
