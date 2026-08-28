# Dual G1 ScaleBFM Residual 实机部署

本入口沿用已经通过 A-only、B-only、both 实机测试的双 namespace 和双
`g1_udp_bridge`。正式策略进程只通过两组 UDP endpoint 向 bridge 发送 29 维绝对 PD
目标，不直接加入 Unitree DDS 网络。

## 1. 准备部署产物

下面的命令会复制 ScaleBFM、MAPPO Actor、CFGen/Kimodo reference 和训练使用的 FK
XML，并生成 SHA256 manifest。每次更换 checkpoint 或 reference 都必须重新执行。

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
uv run --extra dual-policy python scripts/prepare_dual_scalebfm_artifacts.py \
  --reference-bundle /home/yiranwang/TeleHuman/Dual_G1_MJ/results/cfgen_box_3m_kimodo.npz
```

当前默认 checkpoint 是：

```text
Dual_G1_MJ/dual_g1_scalebfm_residual_object/
  26-08-20_01-42-05-292432_MAPPO/checkpoints/best_agent.pt
```

reference 必须是带 `training_*` 字段的双机 bundle，并且必须与要执行的箱子尺寸和
初始 A/B/箱子几何一致。入口会在发出电机指令前检查这些条件。

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

先在 `config/g1/dual_network.yaml` 中写入固定的 A/B 物理网口和对应 MAC。该文件同时保存 namespace、
物理口 IP、veth 和 bridge 配置路径。之后不需要在命令行重复输入网口名称。

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

bridge A/B 启动后，在独立终端启动实时 viewer：

```bash
bash scripts/run_dual_scalebfm_viewer.sh
```

viewer 默认接收三路只读 UDP：

- `55003`：G1 A bridge state mirror，只读取 A 的 29 维实测关节角。
- `55103`：G1 B bridge state mirror，只读取 B 的 29 维实测关节角。
- `55204`：policy visualization，包含三 Tracker 标定后的 A/B pelvis 与箱子姿态、CFGen
  reference、ScaleBFM target、residual 和最终 PD target。

实心蓝色和琥珀色模型分别是 A/B 实机状态；半透明绿色和洋红色模型是 A/B reference；
实心棕色箱子是 Tracker 实测状态，半透明绿色箱子是 CFGen reference。三个带 XYZ 轴的
球形标记表示 Tracker 标定后实际用于 policy 的 A pelvis、B pelvis 和箱子坐标，不是原始
SteamVR 坐标。

按 `G` 在“CFGen reference ghost”和“最终发送给 bridge 的限幅 PD target ghost”之间切换。
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

第一阶段只运行全部感知、FK、ScaleBFM 和 Actor，不使能电机：

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

前 2 秒是 ScaleBFM default pose 过渡，之后才从配置的 `start_frame` 执行 reference。
任一 bridge 状态超时、双机 state skew 超限、任一 Tracker 丢失、几何预检失败、模型出现
非有限值或连续三帧推理超时，都会对两侧发送禁用 hold 并退出。

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
`R` 从头播放，`G` 切换 reference/最终 PD target ghost。

实时终端每秒打印 A、B、visual 三路数据年龄；超过默认 `500 ms` 会显示 `!`。这是 viewer
告警，不代替 policy 中更严格的 `200 ms` bridge state 和 `100 ms` Tracker fail-closed 检查。

## 8. 与实机控制链严格对齐的 sim2sim

sim2sim 不运行另一份简化策略。它只用一个共享 MuJoCo 世界替换两台
`g1_udp_bridge` 和三 Tracker Vive 输入；`deploy_dual_scalebfm_residual.py`、
ScaleBFM、residual actor、限位、限速、力矩限制和双机 coordinator 与实机
完全共用。仿真端口固定为 `127.0.0.1:56001/56002` 和
`127.0.0.1:56101/56102`，不会向两个实机网口发送数据。

一条命令启动 GUI 仿真和正式部署进程：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
bash scripts/run_dual_scalebfm_sim2sim.sh --duration 12
```

无窗口运行：

```bash
bash scripts/run_dual_scalebfm_sim2sim.sh --headless --duration 12
```

也可分两个终端调试。先启动 MuJoCo bridge：

```bash
bash scripts/run_dual_scalebfm_simulator.sh
```

再启动正式 deploy，唯一变化是位姿和 bridge 选择为仿真：

```bash
bash scripts/run_dual_scalebfm_residual.sh \
  --pose-source sim \
  --act-robot both \
  --confirm-actuation ENABLE_MOTORS \
  --no-visualization \
  --duration 12
```

每个 50 Hz 状态帧带同一个 `state_receive_time_ns` 和原子
`dual_sim_pose`。MuJoCo 必须收到 A/B 对应该时间戳的两条命令后，才执行
4 个 200 Hz 物理步。`enable=0` 忽略位置目标并仅施加阻尼，`enable=1`
才应用收到的 `q_des/qd_des/kp/kd`，随后按实机策略元数据裁剪力矩。
