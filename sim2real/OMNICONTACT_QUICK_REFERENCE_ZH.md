# OmniContact 搬箱任务命令速查

适用仓库：`/home/yiranwang/TeleHuman/motion_tracking`。本文只整理命令与必要操作顺序；原理和坐标系定义见 `OMNICONTACT_ZH.md`。

命令中的 `<...>` 是占位符，运行前必须替换为实际序列号、IP、网卡或坐标；不要连尖括号一起复制执行。

> 安全提示：所有实机命令先不加 `--act`。只有机器人已经悬空、支撑或完成其他安全措施，并确认 Tracker、关节顺序、目标点和 Stop 行为后，才使用 `--act --confirm-actuation ENABLE_MOTORS`。

## 0. 路径、端口和按键

常用目录：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
```

| 端口 | 数据方向 | 用途 |
| --- | --- | --- |
| `55001/UDP` | bridge/sim2sim → deploy | 策略状态 |
| `55002/UDP` | deploy → bridge/sim2sim | 电机或仿真 PD 命令 |
| `55003/UDP` | G1 bridge → twin viewer | 只读实测关节镜像 |
| `55004/UDP` | deploy → twin viewer | 只读 reference/ghost/contact |
| `15150/UDP` | Vive publisher → deploy | 远程 Tracker pose pair |

按键：

| 环境 | 进入默认姿态 | 进入 LocoMode | 开始 OmniContact | 停止 |
| --- | --- | --- | --- | --- |
| sim2sim MuJoCo 窗口 | `s` | `b` | `a` | `x` |
| G1 遥控器 | `Start` | `B` | `A` | `Stop/Select` |

pelvis 滑条标定窗口中，`S` 表示保存 JSON；不按 `S` 退出则不保存。

## 1. 一次性安装和编译

Python 环境；直接连接 SteamVR 的机器使用 Vive extra：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
uv sync --extra vive
```

不连接 SteamVR 的策略主机可以只安装基础环境：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
uv sync
```

编译 G1 C++ bridge：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/g1_sim2real
bash scripts/build.sh
```

## 2. Vive 配置准备和备份

仅当本地配置尚不存在时，从示例创建；不会覆盖已有配置：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
test -f config/g1/omnicontact_vive.json || \
  cp config/g1/omnicontact_vive.example.json config/g1/omnicontact_vive.json
```

任何会写配置的标定前先备份：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
cp config/g1/omnicontact_vive.json \
  "config/g1/omnicontact_vive.json.backup.$(date +%Y%m%d-%H%M%S)"
```

检查 JSON 格式和当前变换：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
jq '.' config/g1/omnicontact_vive.json
jq '.world_from_steamvr, .robot_tracker_to_pelvis, .object_tracker_to_object' \
  config/g1/omnicontact_vive.json
```

## 3. Tracker 检查

SteamVR 必须已经启动，两个 Generic Tracker 必须在线。

列出硬件序列号：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
uv run --extra vive python scripts/list_vive_trackers.py --seconds 5
```

查看两个 Tracker 的原始 OpenVR 位姿；不应用任何标定：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
uv run --extra vive python scripts/view_raw_tracker_poses.py \
  --robot-serial <ROBOT_TRACKER_SERIAL> \
  --object-serial <OBJECT_TRACKER_SERIAL>
```

让脚本自动选取前两个 Tracker：

```bash
uv run --extra vive python scripts/view_raw_tracker_poses.py
```

不打开可视化窗口，实时列出当前所有 Tracker 的原始 XYZ 和序列号：

```bash
uv run --extra vive python scripts/identify_vive_trackers.py
```

每次只移动一台 Tracker，观察哪一行的 XYZ 和 `MOVED` 变化，即可确认该设备的固定
`LHR-...` 序列号。按 `Ctrl-C` 结束。

## 4. 标定流程

推荐顺序：Tracker 检查 → 世界系标定 → robot Tracker/pelvis 外参 → 箱子外参和尺寸 → goal → 综合预览 → 最后设置 `calibration_confirmed: true`。

### 4.1 三点标定 SteamVR 世界系

整个 O/+X/+Y 采集过程不要旋转 Tracker：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
uv run --extra vive python scripts/calibrate_vive_world.py \
  --serial <TRACKER_SERIAL> \
  --distance-x 0.5 \
  --distance-y 0.5
```

将采集位置 O 映射到非零任务世界坐标：

```bash
uv run --extra vive python scripts/calibrate_vive_world.py \
  --serial <TRACKER_SERIAL> \
  --distance-x 0.5 \
  --distance-y 0.5 \
  --origin-world <X> <Y> <Z>
```

该脚本只打印结果，不写文件。检查 `measured_x_distance_m`、`measured_y_distance_m` 和 `raw_xy_angle_deg` 后，手工把输出中的 `world_from_steamvr` 写入：

```text
/home/yiranwang/TeleHuman/motion_tracking/sim2real/config/g1/omnicontact_vive.json
```

### 4.2 箱子上表面中心单点快速标定

以下命令会覆盖输出 JSON 中的 `world_from_steamvr` 和 `object_tracker_to_object`；不会修改 pelvis 外参和 goal：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
uv run --extra vive python scripts/calibrate_vive_box_world.py \
  --vive-config config/g1/omnicontact_vive.json \
  --output config/g1/omnicontact_vive.json \
  --box-center-world 1.0 0.0 0.15 \
  --box-top-z 0.30
```

显式指定箱子 Tracker：

```bash
uv run --extra vive python scripts/calibrate_vive_box_world.py \
  --vive-config config/g1/omnicontact_vive.json \
  --output config/g1/omnicontact_vive.json \
  --serial <OBJECT_TRACKER_SERIAL> \
  --box-center-world <X> <Y> <Z> \
  --box-top-z <TOP_Z>
```

只计算和打印、不写 JSON：

```bash
uv run --extra vive python scripts/calibrate_vive_box_world.py \
  --vive-config config/g1/omnicontact_vive.json \
  --box-center-world 1.0 0.0 0.15 \
  --box-top-z 0.30
```

### 4.3 滑动条调整 robot Tracker 到 pelvis

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
uv run --extra vive python scripts/view_calibrated_omnicontact_poses.py \
  --vive-config config/g1/omnicontact_vive.json \
  --tune-robot-tracker-to-pelvis \
  --state-port 0 \
  --no-visualization
```

- XYZ 是米，RPY 是度；六个滑动条实时更新机器人和 pelvis 坐标轴。
- 在 MuJoCo 或滑动条窗口按 `S`，只原子覆盖 `robot_tracker_to_pelvis`。
- 不按 `S` 直接退出，不修改 JSON。
- 默认 XYZ 范围为 ±0.5 m；修改范围：

```bash
--tune-translation-range 1.0
```

### 4.4 标定综合预览

配置已确认时：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
uv run --extra vive python scripts/view_calibrated_omnicontact_poses.py \
  --vive-config config/g1/omnicontact_vive.json \
  --state-port 0 \
  --no-visualization
```

配置仍是 `calibration_confirmed: false` 时，只读预览：

```bash
uv run --extra vive python scripts/view_calibrated_omnicontact_poses.py \
  --vive-config config/g1/omnicontact_vive.json \
  --allow-unconfirmed \
  --state-port 0 \
  --no-visualization
```

检查无误后手工确认 JSON：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
${EDITOR:-nano} config/g1/omnicontact_vive.json
jq '.calibration_confirmed, .object_half_extents_m, .goal_position_w' \
  config/g1/omnicontact_vive.json
```

`calibration_confirmed` 必须是 JSON 布尔值 `true`，不是字符串 `"true"`。

## 5. sim2sim 仿真启动

不能同时启动真实 G1 bridge，因为二者都会使用 `55001/55002`。

终端 1，启动 MuJoCo：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
uv run src/sim2sim.py \
  --robot g1 \
  --bridge-config config/g1/bridge_omnicontact.yaml
```

窗口会标出任务世界、机器人 pelvis 和箱子三个局部坐标系；X 红、Y 绿、Z 蓝。

按实测场景启动（推荐用于把实机布局带入 sim2sim）：

```bash
uv run --extra vive python src/sim2sim.py \
  --robot g1 \
  --bridge-config config/g1/bridge_omnicontact.yaml \
  --initial-scene-source vive \
  --vive-config config/g1/omnicontact_vive.json
```

该模式在启动时读取一次新鲜的 Vive 快照，并用标定世界系中的完整
pelvis/箱子位姿初始化两个 MuJoCo free joint；同时把 JSON 的
`goal_position_w` 随仿真状态发送给控制器。快照完成后 OpenVR 会关闭，后续
pelvis 和箱子由 MuJoCo 物理独立演化，不会持续跟随或瞬移到真实 Tracker。
采样时保持机器人和箱子静止，并确认窗口中的三个坐标轴和实物一致。

对应的终端 2 不需要再单独填写目标点：

```bash
uv run src/deploy_omnicontact.py \
  --robot g1 \
  --pose-source sim \
  --max-target-delta 1.0 \
  --act \
  --confirm-actuation ENABLE_MOTORS
```

目标优先级为：显式 `--goal-position` 覆盖值 → sim2sim 发布的 Vive JSON
目标 → 兼容默认值 `[1.0, 1.0, 0.15]`。Vive JSON 的
`object_half_extents_m` 必须与 MuJoCo 箱子尺寸一致，否则启动会拒绝该场景。
该模式只导入机器人全局 pelvis 位姿，不导入实机关节角；仿真关节仍从
OmniContact DefaultPose 开始。

无窗口运行：

```bash
uv run src/sim2sim.py \
  --robot g1 \
  --bridge-config config/g1/bridge_omnicontact.yaml \
  --headless
```

终端 2，启动仿真策略闭环：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
uv run src/deploy_omnicontact.py \
  --robot g1 \
  --pose-source sim \
  --goal-position 1.0 1.0 0.15 \
  --max-target-delta 1.0 \
  --act \
  --confirm-actuation ENABLE_MOTORS
```

这里的 `--act` 只控制 MuJoCo，不连接实机。机器人以原版 DefaultPose 在任务对应地面位置初始化；按 `s` 后进入并保持 DefaultPose，MuJoCo floating base 继续锁定；按 `b` 后，仿真器在收到首个 LocoMode 命令时同步释放 base。头顶红色圆柱持续表示尚未进入 A 键后的策略。操作顺序：等待 `ZERO TORQUE` → MuJoCo 窗口按 `s`（锁定 base 并保持 DefaultPose）→ 确认默认姿态稳定后按 `b`（同步释放 base 并进入 LocoMode）→ 等待 `LocoMode standing` 并确认机器人稳定 → 按 `a`（红色圆柱消失）启动 CFTrack → CFGen 结束后自动回到 LocoMode → 按 `x` 停止。实机流程仍使用配置的 2 秒 DefaultPose 过渡。

只运行策略评估、不发送仿真命令：

```bash
uv run src/deploy_omnicontact.py \
  --robot g1 \
  --pose-source sim \
  --goal-position 1.0 1.0 0.15 \
  --run-seconds 30
```

## 6. sim2real 实机孪生：标准三终端流程

### 6.1 终端 1：G1 bridge

先查看网卡名称：

```bash
ip -br link
ip -br addr
```

启动 bridge；当前机器示例接口是 `enx6c1ff770ecdd`：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/g1_sim2real
G1_NET=enx6c1ff770ecdd bash scripts/run_bridge.sh
```

通用写法：

```bash
G1_NET=<WIRED_INTERFACE> bash scripts/run_bridge.sh
```

### 6.2 终端 2：只读 OmniContact 孪生窗口

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
uv run --extra vive python scripts/real_omnicontact_viewer.py \
  --vive-config config/g1/omnicontact_vive.json
```

等价的完整端口写法：

```bash
uv run --extra vive python scripts/real_omnicontact_viewer.py \
  --vive-config config/g1/omnicontact_vive.json \
  --state-host 127.0.0.1 \
  --state-port 55003 \
  --visualization-host 127.0.0.1 \
  --visualization-port 55004
```

仅看 Tracker/pelvis/箱子，不接 bridge 和策略可视化：

```bash
uv run --extra vive python scripts/real_omnicontact_viewer.py \
  --vive-config config/g1/omnicontact_vive.json \
  --state-port 0 \
  --no-visualization
```

只诊断 bridge 的实测关节和 IMU，不读取 Tracker：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
uv run src/real_state_viewer.py --robot g1
```

### 6.3 终端 3：先运行无电机输出观察模式

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
uv run --extra vive python src/deploy_omnicontact.py \
  --vive-config config/g1/omnicontact_vive.json \
  --pose-source local \
  --run-seconds 30
```

该模式运行策略并把 reference/ghost 发到 `55004`，但不向 bridge 发送电机命令。检查日志中的 `Pose sanity`，并在孪生窗口确认实机关节、pelvis、箱子、ghost 和 reference 对齐。

### 6.4 确认安全后启用实机电机

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
uv run --extra vive python src/deploy_omnicontact.py \
  --vive-config config/g1/omnicontact_vive.json \
  --pose-source local \
  --act \
  --confirm-actuation ENABLE_MOTORS
```

uv run src/deploy_omnicontact.py \
  --robot g1 \
  --pose-source sim \
  --goal-position 1.0 1.0 0.15 \
  --max-target-delta 1.0 \
  --act \
  --confirm-actuation ENABLE_MOTORS

操作顺序：等待 `ZERO TORQUE` → 遥控器按 `Start` → 等待默认姿态完成 → 按 `B` 进入 LocoMode → 确认场地、机器人和箱子安全 → 按 `A` 进入 OmniContact → `Stop/Select` 停止。

### 6.5 自动诊断日志

按上述标准命令启动时，G1 bridge 和 `deploy_omnicontact.py` 都会自动将终端信息写入同一个目录：

```text
<仓库根目录>/logs/omnicontact/
```

文件名分别以 `bridge_` 和 `deploy_` 开头，并包含启动时间和进程 PID。启动时两个终端都会打印本次使用的完整日志路径。Python 日志额外记录：A 被接受时的完整位姿/新鲜度、Tracker 有效/无效更新累计数、CFGen 规划耗时、首帧策略耗时、控制命令最大间隔、异常堆栈，以及每一次 damping 的触发位置。bridge 日志保留 watchdog 锁存、命令频率、拒绝命令数和 DDS 输出信息。

Python 还会自动生成同名的 `deploy_*.observations.npz` 压缩轨迹。v2 格式从 Zero Torque 开始按每个 bridge 控制状态记录完整启动过程，包括 DefaultPose、等待 B、LocoMode、等待 A、CFGen planning、CFTrackPolicy、结束和异常阶段。它保存两个 Tracker 在 SteamVR 原始坐标系及标定 world 坐标系中的实际变换、由其生成的 pelvis/object 位姿、物体线/角速度、位姿年龄与有效性、Tracker 有效/无效更新累计数、LowState 实测的 29 维 `q_lab/dq_lab`（不是关节目标）、IMU、实际送入 ONNX 的 1244 维 observation、策略内部 5×141 历史、29 维策略 action、限幅前/后的目标、kp/kd、策略耗时和命令间隔。bridge 超时时还会写入 `bridge_state_timeout` 终止行；Tracker 遮挡则写入 `pose_pair_invalid_hold` 行。因此下次可直接区分 Tracker 遮挡、观测突变、策略输出异常、关节限幅和 bridge/DDS 断流。

#### 在 MuJoCo 中回放一次实机启动

使用对应的结构化日志，不需要启动 SteamVR、G1 bridge 或 policy：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
uv run --extra vive python scripts/real_omnicontact_viewer.py \
  --vive-config config/g1/omnicontact_vive.json \
  --replay-log ../logs/omnicontact/deploy_YYYYMMDD_HHMMSS_pidXXXX.observations.npz
```

`--replay-log` 也可以直接传同名 `deploy_*.log`，viewer 会自动查找旁边的 `deploy_*.observations.npz`。纯文本日志本身不含连续状态，若 companion 不存在会直接报出缺失路径。

窗口按日志的原始单调时钟间隔同步回放：蓝/橙金字塔是两个 Tracker，pelvis 根位姿和 G1 关节使用当帧实测反馈，箱子使用当帧 runtime pose/velocity。终端进度条同时显示 `当前帧/总帧`、日志相对时间、原始墙钟时间、`state_packet_seq`、`policy_frame`、`event`、`task_state` 和 pose 有效性，可直接和同名 `deploy_*.log` 的时间及事件对照。

回放按键：

- `Space` 或 `P`：暂停/继续；
- `N`：暂停并前进一帧；
- `B`：暂停并后退一帧；
- `A`：降低一档播放速度；
- `D`：提高一档播放速度；
- `R`：从第 0 帧重新播放。

播放速度档位固定为 `0.25x`、`0.5x`、`1x`、`5x` 和 `10x`。切换速度不会暂停，也不会改变当前回放位置；到达最低或最高档后继续按键会保持当前档位。

常用选项：

```bash
# 半速、从第 1200 帧开始并先暂停
uv run --extra vive python scripts/real_omnicontact_viewer.py \
  --vive-config config/g1/omnicontact_vive.json \
  --replay-log ../logs/omnicontact/deploy_CASE.observations.npz \
  --replay-speed 0.5 \
  --replay-start-frame 1200 \
  --replay-paused

# 循环回放
uv run --extra vive python scripts/real_omnicontact_viewer.py \
  --replay-log ../logs/omnicontact/deploy_CASE.observations.npz \
  --replay-loop
```

v1 旧轨迹仍可回放已有的 pelvis、箱子和实测关节，但旧格式没有保存两个 Tracker 本体，也没有覆盖启动前段；viewer 会用 pelvis/箱子和所选 Vive 标定反推 Tracker，并在终端显示警告。只有新生成的 v2 日志能对两个 Tracker 和完整启动过程进行原样回放。

快速查看文件结构和故障末尾：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
uv run python - <<'PY'
import json
import numpy as np

path = "../logs/omnicontact/deploy_XXXX.observations.npz"
with np.load(path, allow_pickle=False) as data:
    print(json.loads(str(data["metadata_json"])))
    print(data.files)
    print("events:", data["event"][-10:])
    print("pose valid:", data["pose_pair_valid"][-10:])
    print("provider invalid:", data["provider_invalid_count"][-10:])
    print("observation shape:", data["observation"].shape)
PY
```

发生倒地后不要覆盖或编辑日志。直接查看最近文件：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking
ls -lht logs/omnicontact | head -20
rg -n "A accepted|Carry reference ready|CONTROL COMMAND GAP|FATAL|FAILSAFE|WATCHDOG|damping|watchdog_latched" \
  logs/omnicontact
```

之后只需说明“分析最近一次 OmniContact 日志”，即可从该目录配对检查最近的 `deploy_*.log` 和 `bridge_*.log`。

可选地指定 Python 日志位置：

```bash
uv run --extra vive python src/deploy_omnicontact.py \
  --vive-config config/g1/omnicontact_vive.json \
  --pose-source local \
  --log-file /absolute/path/deploy_case.log
```

可选地指定 bridge 日志位置：

```bash
G1_NET=<WIRED_INTERFACE> \
G1_BRIDGE_LOG_FILE=/absolute/path/bridge_case.log \
  bash scripts/run_bridge.sh
```

可用 `--history-file /absolute/path/case.npz` 单独指定观测轨迹路径，或用 `--no-history-file` 只关闭观测轨迹。`--no-file-log` 关闭文本日志；bridge 可设置 `G1_BRIDGE_NO_FILE_LOG=1`。实机故障复现不建议关闭任何日志。

## 7. 远程 Vive UDP 模式

两台机器设置相同的长随机 token：

```bash
export OMNICONTACT_POSE_TOKEN='<LONG_RANDOM_TOKEN>'
```

策略/G1 主机，先无电机输出验证：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
uv run python src/deploy_omnicontact.py \
  --vive-config config/g1/omnicontact_vive.json \
  --pose-source udp \
  --udp-bind 0.0.0.0 \
  --udp-port 15150 \
  --allowed-sender-ip <STEAMVR_WORKSTATION_IP> \
  --run-seconds 30
```

SteamVR 工作站：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
uv run --extra vive python scripts/publish_vive_poses.py \
  --vive-config config/g1/omnicontact_vive.json \
  --target-ip <POLICY_HOST_IP> \
  --port 15150 \
  --bind-ip <STEAMVR_WORKSTATION_IP>
```

验证后，策略/G1 主机使用完整实机启用命令：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
uv run python src/deploy_omnicontact.py \
  --vive-config config/g1/omnicontact_vive.json \
  --pose-source udp \
  --udp-bind 0.0.0.0 \
  --udp-port 15150 \
  --allowed-sender-ip <STEAMVR_WORKSTATION_IP> \
  --act \
  --confirm-actuation ENABLE_MOTORS
```

## 8. 孪生窗口跨机器运行

如果 viewer 不在 G1 bridge 所在机器：

1. 将 `g1_sim2real/config/g1_bridge.yaml` 的 `state_mirror_host` 改为 viewer IP。
2. deploy 增加 `--visualization-host <VIEWER_IP>`。
3. viewer 绑定所有本地接口：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
uv run --extra vive python scripts/real_omnicontact_viewer.py \
  --vive-config config/g1/omnicontact_vive.json \
  --state-host 0.0.0.0 \
  --state-port 55003 \
  --visualization-host 0.0.0.0 \
  --visualization-port 55004
```

策略主机示例：

```bash
uv run --extra vive python src/deploy_omnicontact.py \
  --vive-config config/g1/omnicontact_vive.json \
  --pose-source local \
  --visualization-host <VIEWER_IP> \
  --run-seconds 30
```

`G1_NET` 只选择 Unitree DDS 网卡，不负责 viewer UDP 路由。

## 9. 停止、重启和端口检查

- 正常停止：MuJoCo 用 `x`，实机用遥控器 `Stop/Select`，进程最终用 `Ctrl+C`。
- G1 bridge 第一次收到有效命令后若触发 200 ms watchdog，会进入锁存阻尼；必须停止并重新运行 `run_bridge.sh`。
- 不要同时运行 sim2sim 和真实 bridge。

查看端口占用：

```bash
ss -lunp | rg ':(55001|55002|55003|55004|15150)\b'
```

## 10. 测试和帮助

完整测试：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
PYTHONPATH=src uv run python -m unittest discover -s tests -v
```

语法检查：

```bash
uv run python -m py_compile \
  src/deploy_omnicontact.py \
  scripts/view_calibrated_omnicontact_poses.py \
  scripts/real_omnicontact_viewer.py
```

查看每个入口的最新参数：

```bash
uv run src/sim2sim.py --help
uv run python src/deploy_omnicontact.py --help
uv run --extra vive python scripts/view_calibrated_omnicontact_poses.py --help
uv run --extra vive python scripts/publish_vive_poses.py --help
```
