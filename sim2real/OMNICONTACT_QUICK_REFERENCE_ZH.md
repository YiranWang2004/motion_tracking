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

| 环境 | 开始准备 | 开始任务 | 停止 |
| --- | --- | --- | --- |
| sim2sim MuJoCo 窗口 | `s` | `a` | `x` |
| G1 遥控器 | `Start` | `A` | `Stop/Select` |
| pelvis 滑条标定 | — | `S` 保存 JSON | 不按 `S` 退出则不保存 |

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

这里的 `--act` 只控制 MuJoCo，不连接实机。机器人以原版 DefaultPose 在任务对应地面位置初始化；按 `s` 后立即解除躯干根部锁定并进入 MuJoCo 物理闭环，一个 50 Hz 控制周期后由 LocoMode 接管，不存在悬空阶段。头顶红色圆柱持续表示尚未进入 A 键后的策略。操作顺序：等待 `ZERO TORQUE` → MuJoCo 窗口按 `s`（立即释放根部）→ 等待 `LocoMode standing` 并确认机器人稳定 → 按 `a`（红色圆柱消失）启动 CFTrack → CFGen 结束后自动回到 LocoMode → 按 `x` 停止。实机流程仍使用配置的 2 秒 DefaultPose 过渡。

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

操作顺序：等待 `ZERO TORQUE` → 遥控器按 `Start` → 等待默认姿态完成 → 确认场地和箱子安全 → 按 `A` → `Stop/Select` 停止。

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
