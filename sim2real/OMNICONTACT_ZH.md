# OmniContact 搬箱任务：sim2sim 与 sim2real

命令速查：[OMNICONTACT_QUICK_REFERENCE_ZH.md](OMNICONTACT_QUICK_REFERENCE_ZH.md)

本文说明当前仓库中 OmniContact `carrybox` 任务的安装、仿真、实机运行、Vive 标定和故障安全行为。

本文档对应当前仓库结构。OmniContact 的 MuJoCo 搬箱场景已经随仓库提交，位于 `config/g1/assets/omnicontact_carry_box.xml`，不需要再准备外部 `OmniContact_sim2sim` 仓库。

## 1. 运行架构

OmniContact 集成只实现 `carrybox` 技能，不修改普通 PMG 入口 `src/deploy.py`。高层控制器统一使用 `src/deploy_omnicontact.py`，仿真和实机只替换低层桥接端：

| 模式 | 高层控制器 | 低层桥接 | 位姿来源 |
| --- | --- | --- | --- |
| sim2sim | `deploy_omnicontact.py` | `sim2sim.py` + MuJoCo | MuJoCo 状态包 |
| sim2real 本地 Vive | `deploy_omnicontact.py` | `g1_udp_bridge` + Unitree DDS | 本机 OpenVR |
| sim2real 远程 Vive | `deploy_omnicontact.py` | `g1_udp_bridge` + Unitree DDS | SteamVR 工作站经 UDP |

控制器和桥接器之间通过 UDP 交换状态和命令。策略接口固定为 50 Hz、29 个 G1 关节，策略模型位于 `config/g1/omnicontact/policy.onnx`。

策略每周期执行以下步骤：

1. 读取 29 DoF 关节位置、速度和 IMU 角速度。
2. 读取机器人骨盆和箱子的世界坐标位姿。
3. 使用当前骨盆、箱子、箱子尺寸和目标位置生成 carry-box 参考轨迹。
4. 构造 1244 维 OmniContact 观测并运行 ONNX 推理。
5. 将动作转换成绝对关节目标，应用关节限位和目标变化限幅。
6. 通过 UDP 发送 PD 命令给 MuJoCo 桥接器或 G1 C++ 桥接器。

默认控制配置见：

- `config/g1/omnicontact_carrybox.yaml`
- `config/g1/controller.yaml`
- `config/g1/bridge_omnicontact.yaml`

默认策略控制频率为 50 Hz，默认实机每周期目标变化限幅为 `0.15 rad`，默认默认姿态过渡时间为 2 秒。

## 2. 环境准备

项目要求 Python `>=3.10,<3.11`，建议使用 `uv` 管理环境。

```bash
cd <repo>/motion_tracking/sim2real
uv sync
```

其中 `<repo>` 是包含 `motion_tracking/` 的仓库根目录。当前仓库的绝对路径示例为：

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
```

如果本机需要直接读取 SteamVR/OpenVR，再安装 Vive 可选依赖：

```bash
uv sync --extra vive
```

SteamVR 工作站必须安装并启动 SteamVR，两个 Generic Tracker 必须已经配对并且能被 OpenVR 发现。

## 3. sim2sim 运行步骤

### 3.1 仿真资源

OmniContact 专用桥接配置为：

```text
config/g1/bridge_omnicontact.yaml
```

它使用仓库内置的场景：

```text
config/g1/assets/omnicontact_carry_box.xml
```

sim2sim 窗口默认显示任务世界、G1 pelvis 和箱子三个坐标系的正方向轴：
X 红、Y 绿、Z 蓝。pelvis 和箱子坐标轴直接挂在对应刚体上，会随仿真运动。

该 XML 所需的机器人、ghost 和 mesh 文件也都在 `config/g1/assets/` 下。当前版本不需要 sibling `OmniContact_sim2sim` checkout。

### 3.2 启动 MuJoCo 桥接器

终端一：

```bash
cd <repo>/motion_tracking/sim2real
uv run src/sim2sim.py \
  --robot g1 \
  --bridge-config config/g1/bridge_omnicontact.yaml
```

无图形窗口时可以加：

```bash
--headless
```

当前 OmniContact 仿真配置为：

```yaml
freq:
  physical_hz: 200
  state_decimation: 4
lockstep_policy: true
```

含义是：

- MuJoCo 物理步进为 200 Hz；
- 每 4 个物理周期发布一次状态，因此策略接口为 50 Hz；
- `lockstep_policy` 让仿真等待对应的策略命令后再推进，减少 Python 调度抖动；
- 仍然经过 UDP 状态/命令路径和安全检查，不会绕过部署逻辑。

仿真桥使用 UDP 端口：

```text
状态：55001
命令：55002
```

不要同时运行真实 G1 C++ 桥接器占用这两个端口。

### 3.3 启动 OmniContact 控制器

终端二：

```bash
cd <repo>/motion_tracking/sim2real
uv run src/deploy_omnicontact.py \
  --robot g1 \
  --pose-source sim \
  --goal-position 1.0 1.0 0.15 \
  --max-target-delta 1.0 \
  --act \
  --confirm-actuation ENABLE_MOTORS
```

参数说明：

- `--pose-source sim`：从仿真状态包读取机器人和箱子位姿，不使用 Vive；
- `--goal-position X Y Z`：目标箱子中心，使用 MuJoCo 世界坐标；
- `--act`：允许发送控制命令；
- `--confirm-actuation ENABLE_MOTORS`：电机输出的强制确认字符串；
- `--max-target-delta 1.0`：仿真中用于复现原始 OmniContact 闭环的限幅设置。

仿真配置中的箱子初始中心是 `[1.0, 0.0, 0.15]`。`--goal-position` 只改变目标，不改变箱子初始位置。

### 3.4 仿真按钮和状态顺序

启动两个终端后按以下顺序操作：

1. 等待控制器打印 `ZERO TORQUE`。
2. 让 MuJoCo 窗口保持焦点，按 `s`，进入默认姿态过渡。
3. 等待控制器提示 `LocoMode standing` 已启动；此时由原版零速度 LocoMode
   持续站立，操作员可先释放机器人。
4. 在 MuJoCo 窗口按 `a`，生成 carry-box 参考并开始策略控制。
5. 搬运过程中可以看到下蹲、抓取和放置动作，这是预期行为。
6. CFGen 轨迹结束后会自动回到 LocoMode，并持续站立。
7. 按 `x` 请求停止，控制器发送阻尼命令；需要完全退出时再按 `Ctrl+C`。

仿真按键映射为：

```text
s -> start
a -> A
x -> stop
```

### 3.5 仿真目标限幅说明

真实部署默认 `max_target_delta=0.15 rad`，而原始 OmniContact 轨迹在单个策略周期内可能出现接近 `0.96 rad` 的合法目标变化。因此 sim2sim 示例使用 `--max-target-delta 1.0`。

如果希望测试与实机相同的严格限幅，可以省略该参数，但这会改变训练时的闭环行为，可能使仿真搬箱失败。无论是否覆盖该参数，关节位置限位和 MuJoCo 扭矩限位始终有效。

### 3.6 只评估、不输出电机命令

省略 `--act` 即可进入 no-actuation 模式：控制器会读取状态、初始化参考、运行策略并记录统计，但不会发送任何桥接命令。

```bash
uv run src/deploy_omnicontact.py \
  --robot g1 \
  --pose-source sim \
  --goal-position 1.0 1.0 0.15 \
  --run-seconds 30
```

## 4. sim2real 运行步骤

实机运行由两个进程组成：

1. `g1_sim2real/g1_udp_bridge`：读取 Unitree DDS 的 `rt/lowstate`，接收 UDP PD 命令并发布 `rt/lowcmd`。
2. `sim2real/src/deploy_omnicontact.py`：运行 OmniContact 策略和安全状态机。

### 4.1 编译 G1 C++ 桥接器

```bash
cd <repo>/motion_tracking/g1_sim2real
bash scripts/build.sh
```

### 4.2 配置 G1 网络

推荐使用外部工作站通过有线网连接 G1，并在工作站同时运行 C++ 桥接器和 Python 控制器。启动桥接器时将 `G1_NET` 设置为实际的有线接口：

```bash
cd <repo>/motion_tracking/g1_sim2real
G1_NET=<WIRED_INTERFACE> bash scripts/run_bridge.sh
```

例如：

```bash
G1_NET=enp3s0 bash scripts/run_bridge.sh
```

如果在 G1 机载电脑上运行，通常使用：

```bash
G1_NET=eth0 bash scripts/run_bridge.sh
```

当前 G1 桥接配置的关键值为：

```yaml
low_level:
  mode_pr: 0
  command_timeout_s: 0.20
```

命令 watchdog 为 200 ms。第一次收到有效命令后，如果超过 200 ms 没有新命令，桥接器会发送阻尼并锁存；必须重启桥接器才能重新激活。

### 4.3 本地 OpenVR 模式

本地模式要求运行 Python 控制器的机器同时连接 SteamVR Tracker。

先复制示例配置：

```bash
cd <repo>/motion_tracking/sim2real
cp config/g1/omnicontact_vive.example.json config/g1/omnicontact_vive.json
```

完成 Vive 标定和手工测量后，先运行观察模式：

```bash
uv run --extra vive python src/deploy_omnicontact.py \
  --vive-config config/g1/omnicontact_vive.json \
  --pose-source local \
  --run-seconds 30
```

检查日志中的：

```text
Pose sanity: pelvis=... object=... goal=...
```

确认坐标、姿态、箱子尺寸和目标点正确后，再在机器人已经悬空、支撑或采取其他安全措施的条件下启用电机：

```bash
uv run --extra vive python src/deploy_omnicontact.py \
  --vive-config config/g1/omnicontact_vive.json \
  --pose-source local \
  --act \
  --confirm-actuation ENABLE_MOTORS
```

实机操作顺序：

1. 等待控制器连接 G1 bridge；
2. 等待 `ZERO TORQUE`；
3. 按 G1 遥控器 `Start`；
4. 等待机器人移动到默认姿态；
5. 确认机器人和箱子安全；
6. 按 G1 遥控器 `A`；
7. 搬运过程中按 Stop/Select 停止并进入阻尼。

### 4.4 远程 Vive UDP 模式

该模式将 SteamVR 读取和策略控制分到两台机器：

- SteamVR 工作站：读取两个 Tracker，完成坐标变换并发送 UDP；
- 策略主机：接收校准后的世界坐标位姿，运行 Python 控制器。

两台机器必须使用同一个随机 token：

```bash
export OMNICONTACT_POSE_TOKEN='<LONG_RANDOM_TOKEN>'
```

策略主机：

```bash
cd <repo>/motion_tracking/sim2real
uv run python src/deploy_omnicontact.py \
  --vive-config config/g1/omnicontact_vive.json \
  --pose-source udp \
  --udp-bind 0.0.0.0 \
  --allowed-sender-ip <WORKSTATION_WIRED_IP> \
  --run-seconds 30
```

SteamVR 工作站：

```bash
cd <repo>/motion_tracking/sim2real
uv run --extra vive python scripts/publish_vive_poses.py \
  --vive-config config/g1/omnicontact_vive.json \
  --target-ip <POLICY_HOST_WIRED_IP> \
  --bind-ip <WORKSTATION_WIRED_IP>
```

默认位姿 UDP 端口为 `15150`。UDP 包使用 HMAC-SHA256 校验，接收端按本地到包时间判断新鲜度，并且要求序号递增。机器人 Tracker 和箱子 Tracker 的位姿会以一个原子 pair 发送，避免两者来自不同帧。

## 5. 实机世界坐标系的确定

### 5.1 世界系不是 G1 自动估计的 odometry 系

实机 OmniContact 不使用 G1 IMU 或腿式里程计自动决定任务世界原点。G1 bridge 提供关节、关节速度和 IMU 角速度；机器人骨盆位姿和箱子位姿由 Vive Tracker 提供。

因此，实机世界坐标系是通过人工标定建立的任务坐标系：

- 世界原点由操作者选择；
- 世界 `+X` 和 `+Y` 由操作者选择；
- `+Z` 由右手系自动确定；
- 箱子目标点必须使用同一个世界坐标系。

### 5.2 两个 Tracker 的安装要求

需要两个 Generic Tracker，并保证它们位于同一个 SteamVR Standing Space：

- robot Tracker 刚性固定在 G1 pelvis；
- object Tracker 刚性固定在箱子上。

配置中的刚体变换约定是：

```text
world_from_steamvr        = ^W T_S
robot_tracker_to_pelvis   = ^T_robot T_pelvis
object_tracker_to_object  = ^T_object T_object
```

示例配置中的 `robot_tracker_to_pelvis` 现在放入了一个基于 G1 pelvis 网格尺寸和你描述的安装方向得到的粗略估计：Tracker 位于 pelvis 后表面中央，pelvis 原点相对 Tracker 沿 Tracker `+Z`（机器人前方）约 7 cm，四元数为 xyzw `[0.5, -0.5, 0.5, 0.5]`。该值只能作为启动前的占位，不能替代实际测量；`object_tracker_to_object` 仍需按箱子 Tracker 的实际安装位置和方向精调。

这个姿态的轴映射是：

```text
pelvis +X（机器人前方） -> Tracker +Z
pelvis +Y（机器人左方） -> Tracker -X
pelvis +Z（世界上方）   -> Tracker -Y
```

### 5.3 列出 Tracker

在连接 SteamVR 的机器上运行：

```bash
cd <repo>/motion_tracking/sim2real
uv run --extra vive python scripts/list_vive_trackers.py --seconds 5
```

记录 robot Tracker 和 object Tracker 的硬件序列号，写入 `omnicontact_vive.json`：

```json
{
  "robot_tracker_serial": "...",
  "object_tracker_serial": "..."
}
```

### 5.4 标定 SteamVR 到任务世界的变换

执行：

```bash
uv run --extra vive python scripts/calibrate_vive_world.py \
  --serial <TRACKER_SERIAL> \
  --distance-x 0.5 \
  --distance-y 0.5
```

标定时只使用一个 Tracker，并且整个过程中不要旋转它。程序会要求采集三个位置：

1. `O`：把 Tracker 放到想要定义为世界原点的位置；
2. `+X`：从 `O` 沿世界正 X 方向平移约 `distance-x`；
3. `+Y`：回到 `O`，再沿世界正 Y 方向平移约 `distance-y`。

默认 `--origin-world` 是 `[0, 0, 0]`。如果希望把 `O` 映射到其他世界坐标，可以显式指定：

```bash
--origin-world X Y Z
```

程序根据三个采样点求解 `^W T_S`：

```text
delta_x = positive_x_s - origin_s
delta_y = positive_y_s - origin_s

x_axis_in_s = normalize(delta_x)
y_axis_in_s = normalize(
    delta_y - dot(delta_y, x_axis_in_s) * x_axis_in_s
)
z_axis_in_s = cross(x_axis_in_s, y_axis_in_s)

R_W_from_S = [x_axis_in_s, y_axis_in_s, z_axis_in_s]^T
p_W_from_S = origin_world - R_W_from_S * origin_s
```

这保证：

```text
SteamVR 中的 O   -> 世界 origin_world
SteamVR 中的 +X  -> 世界 +X
SteamVR 中的 +Y  -> 世界 +Y
世界 +Z           -> +X cross +Y
```

标定输出中的 `measured_x_distance_m`、`measured_y_distance_m` 应接近实际尺量距离，`raw_xy_angle_deg` 应接近 90 度。

### 5.5 箱子放置时的单点快速标定

如果每次放置箱子时都能把一个 Tracker 放在箱子上表面中心，可以使用仓库提供的单点快速标定脚本，不必执行 O/+X/+Y 三点标定。

约定如下：

- Tracker 原点放在箱子上表面中心；
- 箱子中心的世界坐标由操作者给出；
- 箱子上表面默认世界高度为 `0.30 m`；
- 箱子中心默认高度为 `0.15 m`；
- Tracker `+X` 与箱子一条水平棱同向；
- Tracker `+Y` 与另一条水平棱平行；
- Tracker `+Z` 朝下；
- 为保持右手系，当前占位外参使用 Tracker 到箱体中心的平移 `[0, 0, 0.15]`，姿态使用 xyzw 四元数 `[1, 0, 0, 0]`（绕 X 轴 180 度）。

这里的 `object_tracker_to_object` 是有意保留的粗略占位值，后续可以根据 Tracker 实际安装位置和姿态在 JSON 中精调。

假设箱子中心世界坐标为 `[1.0, 0.0, 0.15]`，在 SteamVR 工作站执行：

```bash
cd <repo>/motion_tracking/sim2real

uv run --extra vive python scripts/calibrate_vive_box_world.py \
  --vive-config config/g1/omnicontact_vive.json \
  --output config/g1/omnicontact_vive.json \
  --box-center-world 1.0 0.0 0.15 \
  --box-top-z 0.30
```

默认使用 JSON 中的 `object_tracker_serial`。也可以显式指定：

```bash
--serial <OBJECT_TRACKER_SERIAL>
```

脚本会采样 Tracker 位姿并写回：

```text
world_from_steamvr
object_tracker_to_object
```

它不会修改：

```text
robot_tracker_serial
robot_tracker_to_pelvis
goal_position_w
```

所以机器人 Tracker 仍然可以继续用于读取 pelvis 位姿。每次箱子重新放置到新的已知世界坐标时，重新执行该脚本即可更新当前 `world_from_steamvr`。

脚本输出的坐标组合关系是：

```text
^W T_T = ^W T_O * inverse(^T T_O)
^W T_S = ^W T_T * inverse(^S T_T)
```

其中：

- `^S T_T`：OpenVR 读取到的箱子 Tracker 位姿；
- `^T T_O`：当前占位的 Tracker-to-box 外参；
- `^W T_O`：操作者提供的箱子中心世界位姿，当前姿态默认单位姿态；
- `^W T_S`：最终写入 JSON 的 SteamVR 到任务世界变换。

如果输出 JSON 中的 `calibration_confirmed` 不是 `true`，脚本会保留原值并给出警告。检查日志和坐标方向无误后，再将其设置为 JSON 布尔值 `true`，否则部署配置加载器会拒绝启动。

### 5.6 每帧的世界位姿变换

OpenVR 提供 Tracker 在 SteamVR 系中的位姿：

```text
^S T_Trobot
^S T_Tobject
```

运行时组合为：

```text
^W T_Trobot = ^W T_S * ^S T_Trobot
^W T_Tobject = ^W T_S * ^S T_Tobject

^W T_pelvis = ^W T_Trobot * ^T_robot T_pelvis
^W T_object = ^W T_Tobject * ^T_object T_object
```

最终送入策略的 `robot_pose.position_w`、`object_pose.position_w` 和 `goal_position_w` 必须全部属于同一个 `W` 坐标系。

### 5.7 完成 Vive 部署配置

将 `config/g1/omnicontact_vive.example.json` 复制为本地配置，并填写：

- 两个 Tracker 序列号；
- `world_from_steamvr`；
- `robot_tracker_to_pelvis`；
- `object_tracker_to_object`；
- `object_half_extents_m`，注意这是半尺寸；
- `goal_position_w`；
- `calibration_confirmed: true`。

示例结构：

```json
{
  "calibration_confirmed": true,
  "robot_tracker_serial": "ROBOT_SERIAL",
  "object_tracker_serial": "OBJECT_SERIAL",
  "world_from_steamvr": {
    "position_m": [0.0, 0.0, 0.0],
    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]
  },
  "robot_tracker_to_pelvis": {
    "position_m": [0.0, 0.0, 0.0],
    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]
  },
  "object_tracker_to_object": {
    "position_m": [0.0, 0.0, 0.0],
    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]
  },
  "object_half_extents_m": [0.15, 0.15, 0.15],
  "goal_position_w": [3.0, 0.0, 0.26]
}
```

上面的变换数值只是格式示例；真实部署必须替换为实测值。

## 6. 安全行为

代码包含以下保护：

- 未加 `--act` 时绝不发送桥接命令；
- 加 `--act` 时必须同时提供 `--confirm-actuation ENABLE_MOTORS`；
- Tracker 缺失或超过 `pose.max_age_s` 时冻结参考并保持测量关节；
- 非有限、错误形状的策略或状态数据会触发失败退出；
- 策略目标会应用关节限位和每周期目标变化限幅；
- 状态包超时会发送阻尼；
- Vive provider 出错会发送阻尼并退出；
- Python 退出时会尝试发送最终阻尼；
- C++ bridge 超过 200 ms 没有命令时会锁存阻尼，重启 bridge 后才能重新激活；
- Stop 按键会结束策略并发送阻尼。

真实机器人测试前，建议在悬空或充分支撑状态下逐项测试：

1. 关节顺序和初始姿态；
2. Tracker 丢失；
3. UDP 断开；
4. bridge 进程退出；
5. Stop 和 `Ctrl+C`；
6. watchdog 触发；
7. 目标变化限幅；
8. 目标坐标和箱子尺寸是否正确。

## 7. 测试

### 7.1 原始 Tracker 位姿可视化

可以使用 `scripts/view_raw_tracker_poses.py` 单独检查 SteamVR/OpenVR 返回的原始 Tracker 位姿。该脚本不读取 `omnicontact_vive.json`，不会应用 `world_from_steamvr`、Tracker 安装外参或箱子目标变换；显示坐标就是 OpenVR 的 `TrackingUniverseStanding` 坐标。

它会打开一个 MuJoCo 窗口，用两个三棱锥表示两个 Tracker：

- 蓝色：robot Tracker；
- 橙色：object Tracker；
- 每个三棱锥的等边三角形底边为 10 cm，高度为 10 cm；
- 三棱锥底面位于 Tracker 局部 `z=0`，锥尖沿 Tracker 局部 `-Z` 方向；
- 三棱锥的位置和姿态随 OpenVR 读取结果实时更新；
- 世界坐标系原点处显示 X/Y/Z 正方向轴，颜色分别为红/绿/蓝；
- 每个三棱锥原点处显示同样颜色约定的 Tracker 局部 X/Y/Z 轴，局部轴会随 Tracker 姿态旋转；
- 地面使用高对比度棋盘纹理，便于观察位置和高度变化；
- 相机只在首次获得位姿时自动取景，之后保持固定，不会跟随 Tracker 平移；
- Tracker 暂时无效时保留最后位置并降低对应三棱锥透明度。

先列出 Tracker 序列号：

```bash
cd <repo>/motion_tracking/sim2real
uv run --extra vive python scripts/list_vive_trackers.py --seconds 5
```

明确指定两个 Tracker 后运行：

```bash
uv run --extra vive python scripts/view_raw_tracker_poses.py \
  --robot-serial <ROBOT_TRACKER_SERIAL> \
  --object-serial <OBJECT_TRACKER_SERIAL>
```

如果不指定序列号，脚本会按 OpenVR 发现结果排序，自动选择前两个 Generic Tracker：

```bash
uv run --extra vive python scripts/view_raw_tracker_poses.py
```

可用参数：

```text
--robot-serial       蓝色 robot Tracker 序列号
--object-serial      橙色 object Tracker 序列号
--fps                读取和刷新频率，默认 90 Hz
--camera-distance    MuJoCo 相机距离；默认根据两个 Tracker 间距自动设置
```

终端会每秒打印一次有效性和原始位置，例如：

```text
robot=valid, object=valid
  robot raw xyz=[...]
  object raw xyz=[...]
```

这个工具适合检查以下问题：

1. SteamVR 是否能发现两个 Tracker；
2. Tracker 是否在移动时返回连续位姿；
3. robot/object 序列号是否选反；
4. Tracker 重连或遮挡后是否变为 invalid；
5. 在进行世界坐标标定前，确认 OpenVR 原始坐标方向和运动方向。

该工具只做可视化，不连接 G1 bridge，也不会发送任何机器人命令。

### 标定世界系与 sim2real 仿真孪生

使用下面的测试脚本可以直接打开与 sim2sim 相同的 MuJoCo 搬箱场景，并实时显示：

- 标定世界系中的 G1（pelvis 位姿来自 robot Tracker，关节默认使用 `OmniContact.yaml` 的 `default_angles_lab`）；
- 标定世界系中的箱子；
- robot/object 两个 Tracker 的 10 cm 三棱锥及其局部坐标轴；
- 世界原点、pelvis 和箱子坐标轴；
- G1 bridge 镜像的 29 个实测关节；
- 策略生成的 wrist/torso/ankle reference、ghost robot、ghost box、接触状态和起终点平面。

坐标轴颜色约定为 X 红、Y 绿、Z 蓝。孪生窗口是独立只读进程，不应放进
`run_bridge.sh`；MuJoCo/OpenVR 窗口关闭或阻塞不会进入电机命令链路。

脚本只在运行目录生成临时 XML，不修改 `assets/omnicontact_carry_box.xml`，也不发送控制命令：

```bash
cd <repo>/motion_tracking/sim2real
uv run --extra vive python scripts/real_omnicontact_viewer.py \
  --vive-config config/g1/omnicontact_vive.json
```

默认端口关系为：bridge 状态 `55001` 给 deploy，电机命令 `55002` 给 bridge，
只读状态镜像 `55003` 给孪生窗口，策略 reference/ghost `55004` 给孪生窗口。
完整启动顺序如下：

```bash
# 终端 1
cd <repo>/motion_tracking/g1_sim2real
G1_NET=<有线网卡> bash scripts/run_bridge.sh

# 终端 2
cd <repo>/motion_tracking/sim2real
uv run --extra vive python scripts/real_omnicontact_viewer.py \
  --vive-config config/g1/omnicontact_vive.json

# 终端 3：先无电机输出观察
cd <repo>/motion_tracking/sim2real
uv run --extra vive python src/deploy_omnicontact.py \
  --vive-config config/g1/omnicontact_vive.json \
  --pose-source local --run-seconds 30
```

确认坐标、关节和 ghost 对齐后，才在终端 3 加
`--act --confirm-actuation ENABLE_MOTORS`。无电机输出模式也会发布只读
reference/ghost，因此可以先完整检查孪生显示。

如果配置仍处于编辑阶段、`calibration_confirmed` 尚未改成 `true`，可以显式允许只读预览（不会写回配置）：

```bash
uv run --extra vive python scripts/real_omnicontact_viewer.py \
  --vive-config config/g1/omnicontact_vive.json --allow-unconfirmed
```

默认已经从 bridge 的只读镜像端口 `55003` 叠加实时 29 关节；根部平移和姿态仍以 robot Tracker 标定结果为准：

```bash
uv run --extra vive python scripts/real_omnicontact_viewer.py \
  --vive-config config/g1/omnicontact_vive.json \
  --state-host 127.0.0.1 --state-port 55003 \
  --visualization-host 127.0.0.1 --visualization-port 55004
```

`--state-port 0` 可关闭实测关节，`--no-visualization` 可关闭策略叠加，
`--no-robot` 可隐藏 G1 网格。reference/ghost 数据超过 `--stale-timeout`
（默认 0.5 秒）未更新时会自动隐藏，避免把过期姿态误认为实时输出。
`--fps` 控制 viewer 刷新频率。相机只在启动时设定一次，不会锁定或跟随 Tracker 的平移。

#### 交互调整 robot Tracker 到 pelvis 的安装变换

在标定 viewer 上增加下面的参数，会同时打开一个包含六个滑动条的小窗口：

```bash
uv run --extra vive python scripts/view_calibrated_omnicontact_poses.py \
  --vive-config config/g1/omnicontact_vive.json \
  --tune-robot-tracker-to-pelvis \
  --state-port 0 --no-visualization
```

六个滑动条直接表示绝对变换 `robot_tracker_to_pelvis`，即
`^Tracker T_pelvis`：XYZ 单位为米，Roll/Pitch/Yaw 单位为度，旋转采用固定轴
X-Y-Z。拖动后 MuJoCo 中的 pelvis 坐标轴和机器人根部会实时更新。

- 在 MuJoCo 窗口或滑动条窗口按 `S`：只覆盖当前 `--vive-config` JSON 中的
  `robot_tracker_to_pelvis`，其他标定字段保持不变；写入采用临时文件原子替换。
- 不按 `S` 直接关闭任一窗口：不会写配置文件。
- XYZ 默认范围为 `[-0.5, 0.5]` 米；需要更大范围时使用
  `--tune-translation-range <米>`。
- 当前四元数处于欧拉角奇异位形时，面板会选择一个等价 RPY 表达；打开面板
  本身不会保存或改变配置中的四元数，只有按 `S` 才会写入等价的新四元数。

如果孪生窗口在另一台电脑，把 `g1_sim2real/config/g1_bridge.yaml` 中的
`state_mirror_host` 改成 viewer 电脑的有线 IP，并给 deploy 加
`--visualization-host <VIEWER_IP>`；viewer 使用
`--state-host 0.0.0.0 --visualization-host 0.0.0.0`。`G1_NET` 只选择 Unitree
DDS 网卡，不决定 viewer 的 UDP 路由。

```bash
cd <repo>/motion_tracking/sim2real
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

OmniContact 相关测试包括：

- `tests/test_omnicontact_policy.py`：策略、参考轨迹、动作和仿真可视化数据；
- `tests/test_omnicontact_vive.py`：刚体变换、Vive 位姿发布、UDP HMAC 和 Tracker 丢失行为。

## 8. 常见问题

### MuJoCo 找不到 XML

确认当前配置使用：

```yaml
xml_path: "assets/omnicontact_carry_box.xml"
```

并从 `sim2real` 目录启动。不要再使用旧版本的外部 `OmniContact_sim2sim` 路径。

### 控制器一直等待位姿

检查：

- SteamVR 是否已启动；
- 两个序列号是否正确；
- 两个 Tracker 是否都是 Generic Tracker；
- `calibration_confirmed` 是否为 JSON 布尔值 `true`；
- local 模式是否使用 `uv sync --extra vive`；
- UDP 模式两端 token 是否完全一致；
- `--allowed-sender-ip` 是否写成发送端有线 IP。

### 按 A 后没有开始搬箱

只有在以下条件同时满足时 A 才会生效：

- 已经完成 Start 到默认姿态的过渡；
- 机器人 Tracker 和箱子 Tracker 都存在；
- 位姿年龄不超过 `0.10 s`；
- 置信度不低于 `0.90`。

### 实机动作方向不对

优先检查世界坐标和刚体变换，而不是先修改策略：

- `world_from_steamvr` 的 X/Y 是否反向；
- 标定时是否旋转了 Tracker；
- `robot_tracker_to_pelvis` 是否写成了反向变换；
- `object_tracker_to_object` 是否指向箱子几何中心；
- `goal_position_w` 是否使用同一个世界系；
- 箱子半尺寸是否误填成完整尺寸。
