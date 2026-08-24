# OmniContact sim2real 工作记录

本文记录本次对话中对 OmniContact 仓库进行的说明、标定约定、可视化工具和修复。后续调试 Vive、sim2sim 或实机显示时，可以把本文作为快速索引。

## 1. 仓库和运行目录

当前 OmniContact 工程位于：

```text
/home/bcj/wyr/motion_tracking/sim2real
```

建议从该目录运行命令：

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
```

项目已经将 OmniContact 的 MuJoCo 资源放在仓库内，不再依赖外部 `OmniContact_sim2sim` 仓库。主要场景文件是：

```text
config/g1/assets/omnicontact_carry_box.xml
```

该场景包含 G1、ghost G1、箱子、地面、参考物体和搬箱任务所需的可视化 body。

## 2. Tracker 位姿读取代码

底层 OpenVR 读取器：

```text
src/omnicontact/perception/openvr_tracker.py
```

主要接口：

- `OpenVRTrackerReader.start()`：启动 OpenVR；
- `refresh_devices()`：刷新 Generic Tracker 设备和硬件序列号；
- `read_all()`：读取指定 Tracker 的当前位置和四元数；
- `ViveSample`：保存原始位置和 `xyzw` 四元数。

标定后的两 Tracker 位姿接口：

```text
src/omnicontact/perception/vive_pose.py
```

主要类型和函数：

- `RigidTransform`：使用 `position + quaternion_xyzw` 表示刚体变换；
- `ViveDeploymentConfig.load()`：加载 JSON 标定配置；
- `VivePoseProvider.update_once()`：读取两个 Tracker 并发布机器人 pelvis 和箱子位姿；
- `sample_to_transform()`：将 `ViveSample` 转换成刚体变换。

变换约定为：

```text
^W T_Trobot = ^W T_SteamVR · ^SteamVR T_Trobot
^W T_Tobject = ^W T_SteamVR · ^SteamVR T_Tobject

^W T_pelvis = ^W T_Trobot · ^Trobot T_pelvis
^W T_box    = ^W T_Tobject · ^Tobject T_box
```

其中：

- `world_from_steamvr` 是 `^W T_SteamVR`；
- `robot_tracker_to_pelvis` 是 `^Trobot T_pelvis`；
- `object_tracker_to_object` 是 `^Tobject T_box`。

## 3. 原始 Tracker 可视化

脚本：

```text
scripts/view_raw_tracker_poses.py
```

该脚本直接显示 OpenVR `TrackingUniverseStanding` 原始坐标，不应用世界标定和安装外参。

显示特性：

- 两个边长和高度均为 10 cm 的三棱锥；
- robot Tracker 为蓝色，object Tracker 为橙色；
- 三棱锥尖端沿 Tracker 局部 `-Z` 方向；
- 显示 Tracker 局部 X/Y/Z 坐标轴；
- 显示世界坐标轴；
- 明亮的 skybox、灯光和棋盘地面；
- 相机只在启动时取景，不跟随 Tracker 平移。

运行方式：

```bash
uv run --extra vive python scripts/view_raw_tracker_poses.py
```

## 4. 箱子世界坐标标定约定

实际放置箱子时采用以下简化标定方法：

1. 将 object Tracker 放在箱子上表面中心；
2. Tracker 局部 `+Z` 轴朝下；
3. Tracker 的另外两个轴分别与箱子上表面的两条水平棱平行；
4. 箱子上表面高度为 `0.30 m`；
5. Tracker 三个轴与世界坐标系三个轴平行；
6. 根据已知世界原点到箱子中心的变换，求 `world_from_steamvr`。

箱子 Tracker 到箱子中心的安装外参暂时使用占位值：

```json
{
  "position_m": [0.0, 0.0, 0.15],
  "quaternion_xyzw": [1.0, 0.0, 0.0, 0.0]
}
```

其中 `0.15 m` 对应箱子中心到箱子上表面的半高度。该值和姿态后续需要根据实际 Tracker 安装方式精调。

标定脚本：

```text
scripts/calibrate_vive_box_world.py
```

示例命令：

```bash
uv run --extra vive python scripts/calibrate_vive_box_world.py \
  --vive-config config/g1/omnicontact_vive.json \
  --output config/g1/omnicontact_vive.json \
  --box-center-world 1.0 0.0 0.15 \
  --box-top-z 0.30
```

这里的 `--box-center-world 1.0 0.0 0.15` 表示箱子中心在世界系中的目标位置，`--box-top-z 0.30` 表示箱子上表面高度。

## 5. robot Tracker 到 pelvis 的粗略估计

根据“Tracker 紧贴 pelvis 后表面中央”的描述，暂时使用：

```json
"robot_tracker_to_pelvis": {
  "position_m": [0.0, 0.0, 0.07],
  "quaternion_xyzw": [0.5, -0.5, 0.5, 0.5]
}
```

对应的局部坐标约定：

- Tracker `+Z` 朝机器人前方；
- Tracker `+X` 朝机器人右方；
- Tracker `+Y` 朝机器人下方。

当前占位旋转表示：

```text
pelvis +X -> Tracker +Z
pelvis +Y -> Tracker -X
pelvis +Z -> Tracker -Y
```

该值只是估计值，不能视为最终机械安装标定结果。实机动作方向或位置出现偏差时，应优先重新测量该外参。

## 6. sim2sim 场景中的初始状态

sim2sim 配置：

```text
config/g1/bridge_omnicontact.yaml
```

关键字段：

```yaml
xml_path: "assets/omnicontact_carry_box.xml"

task_object:
  body_name: "box"
  geom_name: "box_geom"
  initial_position: [1.0, 0.0, 0.15]

root_qpos_control: [0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0]

pre_control_marker:
  body_name: "mid360_link"
  offset: [0.0, 0.0, 0.25]
  radius: 0.055
  half_length: 0.10
  rgba: [1.0, 0.0, 0.0, 0.90]
```

含义：

- `task_object.initial_position` 是箱子初始中心位置；
- 箱子姿态由场景中的 free joint 初始四元数决定，默认是单位姿态；
- `root_qpos_control` 用于把机器人初始化到任务对应的地面根部位姿；按下 `s` 后
  保持根部锁定和 DefaultPose，按下 `b` 并收到首个 LocoMode 命令时同步释放；
- sim2sim 的 `home_q` 使用原版 DefaultPose；`sim_prepare_seconds: 0.02` 完成一个
  50 Hz 的 DefaultPose 过渡后持续保持该姿态，按 `b` 后才由原版 LocoMode
  接管。实机继续使用 `prepare_seconds: 2.0`，并由遥控器 `B` 触发 LocoMode；
- `pre_control_marker` 是按下 A 之前跟随显示在机器人头顶的红色圆柱，进入
  A 键后的策略时隐藏；
- `0.793 m` 是 sim2sim 中 G1 pelvis 的名义高度。

sim2sim viewer 入口位于：

```text
src/sim2sim.py
```

通过 `mujoco.viewer.launch_passive(model, data, ...)` 启动 MuJoCo 窗口，并更新真实机器人、ghost、箱子和参考可视化对象。

## 7. 标定世界系综合可视化脚本

新增脚本：

```text
scripts/view_calibrated_omnicontact_poses.py
```

该脚本复用 sim2sim 的搬箱场景，实时显示：

- 标定世界系中的 G1；
- 标定世界系中的箱子；
- robot Tracker 和 object Tracker 的三棱锥；
- 两个 Tracker 各自的局部坐标轴；
- 世界坐标轴；
- pelvis 坐标轴；
- 箱子坐标轴。

默认机器人关节角来自：

```text
config/g1/omnicontact/OmniContact.yaml
```

其中的 `default_angles_lab`。机器人 pelvis 的位置和姿态来自 robot Tracker 的标定结果，箱子位姿来自 object Tracker 的标定结果。

基本命令：

```bash
uv run --extra vive python scripts/view_calibrated_omnicontact_poses.py \
  --vive-config config/g1/omnicontact_vive.json
```

如果配置还没有确认：

```bash
uv run --extra vive python scripts/view_calibrated_omnicontact_poses.py \
  --vive-config config/g1/omnicontact_vive.json \
  --allow-unconfirmed
```

如果需要用 G1 bridge 的实时 29 关节状态覆盖默认关节角：

```bash
uv run --extra vive python scripts/view_calibrated_omnicontact_poses.py \
  --vive-config config/g1/omnicontact_vive.json \
  --state-host 127.0.0.1 \
  --state-port 55001
```

此 viewer 是只读工具：

- 不启动 OmniContact 策略；
- 不发送 command UDP；
- 不发送电机控制命令；
- 不自动修改标定 JSON；
- 只读取 Vive 和可选的状态 UDP。

可用选项：

```text
--xml-path       指定其它 MuJoCo 场景 XML
--fps            viewer 刷新频率，默认 60 Hz
--state-host     G1 状态 UDP 地址
--state-port     G1 状态 UDP 端口；不指定时使用默认关节姿态
--no-robot       隐藏 G1 网格，只观察箱子、Tracker 和坐标系
--allow-unconfirmed 只读预览未确认配置
```

## 8. 曾遇到的路径错误及修复

错误信息：

```text
FileNotFoundError: .../config/g1/assets/bridge_omnicontact.yaml
```

原因是脚本曾经把 `xml_path.parent`（即 `config/g1/assets`）传给关节配置加载函数。修复后脚本从：

```text
config/g1/bridge_omnicontact.yaml
config/g1/omnicontact/OmniContact.yaml
```

读取关节名称和默认关节角。

因此不需要修改命令，也不需要手动创建 `config/g1/assets/bridge_omnicontact.yaml`。

## 9. 验证命令

脚本静态编译：

```bash
uv run python -m py_compile scripts/view_calibrated_omnicontact_poses.py
```

代码空白检查：

```bash
git diff --check
```

运行仓库测试：

```bash
PYTHONPATH=src uv run python -m unittest discover -s tests -v
```

当前已有的 16 项测试全部通过。

## 10. 当前注意事项

1. `calibration_confirmed` 必须在正式部署前设置为 JSON 布尔值 `true`。
2. `robot_tracker_to_pelvis` 和 `object_tracker_to_object` 目前仍有粗略占位性质，需要根据实际安装位置和方向精调。
3. MuJoCo free joint 使用 `wxyz` 四元数，而 Vive 和配置文件使用 `xyzw`；脚本中已经显式完成转换。
4. Tracker 丢失时 viewer 会保留上一帧姿态，重新检测到设备后继续更新。
5. 该工具用于观察世界坐标、安装外参和模型位置是否一致，不等价于实机控制前的最终安全验证。
