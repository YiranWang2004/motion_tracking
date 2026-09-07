# 双机 sim2sim：启动时导入实测 Vive 场景

> 2026-09-07 默认 reference 已替换为原地抬放动作的世界 Z 轴 −90°版本，双机沿 Y 排列；间距仍为 1.8 m。详见 [旋转与验证记录](DUAL_REFERENCE_ROTATION_ZH.md)。此前完整闭环数据属于旋转前动作。

> 当前双机默认采用世界系参考：`reference_alignment: none`。实际 A、B、箱子分别与各自 reference 世界系位姿比较；虚影不随实际 A 平移或旋转。旧的 A 锚定描述仅适用于显式选择 `xyyaw` 的历史配置。位置门限暂沿用 `preflight.max_partner_position_error_m`，现在同时检查 A 和 B。实测标定世界系需与 motion 世界系一致；本次不自动重定位参考、不放宽门限。

双机支持与单机器人类似的 `--initial-scene-source vive` 模式。启动时读取一次新鲜的
标定世界系位姿，初始化机器人 A pelvis、机器人 B pelvis 和箱子的 MuJoCo free joint。
位置和完整朝向均导入；采样后关闭 OpenVR，后续只由 MuJoCo 物理演化，不持续跟随 Tracker。
关节角仍从 DefaultPose 初始化，不读取实机关节 bridge，也不向实机发送控制命令。

## 一条命令启动

先完成 SteamVR/Tracker 跟踪和双机 JSON 标定，采样时保持两台机器人及箱子静止。

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
bash scripts/run_dual_scalebfm_sim2sim.sh \
  --config config/g1/dual_scalebfm_contact_v2_8192.yaml \
  --initial-scene-source vive \
  --vive-config config/g1/omnicontact_vive_dual.json
```

空载、无箱子的 ScaleBFM-only 模式：

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
bash scripts/run_dual_scalebfm_sim2sim.sh \
  --config config/g1/dual_scalebfm_contact_v2_8192.yaml \
  --scalebfm-only \
  --initial-scene-source vive \
  --vive-config config/g1/omnicontact_vive_dual.json
```

only 模式仅需两台机器人 Tracker，不读取箱子 Tracker，不导入箱子位姿，不检查箱子尺寸。
添加 `--headless` 可以无窗口运行，在仿真终端输入 s/b/a/x 后回车。

## 分两个终端启动

终端 1 负责一次 Vive 采样和仿真：

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
bash scripts/run_dual_scalebfm_simulator.sh \
  --config config/g1/dual_scalebfm_contact_v2_8192.yaml \
  --initial-scene-source vive \
  --vive-config config/g1/omnicontact_vive_dual.json
```

终端 2 始终读取仿真位姿：

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
bash scripts/run_dual_scalebfm_residual.sh \
  --config config/g1/dual_scalebfm_contact_v2_8192.yaml \
  --pose-source sim \
  --act-robot both \
  --confirm-actuation ENABLE_MOTORS \
  --no-visualization
```

分终端空载测试须在两条命令中都加 `--scalebfm-only`。此模式不启动真实机器人 bridge。
一键脚本将 Vive 初始化参数只传给仿真进程；策略进程继续使用 `--pose-source sim`。
两个仿真 wrapper 已包含 `uv --extra vive` 依赖。

## 参数和约束

| 参数 | 默认值 / 含义 |
| --- | --- |
| `--initial-scene-source` | `config`：保留原来的 reference/config 初始布局；`vive`：导入实测布局 |
| `--vive-config` | 显式路径相对当前工作目录；省略时读取 YAML 的 `vive_config`，相对 YAML 所在目录 |
| `--vive-hz` | 100 Hz，采样轮询频率 |
| `--pose-wait-timeout` | 10 秒，等待完整新鲜快照的上限 |
| `--pose-max-age` | 0.1 秒，快照最大年龄 |

JSON 使用与双机实机部署相同的 `world_from_steamvr`、两台 `tracker_to_pelvis`、
箱子外参和 Tracker serial，要求 `calibration_confirmed: true`。必须用双机 JSON，不能
直接使用只有一台机器人配置的单机 JSON。完整模式的 `object_half_extents_m` 必须匹配
仿真和 reference 尺寸，当前默认原地抬放动作半尺寸是 `[0.5, 0.15, 0.15]` 米。
不一致时拒绝初始化，不会为实测箱子任意缩放模型或动作。

读取失败、数据缺失或过期时明确报错并关闭 Vive provider，不回退到默认布局。
采样的完整 pelvis 位姿覆盖配置中的初始高度；关节维持 DefaultPose，速度初始化为零。
因此采样时真实机器人姿态应与要模拟的 DefaultPose 场景相容，世界标定地面应对应仿真地面。

按键流程保持不变：等待 ZERO TORQUE → S → 两秒 DefaultPose → ready → B →
ScaleBFM DefaultPose standing → 确认稳定 → A → reference → 站立 → X。
仿真在 B 前保持导入的 free joint 初始位姿，B 后释放。A 时直接比较各自标定世界系位姿与 reference 世界系位姿，检查初始几何。该功能不重新生成动作，不保证任意 A/B/箱子相对布局都适配现有 motion；
也没有新增单机器人式的 `--goal-position` 接口。

## 验证范围

自动化测试覆盖 provider 成功/失败时关闭、过期与缺失拒绝、完整位置和朝向导入、
DefaultPose 关节保持、锁定初始位姿一致、空载忽略箱子、箱子尺寸不一致拒绝、源数据变化
不驱动后续仿真。场景测试使用真实 MuJoCo 和合成标定位姿。
本次没有运行实际 SteamVR/Tracker 采样或真实机器人。

## 在同一 MuJoCo 窗口比较参考虚影

sim2sim 默认叠加青色半透明的两台机器人参考虚影；有物理箱子时同时显示参考箱子，
`--scalebfm-only` 隐藏参考箱子。原有启动命令不变，无需打开额外 viewer 或可视化 UDP。
MuJoCo 窗口按 G 切换虚影显示。

- 按 A 前：显示任务 `start_frame` 的参考姿态，固定显示 reference 的世界系坐标（`reference_alignment: none`），便于在预检之前检查 B 和箱子的相对位置。
  这是任务起始姿态预览，不是 B 阶段的静态 DefaultPose 参考。
- 执行任务时：参考世界系保持固定，虚影跟随双机命令中的 reference 帧。
  保留真实的空间误差，不单独把 B 或箱子移到实际位置。
- 任务完成后：保留最后执行帧，G 可以隐藏虚影。

当前默认不做以 A 为锚点的平移、旋转或高度对齐。虚影关节来自 motion，不是 actor 输出的 PD target。渲染使用独立 MjData，
只向 viewer 场景添加几何，不增加物理模型的刚体、质量或碰撞，不改变实际关节状态。

此前 `0.781 m` 报错来自旧的 A 锚定逻辑。现在 A/B 分别计算
`norm(actual_position_w - reference_position_w)`，报错同时列出世界系实际和参考坐标。应在按 A 前观察虚影的站位
和朝向，检查真实布局、A/B Tracker 对应关系及 pelvis 外参；该显示不放宽预检阈值。
