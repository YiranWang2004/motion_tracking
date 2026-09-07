# 双 G1：用 ScaleBFM 跟踪 DefaultPose 站立

2026-09-06 更新。适用于双机 residual 和 `--scalebfm-only`，sim2sim 和实机使用同一实现。
正式入口不再加载 `LocoMode.onnx`，B 阶段也不调用 residual actor。

## 当前按键流程

`ZERO TORQUE → Start/s → 两秒 DefaultPose PD 过渡 → DefaultPose ready → B/b → ScaleBFM DefaultPose standing → A/a → 参考动作 → ScaleBFM DefaultPose standing → Stop/x`

- S 阶段：从实测关节角插值至 DefaultPose，使用原有直接 PD 过渡；仿真保持两台 base 锁定。
- B 阶段：ScaleBFM 跟踪静态 DefaultPose 参考，使用基础模型自身的 PD 增益。
  仿真收到两路同帧站立命令后同步释放 base；有箱模式同时释放箱子。
- A 阶段：初始化任务对齐和任务历史，开始推进共享 reference 帧。完整模式加入 residual，
  `--scalebfm-only` 使用纯基础模型、无物理箱子，也不执行箱子相关终止检查。
- Reference 完成：重新初始化静态站立参考，回到 ScaleBFM 站立。当前进程不重复播放任务。

仍须按顺序重新按下 Start/B/A。实机没有浮动基座锁定；输入仍来自既有遥控器或 bridge
控制接口，任一侧的有效按钮沿控制双机共享阶段。推理终端没有新增键盘输入功能。

## 静态参考如何生成

实现位于 `src/dual_runtime/scalebfm_standing.py` 的 `DualScaleBFMStanding`。
首次进入站立时，分别记录两台当前 pelvis 高度；取各自当前 XY 和 yaw，结合 DefaultPose
关节角做 FK，构建关键身体的世界坐标参考。六个未来参考帧全部相同，表达原地站立。
参考不会每个 tick 跟随机器人漂移。任务完成后以新的 XY/yaw 重新锚定，保留首次站立高度，
避免把动作结束时的下蹲高度作为长期站立目标。

站立和任务共用已加载的 ScaleBFM 网络、FK 资源和控制模式，但各自维护观测历史和上一帧
基础 action。B 阶段每个控制周期将两台机器人作为 batch=2 一次推理，不推进任务帧、不污染
任务历史。A 阶段继续使用原有任务初始化逻辑。`standing_asset_dir: omnicontact` 仍用于
读取 DefaultPose 和关节顺序配置，不再表示需要走路网络。

日志状态为 `scalebfm_standing`，metadata 中记录 `standing_policy: scalebfm_default_pose`。
站立处理耗时查看 `processing_time_s`；任务的 `inference_time_s` 不代表站立阶段耗时。
目标使用关节范围、阶段变化量和估计力矩限制；站立绝对倾角保护继续有效，不用于动态任务。
配置使用 `control.phase_target_delta.scalebfm_standing: 1.0`；旧 `loco_standing` 配置键和
协议阶段保留兼容，正式入口选择新控制器。

## 启动命令

命令与替换站立策略前相同，不需要新增参数。推荐先做空载 sim2sim：

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
bash scripts/run_dual_scalebfm_sim2sim.sh \
  --config config/g1/dual_scalebfm_contact_v2_8192.yaml \
  --scalebfm-only
```

完整带箱 sim2sim：

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
bash scripts/run_dual_scalebfm_sim2sim.sh \
  --config config/g1/dual_scalebfm_contact_v2_8192.yaml
```

实机在完成双 bridge、Vive 标定和支撑准备后，空载推理入口为：

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
bash scripts/run_dual_scalebfm_residual.sh \
  --config config/g1/dual_scalebfm_contact_v2_8192.yaml \
  --pose-source vive \
  --scalebfm-only \
  --act-robot both \
  --confirm-actuation ENABLE_MOTORS \
  --no-visualization
```

完整实机模式去掉 `--scalebfm-only`，并恢复箱子 Tracker 和相应几何检查所需配置。
完整前置步骤及分终端命令见 [空载测试指南](DUAL_SCALEBFM_ONLY_TEST_ZH.md)、
[部署指南](DUAL_SCALEBFM_DEPLOY_ZH.md) 和 [仿真启动速查](DUAL_SIM2SIM_QUICK_REFERENCE_ZH.md)。

## 本次验证

相关 82 项测试通过，覆盖静态参考、独立历史、双机批量推理、B/A/完成状态转换、
仿真释放、空载模式及既有部署保护。真实基础模型另外通过两组独立本机 UDP headless 验证：

| 场景 | 结果 | 站立最大倾角 | 站立处理耗时 P95 |
| --- | --- | --- | --- |
| 空载，默认原地抬放 reference | 完成第 1～429 帧，回到 ScaleBFM 站立，正常停止 | 两台均约 3.22° | 5.92 ms |
| 带箱完整模式，仅 S/B 站立，不按 A | 751 个站立 tick，约 15.02 秒，正常停止 | 两台均约 3.22° | 6.55 ms |

空载运行合计记录 672 个站立 tick（含任务前后），最终 pelvis 高度约 0.7845/0.7848 m；
带箱站立最终高度两台均约 0.7869 m。两组记录的 tick 均正常。

日志分别位于：

- `logs/dual_scalebfm_only_validation/20260906_211319_069391986/`
- `logs/dual_scalebfm_standing_validation/20260906_211440_696890561/`

每组包含 `rollout.npz`、`metadata.json`、`console.log`、`validation.json`。
本次没有执行实机或实际 Vive 测试，也没有验证新站立控制器下的完整带箱搬运动作。
此前 LocoMode 的日志仅作为历史记录，不代表新控制器的测试结果。

## S 后迟迟没有 DefaultPose ready 时如何定位

`default_pose_duration_s: 2.0`，50 Hz 下过渡为 100 个有效控制周期，从
`Start accepted` 开始计数。`WAITING FOR BRIDGES` 不计入过渡；启动输入等待上限为
60 秒。仿真 lockstep 在等待配对命令时不会推进物理或重复计数，因此机器或 GUI 卡顿时，
墙钟时间可能超过 2 秒；ready 表示插值周期完成，不是关节误差自动收敛判据。

- 只有 `WAITING FOR BRIDGES`：控制器尚未拿到完整有效输入，S 未使能。
  现在每两秒打印具体原因和 A/B 接收情况；GUI 模式须完成 MuJoCo 窗口初始化。
- 出现 `ZERO TORQUE`，没有 `Start accepted`：按键未被接受。GUI 在 MuJoCo 窗口按 S；
  headless 在仿真终端输入 s 后回车。初始握手之前的按键现在明确提示忽略，需重新按 S。
- 出现 `Start accepted`：过渡已开始，正常 100 tick 后打印 ready。
  重传同一仿真快照不算新的过渡周期，不要通过缩短保护或跳过阶段掩盖通信停顿。

旧代码会静默吞掉首帧上的 S，因为启动时的按钮状态只用于初始化按键沿。
本次让仿真初始握手不接受运动按键，并加入明确反馈；实机的启动按住保护保留，
同样打印松开重按提示。仅凭 WAITING 那几行不能确定输入未就绪的具体底层原因。

本次启动问题回归：59 项相关测试通过；另以真实模型、独立本机 UDP 端口完成
headless ZERO TORQUE → S → 100 tick → ready → X。日志：
`logs/dual_startup_validation/20260906_212141_582148652/rollout.npz`，
终端记录位于 `logs/dual_startup_validation/console.log`。没有复现 GUI 初始化停顿。

### 已修复：启动持续报 skewed_bridge_state

收到两路状态但持续报 `skewed_bridge_state` 的原因，是仿真在确认两路
`state_receive_time_ns` 相同后，仍套用了实机的 50 ms 接收时刻偏差检查。
MuJoCo lockstep 的同一快照在等待命令时保持冻结，UDP 重传和接收线程调度会使两路
到达时刻不同，却不表示物理状态来自不同周期。拒绝该帧后控制器不发命令，仿真不能
推进，只能继续重传，因而可能一直停留在启动等待。

现在仿真使用相同且非空的源快照时间戳确认配对，不再对这样的帧套用接收时刻偏差门限。
仍检查单包新鲜度、位姿新鲜度、跨快照配对、运行时超时；实机继续执行原来的偏差门限。
旧版无源时间戳输入仍保留接收偏差检查。无需修改 YAML 或提高超时。
回归覆盖同帧接收相差 80 ms 时仿真进入 ZERO TORQUE 并接受 S，实机仍拒绝，
以及同帧但数据过期仍拒绝的情形。修改后须重启仿真和部署进程。
