# 单 G1 ScaleBFM / Pico 遥操作

这是独立的单机器人运行链路，不加载双机 residual、partner、object 或 reference_bundle.npz。
主机推理和本体推理运行同一入口，只改变 bridge/Pico/Vive 地址和计算后端。
PMG 的 `src/deploy.py`、原有双机入口和用户已有配置保持独立。

## 代码划分

- `src/scalebfm/`：共享网络、checkpoint、历史观测、关节顺序、FK、TensorRT worker。
- `src/scalebfm_tracking/`：单机配置、Pico 实时参考、根部定位和 bridge 控制循环。
- `src/dual_runtime/`：原双机业务；旧公共模块路径通过兼容导入继续工作。
- `config/g1/tracking_scalebfm.yaml`：单机参数。artifact 路径相对于 YAML；只读取模型、metadata、mode table、XML 四个文件。

## 定位与参考语义

| 模式 | 根部位置 | 根部朝向 | 限制 |
| --- | --- | --- | --- |
| `local` | IMU + 支撑脚 FK 的局部里程计 | IMU，启动 yaw 归零 | 无力传感器；平地接触假设，可能漂移，不能识别所有滑步 |
| `vive` | 单个 pelvis Tracker 的标定世界位置 | IMU，启动时对齐 Vive yaw | 检查 Vive/IMU 朝向差；Tracker 失效不切换 local |

Vive 变换顺序：`T_world_steamvr * T_steamvr_tracker * T_tracker_pelvis`。
复用 `omnicontact_vive_dual.json` 中选定 `robot_a` 或 `robot_b` 的 serial/extrinsic；
**不要求另一个机器人和箱子 Tracker 在线**。必须确认 `calibration_confirmed` 和安装外参。

local 的根部高度来自初始支撑脚 FK 加 sole_height，平移来自固定地面接触点约束。
它不是“将世界位置置零”，但也不是实测全局位姿。没有脚底力信号，腾空、滑步、楼梯、悬吊运动均不在该估计器的适用范围内。
几何检测失去支撑或出现速度突变时退出。不能仅凭此检测宣称真机安全。

每次 Pico A 开始/恢复时，把当时 Pico 根部与机器人根部做固定平移/yaw 对齐；
保留之后的相对位移和高度变化，不逐帧重新居中。默认用 0.5 秒混合进入参考。
默认六帧偏移 `[0,1,2,3,4,5]`、50 Hz：用已经收到的历史组成窗口，当前目标延后 100 ms，
再叠加 teleop server 的 lookback、重定向和网络延迟；不伪造未来预测。
`control_mode` 仍是 ScaleBFM 的身体部位 mask，不是“有没有全局定位”开关。

Pico server 增加兼容旧客户端的 v2 元数据：关节名、请求 ID、源样本时间、源年龄和按键。
关节按名字重排；重复旧帧不刷新有效期。使用源年龄加请求 RTT 判断新鲜度，不依赖主机/本体时钟同步。
单个 PUSH/PULL 服务只能配一个 inference 消费者；不能同时运行 PMG 和 ScaleBFM。
**更新代码后需重新启动 teleop server**，旧 server 没有新协议，不能用于本入口。

## 环境与离线检查

以下命令都在仓库根目录运行；本体部署也需要复制新增源码、配置和四个 artifacts。
没有自动同步到 G1，也没有自动修改 bridge 配置。

基础依赖沿用 sim2real 环境，增加 PyTorch（`scalebfm` extra）；只有运行 OpenVR 服务的主机需要 `vive` extra。
Jetson 应使用与其 JetPack 匹配的 PyTorch，或 CPU PyTorch 做预处理 + 系统 TensorRT worker；不要用普通桌面 CUDA wheel 覆盖板载环境。
启动脚本直接用 `sim2real/.venv/bin/python`，不隐式执行 uv sync；可用 `SCALEBFM_PYTHON` 指定已有解释器。
在已有 SDK 的环境中安装依赖时注意不要通过环境同步移除手工安装的 XRoboToolkit SDK。

```bash
bash sim2real/scripts/run_scalebfm_tracking.sh --check --samples 100
```

该检查不打开机器人、Pico、Vive socket。真实 checkpoint 执行 FK、历史/任务构造及 batch=1 推理。
激活控制前也会做预热/基准测试；p95 >= 20 ms 拒绝 `--actuate`，但离线通过不保证负载下持续 50 Hz。

## 主机推理

1. 主机启动 XRoboToolkit PC service，连接 Pico，完成校准和 body streaming。
2. 主机启动更新后的重定向服务（已有进程应先正常退出，不可重复占端口）：

```bash
cd sim2real/teleop
../.venv/bin/python serve_xrobot_teleop.py --robot g1
```

3. G1 上的 bridge：在独立配置中把 `udp.state_host` 指向**主机有线 IP**，`state_port=55001`；
   `cmd_bind_host=0.0.0.0`、`cmd_port=55002`。保留 50 Hz 状态发送和 0.20 秒 watchdog。
   使用本体内部 DDS 网卡和当前仓库的 session-aware bridge，不能用旧 bridge 二进制。
   实机吊带/支撑与急停按既有硬件操作规程准备好；启动 bridge 会释放原运动服务。

```bash
# 在 G1 仓库根目录，网卡和配置路径须按实际填写。
G1_NET=<本体DDS网卡> G1_BRIDGE_CONFIG=<外部推理bridge.yaml绝对路径> \
  bash g1_sim2real/scripts/run_bridge.sh
```

4. 主机运行只读观察，`<G1有线IP>` 是机器人有线地址：

```bash
bash sim2real/scripts/run_scalebfm_tracking.sh \
  --cmd-host <G1有线IP> --pose-mode local --duration 20
```

不加 `--actuate` 不发送 PD 命令，也不发起 bridge session；但会独占状态端口和 Pico 消费通道。
观察进程退出、确认输入/限幅/时延正常后，才在现场准备妥当时加 `--actuate`。

## 本体推理

Pico/重定向仍在主机，本体 bridge 与 policy 通过 localhost 通信。
bridge 可使用现有 `g1_sim2real/config/bridge_onboard_scalebfm.yaml`；无需启动双机 relay/coordinator。

```bash
# G1 上，G1_NET 必须按这台机器实际 DDS 网卡填写。
G1_NET=<本体DDS网卡> bash g1_sim2real/scripts/run_onboard_bridge.sh

# 另一个 G1 终端；默认 cmd_host=127.0.0.1。
bash sim2real/scripts/run_scalebfm_tracking.sh \
  --pico-host <主机可达IP> --pose-mode local --duration 20
```

主机 teleop 的 request/reply/control 端口 28701/28702/28703 必须绑定可达地址，并仅在受信网络开放。
确认只读运行正常后，去掉 duration、加 `--actuate`。若板载 PyTorch 不满足时延，使用下述 TensorRT。

## 加入 Vive（主机/本体推理都适用）

SteamVR 始终在连接 Tracker 的主机上运行。启动单 pelvis 服务：

```bash
sim2real/.venv/bin/python sim2real/scripts/serve_scalebfm_vive.py \
  --config sim2real/config/g1/omnicontact_vive_dual.json --robot a \
  --bind tcp://0.0.0.0:28710
```

在对应 inference 命令中改为：

```bash
--pose-mode vive --vive-endpoint tcp://<运行SteamVR的主机IP>:28710
```

主机同机运行可用 `127.0.0.1`，并将服务 bind 改为 localhost。
服务在收到请求后读取有效 Tracker，客户端以 RTT 为年龄上界；断线超过 100 ms 停止，不复用过期位姿。
服务没有网络鉴权，只能在受信专网运行。

## 操作顺序与停止

只有加了 `--actuate` 才执行以下动作：

1. G1 手柄 Start：从实测关节位置用 3 秒插值到 ScaleBFM metadata 默认姿态。
2. 插值完成后 G1 A：启动 ScaleBFM，以固定站立参考维持姿态。
3. 再次按 Pico A（右手 `right_key_one`）：对齐并开始跟随。启动时已按住的按钮不视为新的启动沿。
4. Pico X（左手 `left_key_one`）：冻结当前参考，策略仍保持运行；它**不是急停**。
5. G1 Select/stop 或 Ctrl-C：退出，尝试发送阻尼。异常同样退出，不自动重新使能。

控制包含关节范围、每周期 target 变化、基于实测 q/dq 的 PD 力矩边界交集校验。
历史 action 使用实际限幅后的目标反算，而不是未执行的网络原始输出。
状态/Pico/Vive 过期、计算超时、IMU 倾角异常等均拒绝继续发送目标；bridge watchdog 是进程卡死后的另一层保护。
冻结参考后不需要 Pico 持续在线，机器人仍依靠自身状态与选定定位保持参考；退出请用 G1 stop/进程停止。

## 单模型 TensorRT

主机需要额外安装 `onnx` 才能导出。新目录只绑定上述四个源文件 hash，不绑定双机 side 或 NPZ。

```bash
sim2real/.venv/bin/python sim2real/scripts/export_scalebfm_tracking.py \
  --config sim2real/config/g1/tracking_scalebfm.yaml --output <导出目录>
```

复制导出目录到 G1 后，在**目标板**使用它的系统 TensorRT 编译（engine 不应跨设备直接复用）：

```bash
/usr/bin/python3 sim2real/scripts/build_onboard_tensorrt.py --single --directory <导出目录>
bash sim2real/scripts/run_scalebfm_tracking.sh --check \
  --trt-directory <导出目录> --trt-python /usr/bin/python3
```

实际运行继续带 `--trt-directory`，主进程 `--device cpu`。worker 处理 GPU；校验版本、源文件/engine hash、输入输出形状，限制请求时间。
纯 PyTorch CUDA 则使用 `--device cuda`，不要与 worker 参数混用。

## 验证边界

已提供离线测试覆盖关节重排、延迟窗口/固定对齐、接触里程计、Vive 变换和 RTT 过期、
Pico 重复帧/按钮沿、限幅交集、observe-only 无命令、故障阻尼和单模型 worker。
尚未完成本体 TensorRT 实测、真机稳定性、接触估计漂移或长时间端到端时延验证；不得视为已通过上机验收。

本次主机 CPU 离线检查（100 次、预热后、单线程 PyTorch）：六帧参考构造 + FK + 策略的
p50 约 6.22 ms、p95 约 6.98 ms。ONNX CPU 与 PyTorch 三组随机输入最大绝对差约 `3.22e-6`。
这些不是网络到执行器的端到端时延。

本次全量回归：198 passed、9 skipped、3 failed；新增单机/单模型专项均通过。
三个既有失败：`test_default_dual_network_config_matches_bridge_endpoints`
写死 `enp10s0`，与当前配置网卡不同；`test_two_onboard_processes_complete_and_stop_on_vive_loss`
两个参数分支触发 `control_scheduler_stall`。在临时目录恢复迁移前公共模块后复现了相同失败，
未修改用户网卡配置或放宽旧双机调度保护。

```bash
PYTHONPATH=sim2real/src:sim2real sim2real/.venv/bin/python -m pytest -q sim2real/tests
```
