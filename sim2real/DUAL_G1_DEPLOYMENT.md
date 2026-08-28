# 双 G1 同时部署与逐环节验收手册（vive 分支）

本文用于一台推理主机通过两个独立网口控制两台 Unitree G1。两台机器人均保留默认地址 `192.168.123.164`，因此每个物理网口和对应 bridge 必须位于独立 Linux network namespace。

本文先验收当前已经实现的“双机独立 PD 控制最小闭环”，再说明正式 Dual ScaleBFM/OmniContact policy 上机前必须增加的验收环节。不得跳过前序阶段直接运行双机 learned policy。

## 1. 当前能力边界

当前可执行链路为：

```text
主 namespace                               network namespace

deploy_dual_omnicontact.py
  ├─ RobotSession A
  │   10.201.1.1:55001 <---------------- bridge A (g1a)
  │   10.201.1.2:55002 ----------------> enp10s0 -> G1 A
  │
  └─ RobotSession B
      10.201.2.1:55101 <---------------- bridge B (g1b)
      10.201.2.2:55102 ----------------> enp11s0 -> G1 B
```

最小验证策略使用第一次收到的关节状态作为固定基准：

```text
G1 A: left_elbow_joint = initial_q[21] + 0.03 rad
G1 B: left_elbow_joint = initial_q[21] - 0.03 rad
```

目标不会随每一帧测量值累积。单周期变化限制为 `0.005 rad`，PD 增益来自 `config/g1/controller.yaml`。

当前尚未完成：

- Dual ScaleBFM/OmniContact 联合 observation 和 58 维 action adapter；
- 三 Tracker（G1 A、G1 B、物体）的实际 `DualPoseProvider`；
- 双机 default-pose、standing、task 状态机；
- 双机实机 DDS 隔离与机械动作验证。

因此第 2～14 节是当前可执行的独立控制验收；第 15 节是正式双机 policy 的准入条件。

## 2. 安全规则

从启动 `g1_udp_bridge` 开始就按有电机副作用处理：

- bridge 配置会调用 Unitree `MotionSwitcherClient.ReleaseMode()`；
- bridge 退出时会发送 damping；
- coordinator 即使使用 `--act-robot none`，也会向两侧发送 `enable=0` 的 damping/hold 命令；
- 收到第一条有效 UDP 命令后，如果 200 ms 内没有后续命令，bridge watchdog 会发送 damping 并永久锁存，必须重启 bridge 才能重新接收命令。

每次实机测试必须满足：

1. 两台机器人都有独立机械支撑或悬吊，脚不能承重站立。
2. A、B 贴有清晰标签，网线、遥控器、日志终端使用同一标签。
3. 一人操作推理主机，一人负责两台机器人的急停和机械观察。
4. 两台遥控器/急停均已验证；操作员能在一秒内切断电机输出。
5. 周围没有人员、硬物、线缆缠绕或可能被手臂撞击的物体。
6. 任一输出、姿态、网口或机器人身份不确定时，立即停止。

## 3. 固定本次测试参数

| 项目 | G1 A | G1 B |
| --- | --- | --- |
| 机器人 IP | `192.168.123.164` | `192.168.123.164` |
| 主机物理网口 | `enp10s0` | `enp11s0` |
| network namespace | `g1a` | `g1b` |
| namespace 内主机 IP | `192.168.123.201/24` | `192.168.123.201/24` |
| 主 namespace veth | `10.201.1.1/24` | `10.201.2.1/24` |
| namespace veth | `10.201.1.2/24` | `10.201.2.2/24` |
| 状态 UDP 端口 | `55001` | `55101` |
| 命令 UDP 端口 | `55002` | `55102` |
| 状态镜像端口 | `55003` | `55103` |

这些是 [config/g1/dual_network.yaml](config/g1/dual_network.yaml) 中的默认值。实机网口名称
不同只修改该文件的 `robot_a.interface`、`robot_b.interface` 和对应 `expected_mac`，不要再逐条
修改命令或启动脚本。bridge 启动前会同时核对接口名和 MAC，防止 A/B 网口被交换。
读取器还会检查 veth 地址与两个 bridge YAML 的 UDP 地址一致，避免出现两套配置源。

## 4. 阶段 0：代码与软件环境

```bash
cd /home/yiranwang/TeleHuman/motion_tracking
git branch --show-current
git log -1 --oneline
git status --short
```

放行条件：分支是 `vive`；目标基线为已核对的 `2e83390` 或其明确后续提交；工作区修改来源清楚。

安装依赖并测试：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
uv sync --extra vive
PYTHONPATH=src uv run pytest -q tests/test_dual_runtime.py
PYTHONPATH=src:. uv run pytest -q tests/test_omnicontact_vive.py
```

预期至少为：

```text
test_dual_runtime.py: 4 passed
test_omnicontact_vive.py: 5 passed
```

构建 bridge：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/g1_sim2real
bash scripts/build.sh
test -x build/g1_udp_bridge
```

任何测试或构建失败都不得进入实机阶段。

## 5. 阶段 1：确认物理网口身份

先不连接机器人：

```bash
ip -br link
ip -br addr
ethtool -i enp10s0
ethtool -i enp11s0
```

分别插拔 A、B 网线，确认：

```text
只插拔 G1 A 网线 -> 只有 enp10s0 的 LOWER_UP/NO-CARRIER 变化
只插拔 G1 B 网线 -> 只有 enp11s0 的 LOWER_UP/NO-CARRIER 变化
```

物理接口、机器人标签和线缆必须一一对应。

## 6. 阶段 2：创建同 IP 隔离网络

先校验配置并预览拓扑；该命令不会修改网络：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
bash scripts/setup_dual_network.sh
```

关闭正在占用这两个接口的连接配置。若 NetworkManager 自动管理接口，应先确认连接名再断开，不能删除未知 connection profile。

确认预览中的 A/B 网口和接线一致后，一条命令创建或修复整个拓扑：

```bash
sudo bash scripts/setup_dual_network.sh --apply
sudo bash scripts/setup_dual_network.sh --check
```

`--apply` 可重复执行，不会删除或替换来源不明的半残 veth；遇到不完整拓扑会停止并要求人工检查。
namespace 在重启后消失，因此每次主机重启后重新执行一次 `--apply`。

检查：

```bash
ip -br addr show veth-g1a veth-g1b
sudo ip netns exec g1a ip -br addr
sudo ip netns exec g1b ip -br addr
sudo ip netns exec g1a ip route
sudo ip netns exec g1b ip route
```

放行条件：

```text
主 namespace 看到 10.201.1.1 和 10.201.2.1
g1a 看到 192.168.123.201 和 10.201.1.2
g1b 看到 192.168.123.201 和 10.201.2.2
g1a/g1b 各自有独立的 192.168.123.0/24 直连路由
```

不要在主 namespace 添加到 `192.168.123.164` 的路由，也不要在物理接口之间创建 bridge/bond。

## 7. 阶段 3：IP、ARP 与断线隔离

打开两台机器人网络，暂不启动 bridge：

```bash
sudo ip netns exec g1a ping -c 5 -I enp10s0 192.168.123.164
sudo ip netns exec g1b ping -c 5 -I enp11s0 192.168.123.164
sudo ip netns exec g1a ip neigh show dev enp10s0
sudo ip netns exec g1b ip neigh show dev enp11s0
```

两个 namespace 应分别得到对应机器人的 ARP/MAC。

断线测试：

1. 分别持续运行 A、B ping。
2. 只拔 A 网线；A ping 中断，B 必须继续。
3. 恢复 A 后只拔 B；B ping 中断，A 必须继续。

任意一次断线影响了错误对象或同时影响两侧，都不能放行。

## 8. 阶段 4：配置一致性

```bash
cd /home/yiranwang/TeleHuman/motion_tracking
sed -n '1,40p' g1_sim2real/config/bridge_omnicontact_a.yaml
sed -n '1,40p' g1_sim2real/config/bridge_omnicontact_b.yaml
sed -n '1,80p' sim2real/config/g1/dual_network.yaml
sed -n '1,50p' sim2real/config/g1/dual_omnicontact.yaml
```

必须满足：

```text
bridge A state -> 10.201.1.1:55001 = coordinator A state bind
coordinator A cmd -> 10.201.1.2:55002 = bridge A cmd bind
bridge B state -> 10.201.2.1:55101 = coordinator B state bind
coordinator B cmd -> 10.201.2.2:55102 = bridge B cmd bind
```

同时确认 A/B 配置为：

```yaml
command_timeout_s: 0.20
release_motion_service: true
release_required: true
```

这表示下一阶段启动 bridge 会释放运动服务，必须先完成机械支撑和急停准备。

## 9. 阶段 5：启动两个 bridge

先支撑两台机器人。推荐在一个终端同时启动，并让任一 bridge 退出时联动停止另一侧：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
bash scripts/run_dual_bridges.sh
```

需要分别观察或调试时，也可以使用两个终端：

```bash
bash scripts/run_dual_bridge_a.sh
bash scripts/run_dual_bridge_b.sh
```

这三个脚本都自动读取 `dual_network.yaml`，日常命令不再传网口或 namespace。旧的
`G1_NET_A/B`、`G1_NETNS_A/B` 只保留用于一次性的诊断覆盖。

日志必须分别确认：

- `network_interface` 正确；
- motion service 已释放或原本已 released；
- 收到合法 `lowstate` 和 `mode_machine`；
- A 端口为 `55001/55002`，B 为 `55101/55102`；
- 状态约 50 Hz；
- CRC、fatal tick sync、DDS discovery、state send error 没有持续增长。

诊断：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
bash scripts/diagnostics_dual_network.sh
ss -lunp | rg ':(55001|55101)\b'
sudo ip netns exec g1a ss -lunp | rg ':55002\b'
sudo ip netns exec g1b ss -lunp | rg ':55102\b'
```

抓 DDS 流量必须进入 namespace：

```bash
sudo ip netns exec g1a tcpdump -ni enp10s0 udp
sudo ip netns exec g1b tcpdump -ni enp11s0 udp
```

物理接口已不在主 namespace，不能直接运行 `tcpdump -i enp10s0`。

## 10. 阶段 6：双机未使能命令

保持机械支撑：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
bash scripts/run_dual_deploy.sh --duration 2 --act-robot none
```

该模式向 A/B 连续发送 `enable=0` 的 damping/hold，不发送测试关节偏置。检查：

- coordinator 同时收到两路 state；
- 运行约 2 秒后正常退出；
- A/B command rate 都接近 50 Hz；
- 两台机器人无主动位置动作；
- 退出约 200 ms 后，两个 bridge watchdog 锁存 damping。

完成后必须 `Ctrl-C` 停止并重新启动两个 bridge。锁存后的进程不能用于下一阶段。

## 11. 阶段 7：只使能 G1 A

重新启动 A/B bridge：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
bash scripts/run_dual_deploy.sh \
  --duration 2 \
  --act-robot a \
  --confirm-actuation ENABLE_MOTORS
```

预期：

```text
G1 A left_elbow_joint：相对开始时固定基准增加约 0.03 rad
G1 B：enable=0 damping/hold，不产生对应位置动作
```

必须检查动作对象为 A、关节为左肘、方向正确、目标不累积漂移、B 无跟随或抖动。任一项错误立即急停。

完成后停止并重启两个 bridge。

## 12. 阶段 8：只使能 G1 B

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
bash scripts/run_dual_deploy.sh \
  --duration 2 \
  --act-robot b \
  --confirm-actuation ENABLE_MOTORS
```

预期：

```text
G1 A：enable=0 damping/hold，不产生对应位置动作
G1 B left_elbow_joint：相对开始时固定基准减少约 0.03 rad
```

只有 B 左肘执行负方向小偏置才可放行。完成后再次停止并重启两个 bridge。

## 13. 阶段 9：双机同时接收不同目标

A-only 和 B-only 均通过后：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
bash scripts/run_dual_deploy.sh \
  --duration 2 \
  --act-robot both \
  --confirm-actuation ENABLE_MOTORS
```

预期：

```text
G1 A 左肘：initial + 0.03 rad
G1 B 左肘：initial - 0.03 rad
```

两侧应在同一轮测试中方向相反且互不串线。保存 A/B bridge 日志、coordinator 输出、带 A/B 标签的视频和两侧抓包：

```bash
sudo ip netns exec g1a tcpdump -ni enp10s0 udp -w /tmp/dual_g1_a.pcap
sudo ip netns exec g1b tcpdump -ni enp11s0 udp -w /tmp/dual_g1_b.pcap
```

出现持续漂移、振荡、身份反转或非左肘动作时立即停止。

## 14. 阶段 10：故障注入与 fail-closed

故障测试仍需机械支撑，每个用例前重启两个 bridge。

### 14.1 拔掉 A 网线

在 `--act-robot both` 期间拔 A：

- A lowstate 消失；
- A bridge 最终 watchdog damping/锁存；
- coordinator 报 `missing_or_skewed_bridge_state` 并退出；
- B 收到 `enable=0` hold/damping，不能继续测试目标。

### 14.2 拔掉 B 网线

结果必须与 A 对称，A 不能继续独立运动。

### 14.3 中断 coordinator

运行中按 `Ctrl-C`：coordinator 的 `finally` 向两侧请求 `enable=0` hold，随后 watchdog 锁存 damping。两台机器人都不能继续原命令。

### 14.4 停止一个 bridge

只停止 bridge A，coordinator 应因 A state 超时退出并使 B fail-closed。交换 A/B 重复。

### 14.5 状态不同步

当 A/B state arrival skew 超过 `0.05 s` 时，coordinator 必须拒绝输出并双侧 fail-closed。

以上全部通过后，才证明最小的“同一主机独立寻址并同时控制两台同 IP G1”链路成立。

## 15. 正式 Dual policy 上机准入

当前最小入口不能直接运行 Dual ScaleBFM。正式 adapter 完成后还必须通过以下检查。

### 15.1 模型契约

`policy_metadata.json` 必须填写并强校验：模型文件/哈希、observation shape/顺序、58 维 action 布局、A/B 29 DoF joint order、absolute/residual 类型、default pose、action scale、kp/kd、控制频率和 history 长度。

必须有测试分别给 A、B action 注入唯一值，证明不会映射到另一台机器人。

### 15.2 三 Tracker Vive

正式协作 policy 至少需要：

```text
robot_a_tracker_serial
robot_b_tracker_serial
object_tracker_serial
```

一个 OpenVR reader 必须在同一采样循环原子发布三者。分别完成世界坐标、A/B tracker-to-pelvis、object tracker-to-object 标定；静止记录 30 秒抖动和丢帧；遮挡任意 Tracker 时双机 fail-closed。

实测完成前，`omnicontact_vive_dual.json` 的 `calibration_confirmed` 必须保持 `false`。

### 15.3 无电机 dry-run

正式 policy 首次连接只计算记录，不发送 `enable=1`。验证 observation/action 有限、joint mapping 正确、推理稳定达到 50 Hz、command gap 远小于 200 ms、pose/state age 和 A/B skew 合格、target 与 target delta 均通过 limiter。

### 15.4 支撑状态与落地顺序

正式 policy 仍然先 A-only、再 B-only、最后 both。支撑测试全部通过后依次推进：

1. A 单机地面站立，B 禁用；
2. B 单机地面站立，A 禁用；
3. 双机无物体站立；
4. 双机接触空载物体；
5. 极低速度、极短协作轨迹；
6. 逐步增加轨迹和负载。

每一级都重新测试 Tracker 遮挡、网线断开、bridge 崩溃和急停。

## 16. 日常启动顺序

1. 检查 A/B 标签、支撑、急停和区域。
2. 执行 `sudo bash scripts/setup_dual_network.sh --apply` 创建/检查两个 namespace 和 veth。
3. 分别 ping 两台 `192.168.123.164`。
4. 执行 `bash scripts/run_dual_bridges.sh`，核对 A/B 接口、端口和 lowstate。
5. 确认任一 bridge 退出会联动停止另一侧。
6. 检查两路错误计数。
7. 正式 policy 场景再启动 Vive 三 Tracker provider。
8. 启动 coordinator dry-run。
9. 操作员和安全员确认后才允许 `enable=1`。
10. 结束时先停 coordinator，确认两侧 damping，再停 bridge。

## 17. 清理 network namespace

先停止 coordinator 和两个 bridge，确认接口未被占用：

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
sudo bash scripts/setup_dual_network.sh --teardown
```

若接口由 NetworkManager 管理，按现场原配置恢复，不删除未知 profile。

## 18. 验收记录

| 项目 | 结果 |
| --- | --- |
| `vive` commit | 记录 |
| A/B 机器人序列号 | 记录 |
| A/B 网卡名和 MAC | 记录 |
| A-only 小偏置 | 通过/失败 |
| B-only 小偏置 | 通过/失败 |
| 双机同时相反目标 | 通过/失败 |
| 拔 A 网线 fail-closed | 通过/失败 |
| 拔 B 网线 fail-closed | 通过/失败 |
| 停止 A/B bridge | 通过/失败 |
| coordinator 中断 | 通过/失败 |
| tcpdump 与日志 | 路径 |
| 操作员/安全员 | 记录 |

更换机器人、网卡、线缆、bridge、joint mapping 或 policy 后，应至少重新执行网络隔离、A-only、B-only、双机同时控制和断线 fail-closed 测试。
