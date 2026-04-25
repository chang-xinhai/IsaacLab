# Isaac-Ant-Direct-v0 手搓 PPO Pipeline 完整讲稿

> 配套代码：`source/isaaclab_tasks/isaaclab_tasks/direct/ant_RL_codex.py`
> 原始任务：`python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Ant-Direct-v0`
> 讲解目标：用一个 standalone 文件完整解释 IsaacLab Ant Direct RL 的仿真、环境、观测、奖励、rollout、GAE、PPO 更新全过程。

---

## 目录

1. [本讲稿的目标](#1-本讲稿的目标)
2. [从原版 IsaacLab 到手搓版的对应关系](#2-从原版-isaaclab-到手搓版的对应关系)
3. [整体 Pipeline 总览](#3-整体-pipeline-总览)
4. [Ant 任务的数学抽象](#4-ant-任务的数学抽象)
5. [Isaac Sim / IsaacLab 仿真层](#5-isaac-sim--isaaclab-仿真层)
6. [环境状态、动作和观测](#6-环境状态动作和观测)
7. [reset 流程](#7-reset-流程)
8. [step 流程：从 policy action 到 physics integration](#8-step-流程从-policy-action-到-physics-integration)
9. [Reward 设计与数学解释](#9-reward-设计与数学解释)
10. [Done / Termination / Auto Reset](#10-done--termination--auto-reset)
11. [Actor-Critic Policy](#11-actor-critic-policy)
12. [Rollout 数据结构](#12-rollout-数据结构)
13. [GAE：Generalized Advantage Estimation](#13-gaegeneralized-advantage-estimation)
14. [PPO：Clipped Policy Optimization](#14-ppoclipped-policy-optimization)
15. [一次完整训练 iteration 展开](#15-一次完整训练-iteration-展开)
16. [关于前向运动学、动力学、积分和 IK 的说明](#16-关于前向运动学动力学积分和-ik-的说明)
17. [会议讲解建议顺序](#17-会议讲解建议顺序)
18. [常见问题与回答](#18-常见问题与回答)

---

# 1. 本讲稿的目标

IsaacLab 官方训练命令是：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Ant-Direct-v0
```

官方链路中包含很多框架层：

- Gym registry
- Hydra config
- `DirectRLEnv`
- `RslRlVecEnvWrapper`
- RSL-RL `OnPolicyRunner`
- PPO storage / optimizer

这些框架非常适合工程使用，但第一次理解时不够直观。

所以我们写了一个手搓版：

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/ant_RL_codex.py \
    --num_envs 64 \
    --max_iterations 10 \
    --headless
```

这个文件只使用 IsaacLab / IsaacSim 的：

- `AppLauncher`
- `SimulationContext`
- `InteractiveScene`
- `Articulation`
- `ANT_CFG`

其余全部手写：

- Ant task config
- reset
- step
- observation
- reward
- done
- actor-critic
- rollout storage
- GAE
- PPO update
- training loop

目标是把 Isaac-Ant-Direct-v0 的强化学习过程解释成：

\[
\text{observation} \rightarrow \text{policy} \rightarrow \text{action} \rightarrow \text{physics} \rightarrow \text{reward/done} \rightarrow \text{PPO update}
\]

---

# 2. 从原版 IsaacLab 到手搓版的对应关系

| 原版 IsaacLab / RSL-RL | 手搓版 `ant_RL_codex.py` | 作用 |
|---|---|---|
| `train.py` | `train()` | 训练入口 |
| `gym.make("Isaac-Ant-Direct-v0")` | `HandWrittenAntEnv(...)` | 创建环境 |
| `AntEnvCfg` | `AntTaskCfg` | Ant 任务参数 |
| `AntSceneCfg / LocomotionEnv._setup_scene()` | `AntSceneCfg` | 地面、灯光、Ant 资产 |
| `DirectRLEnv.reset()` | `HandWrittenAntEnv.reset()` | 环境重置 |
| `DirectRLEnv.step()` | `HandWrittenAntEnv.step()` | 环境单步推进 |
| `_get_observations()` | `compute_observations()` | 构造 policy 输入 |
| `_get_rewards()` | `compute_rewards()` | 计算 reward |
| `_get_dones()` | `compute_dones()` | 判断终止 |
| `RslRlVecEnvWrapper` | 无，直接 tensor 接口 | 格式适配在这里省略 |
| `OnPolicyRunner.learn()` | `train()` 主循环 | rollout + PPO update |
| `ActorCritic` from RSL-RL | `ActorCritic` | policy/value 网络 |
| `RolloutStorage` from RSL-RL | `RolloutStorage` | 存储 rollout 数据 |
| `PPO` from RSL-RL | `PPO` | 优化 actor-critic |

---

# 3. 整体 Pipeline 总览

## 3.1 高层流程

```text
启动脚本 ant_RL_codex.py
        |
        v
AppLauncher 启动 Isaac Sim
        |
        v
SimulationContext 创建物理仿真上下文
        |
        v
InteractiveScene 创建并复制 N 个 Ant 环境
        |
        v
HandWrittenAntEnv.reset()
        |
        v
for iteration in max_iterations:
    rollout N_envs * T_steps transitions
    compute GAE advantages / returns
    PPO update actor-critic
    save checkpoint
```

数学上可以把整个训练看成一个交替过程：

\[
\theta_k \xrightarrow{\text{rollout}} \mathcal{D}_k
\xrightarrow{\text{PPO update}} \theta_{k+1}
\]

其中：

- \(\theta_k\)：第 \(k\) 轮 policy 参数
- \(\mathcal{D}_k\)：用当前 policy 收集的 on-policy 数据
- \(\theta_{k+1}\)：PPO 更新后的 policy 参数

---

## 3.2 一次 RL step 的流程

```text
obs_t [N, 36]
    |
    v
Actor-Critic policy
    |
    v
action_t [N, 8]
    |
    v
joint_effort = 0.5 * 15 * action_t
    |
    v
physics step x 2
    |
    v
read root pose / velocity / joint state
    |
    v
reward_t, done_t, obs_{t+1}
```

其中：

- \(N\)：并行环境数量，默认 4096
- observation 维度：36
- action 维度：8
- `decimation = 2`，即一个 RL step 内执行 2 个 physics step

---

# 4. Ant 任务的数学抽象

强化学习环境通常抽象为 MDP：

\[
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, r, \gamma)
\]

对于 Ant 任务：

| 符号 | 含义 | 在代码中 |
|---|---|---|
| \(s_t\) | 仿真真实状态 | root pose, joint pos, velocity 等 |
| \(o_t\) | policy 可见观测 | `obs`，36 维 |
| \(a_t\) | policy 输出动作 | `actions`，8 维 |
| \(P(s_{t+1}\mid s_t,a_t)\) | 物理转移 | Isaac Sim physics integration |
| \(r_t\) | reward | `compute_rewards()` |
| \(\gamma\) | 折扣因子 | `gamma = 0.99` |

注意：policy 不直接看到完整物理状态 \(s_t\)，而是看到观测 \(o_t\)：

\[
o_t = h(s_t, a_{t-1})
\]

在代码中，`obs` 还包含上一时刻 action：

\[
o_t = [\text{height}, v_{local}, \omega_{local}, \text{yaw}, \text{roll}, \dots, q_{joint}, \dot q_{joint}, a_{t-1}]
\]

---

# 5. Isaac Sim / IsaacLab 仿真层

## 5.1 SimulationContext

代码：

```python
sim_cfg = SimulationCfg(dt=cfg.sim_dt, render_interval=cfg.decimation, device=device)
self.sim = SimulationContext(sim_cfg)
```

这里创建物理仿真上下文。

关键参数：

```python
sim_dt = 1 / 120
```

物理步长为：

\[
\Delta t_{phys} = \frac{1}{120}\ \text{s}
\]

---

## 5.2 RL step 与 physics step

Ant 中：

```python
decimation = 2
```

所以：

\[
\Delta t_{RL} = \text{decimation} \cdot \Delta t_{phys}
= 2 \cdot \frac{1}{120}
= \frac{1}{60}\ \text{s}
\]

也就是说：

- physics 以 120 Hz 跑
- policy 以 60 Hz 控制

代码中表现为：

```python
for _ in range(self.cfg.decimation):
    set_joint_effort_target(...)
    sim.step()
```

---

## 5.3 InteractiveScene

代码：

```python
scene_cfg = AntSceneCfg(num_envs=num_envs, env_spacing=cfg.env_spacing, replicate_physics=True)
self.scene = InteractiveScene(scene_cfg)
self.robot = self.scene["robot"]
```

这里完成：

1. 创建地面
2. 创建灯光
3. 加载 Ant USD asset
4. 复制出 \(N\) 个并行环境

如果：

```python
num_envs = 4096
```

则同时仿真 4096 只 Ant。

这也是 IsaacLab RL 高吞吐的核心。

---

# 6. 环境状态、动作和观测

## 6.1 真实物理状态

Ant 的真实仿真状态可以抽象为：

\[
s_t = (x_t, R_t, v_t, \omega_t, q_t, \dot q_t)
\]

其中：

| 符号 | 含义 | IsaacLab tensor |
|---|---|---|
| \(x_t \in \mathbb{R}^3\) | root position | `root_pos_w` |
| \(R_t \in SO(3)\) | root orientation | `root_quat_w` |
| \(v_t \in \mathbb{R}^3\) | root linear velocity | `root_lin_vel_w` |
| \(\omega_t \in \mathbb{R}^3\) | root angular velocity | `root_ang_vel_w` |
| \(q_t \in \mathbb{R}^8\) | joint positions | `joint_pos` |
| \(\dot q_t \in \mathbb{R}^8\) | joint velocities | `joint_vel` |

---

## 6.2 动作空间

Ant 有 8 个可控关节：

\[
a_t \in \mathbb{R}^8
\]

代码：

```python
action_dim = 8
```

policy 输出的 action 不直接是位置目标，而是转成关节 effort：

```python
joint_efforts = action_scale * joint_gears * actions
```

Ant 参数：

```python
action_scale = 0.5
joint_gears = 15
```

所以：

\[
\tau_t = 0.5 \cdot 15 \cdot a_t = 7.5 a_t
\]

其中：

- \(a_t\)：policy 输出动作
- \(\tau_t\)：实际施加到关节上的 effort / torque-like command

---

## 6.3 观测空间：36 维

代码：

```python
obs_dim = 36
```

观测由以下部分拼接：

| 项 | 维度 | 数学记号 | 含义 |
|---|---:|---|---|
| torso height | 1 | \(z_t\) | root z 高度 |
| local linear velocity | 3 | \(R_t^\top v_t\) | 局部坐标线速度 |
| local angular velocity | 3 | \(R_t^\top \omega_t\) | 局部坐标角速度 |
| yaw | 1 | \(\psi_t\) | 偏航角 |
| roll | 1 | \(\phi_t\) | 横滚角 |
| angle to target | 1 | \(\alpha_t\) | 朝向目标的相对角 |
| up projection | 1 | \(u_t\) | 身体直立程度 |
| heading projection | 1 | \(h_t\) | 朝目标方向程度 |
| joint positions scaled | 8 | \(\tilde q_t\) | 归一化关节位置 |
| joint velocities scaled | 8 | \(0.2\dot q_t\) | 缩放关节速度 |
| previous/current actions | 8 | \(a_{t-1}\) | 上一次动作 |
| **总计** | **36** |  |  |

即：

\[
o_t =
\left[
 z_t,
 R_t^\top v_t,
 R_t^\top \omega_t,
 \psi_t,
 \phi_t,
 \alpha_t,
 u_t,
 h_t,
 \tilde q_t,
 0.2\dot q_t,
 a_{t-1}
\right]
\]

---

# 7. reset 流程

## 7.1 reset 的目的

reset 要做的是把某些 Ant 放回默认初始状态。

数学上：

\[
s_t \leftarrow s_0
\]

代码中包括：

```python
root_state = default_root_state
joint_pos = default_joint_pos
joint_vel = default_joint_vel
```

然后写回仿真：

```python
write_root_pose_to_sim(...)
write_root_velocity_to_sim(...)
write_joint_state_to_sim(...)
```

---

## 7.2 env origin

并行环境不是都放在世界原点，而是每个环境有自己的 origin：

```python
root_state[:, :3] += self.scene.env_origins[env_ids]
```

也就是：

\[
x_0^{(i)} = x_{default} + o^{(i)}
\]

其中：

- \(i\)：第 \(i\) 个并行环境
- \(o^{(i)}\)：该环境在世界坐标中的 offset

---

## 7.3 potential 初始化

Ant 的 progress reward 用 potential 差值：

\[
\Phi_t = -\frac{\lVert p_{target} - p_t \rVert}{\Delta t_{phys}}
\]

reset 时需要初始化：

```python
self.potentials[env_ids] = -norm(to_target) / sim_dt
self.prev_potentials[env_ids] = self.potentials[env_ids]
```

否则新 episode 会继承旧 episode 的 progress 信息，reward 会错。

---

# 8. step 流程：从 policy action 到 physics integration

## 8.1 step 总流程

```text
actions [N, 8]
    |
    v
缓存 self.actions
    |
    v
for decimation=2:
    joint_efforts = 7.5 * actions
    robot.set_joint_effort_target(joint_efforts)
    scene.write_data_to_sim()
    sim.step()
    scene.update()
    |
    v
compute_dones()
compute_rewards()
reset done envs
compute_observations()
```

---

## 8.2 动力学视角

Ant 是一个浮动基座多刚体系统，可以抽象成广义坐标：

\[
q = (x_{base}, R_{base}, q_{joint})
\]

广义速度：

\[
\dot q = (v_{base}, \omega_{base}, \dot q_{joint})
\]

刚体动力学形式可写为：

\[
M(q)\ddot q + C(q, \dot q)\dot q + g(q) = S^\top \tau + J_c(q)^\top \lambda
\]

其中：

| 符号 | 含义 |
|---|---|
| \(M(q)\) | 质量矩阵 |
| \(C(q,\dot q)\dot q\) | 科氏/离心项 |
| \(g(q)\) | 重力项 |
| \(\tau\) | 关节 effort 输入 |
| \(J_c^\top \lambda\) | 接触约束力，例如脚和地面接触 |
| \(S\) | actuated joints 选择矩阵 |

在我们的代码中，RL 只输出：

\[
\tau_t = 7.5 a_t
\]

真正的 \(M, C, g, J_c, \lambda\) 和积分都由 PhysX / Isaac Sim 处理。

---

## 8.3 数值积分视角

物理仿真器每个 physics step 近似做：

\[
\dot q_{k+1} = \dot q_k + \Delta t_{phys}\, f_{vel}(q_k, \dot q_k, \tau_k)
\]

\[
q_{k+1} = q_k + \Delta t_{phys}\, f_{pos}(q_k, \dot q_{k+1})
\]

真实 PhysX 内部会处理：

- rigid body integration
- articulation constraints
- contact solving
- friction
- collision
- joint limits

对 RL 来说，它只看到一个黑箱转移：

\[
s_{t+1} \sim P(s_{t+1}\mid s_t, a_t)
\]

这里的 \(P\) 是由数值物理仿真器隐式定义的。

---

# 9. Reward 设计与数学解释

总 reward：

\[
r_t = r_{progress} + r_{alive} + r_{up} + r_{heading}
      - c_{action} - c_{energy} - c_{limit}
\]

若摔倒：

\[
r_t = r_{death} = -2
\]

---

## 9.1 progress reward

目标点是每个环境 origin 前方 \(+x\) 方向很远的位置：

\[
p_{target}^{(i)} = o^{(i)} + [1000, 0, 0]^\top
\]

定义 potential：

\[
\Phi_t = -\frac{\lVert p_{target} - p_t \rVert_2}{\Delta t_{phys}}
\]

progress reward：

\[
r_{progress,t} = \Phi_t - \Phi_{t-1}
\]

如果 Ant 靠近目标：

\[
\lVert p_{target} - p_t \rVert < \lVert p_{target} - p_{t-1} \rVert
\]

则：

\[
\Phi_t > \Phi_{t-1}
\]

所以：

\[
r_{progress,t} > 0
\]

直观上：**朝 +x 跑得越快，progress reward 越大。**

---

## 9.2 heading reward

heading projection：

\[
h_t = \hat f_t \cdot \hat d_t
\]

其中：

- \(\hat f_t\)：机器人身体局部 x 轴在世界坐标下的方向
- \(\hat d_t\)：从机器人指向目标的单位向量

reward：

\[
r_{heading} =
\begin{cases}
w_h, & h_t > 0.8 \\
\frac{w_h}{0.8}h_t, & h_t \le 0.8
\end{cases}
\]

代码参数：

```python
heading_weight = 0.5
```

即：朝目标方向越准，奖励越高。

---

## 9.3 up reward

up projection：

\[
u_t = \hat z_{body,t} \cdot \hat z_{world}
\]

reward：

\[
r_{up} =
\begin{cases}
w_u, & u_t > 0.93 \\
0, & \text{otherwise}
\end{cases}
\]

代码参数：

```python
up_weight = 0.1
```

即：身体足够直立才给 bonus。

---

## 9.4 alive reward

只要没摔倒，每步给固定奖励：

\[
r_{alive} = 0.5
\]

作用：鼓励 Ant 延长 episode，不要快速摔倒。

---

## 9.5 action cost

动作幅度惩罚：

\[
c_{action} = \alpha_a \sum_i a_i^2
\]

其中：

```python
actions_cost_scale = 0.005
```

即：

\[
\alpha_a = 0.005
\]

作用：避免 policy 输出过大的关节命令。

---

## 9.6 electricity / energy cost

能耗惩罚：

\[
c_{energy} = \alpha_e \sum_i |a_i \dot q_i s_v| m_i
\]

其中：

- \(a_i\)：第 \(i\) 个动作
- \(\dot q_i\)：第 \(i\) 个关节速度
- \(s_v = 0.2\)：`dof_vel_scale`
- \(m_i\)：`motor_effort_ratio`，这里全为 1
- \(\alpha_e = 0.05\)：`energy_cost_scale`

直觉：动作大且关节速度大，认为耗能更高。

---

## 9.7 joint limit cost

关节接近极限时惩罚：

\[
c_{limit} = \sum_i \mathbf{1}(\tilde q_i > 0.98)
\]

其中 \(\tilde q_i\) 是归一化后的关节位置。

作用：避免关节长期顶在 limit 上。

---

## 9.8 death override

如果：

\[
z_{root} < 0.31
\]

则认为摔倒：

\[
r_t = -2.0
\]

注意这里是 override，而不是在原 reward 上再加一个负数。

---

# 10. Done / Termination / Auto Reset

## 10.1 两类 done

代码中区分：

```python
terminated = root_z < termination_height
truncated = episode_length >= max_episode_length - 1
```

数学上：

\[
d_t^{death} = \mathbf{1}(z_t < 0.31)
\]

\[
d_t^{timeout} = \mathbf{1}(T_t \ge 899)
\]

最终：

\[
d_t = d_t^{death} \lor d_t^{timeout}
\]

---

## 10.2 自动 reset

step 内部做：

```python
reset_env_ids = dones.nonzero(...)
self.reset_idx(reset_env_ids)
```

这意味着：

- 当前 step 返回的 `rewards` 和 `dones` 仍然对应终止前的 transition
- 返回的 `obs` 已经是 reset 后的新 episode 初始 obs

这是很多 vectorized RL env 的常见做法。

---

# 11. Actor-Critic Policy

## 11.1 Gaussian policy

Actor 网络输出 action mean：

\[
\mu_\theta(o_t) \in \mathbb{R}^8
\]

同时有可学习 log standard deviation：

\[
\log \sigma \in \mathbb{R}^8
\]

policy 定义为高斯分布：

\[
\pi_\theta(a_t\mid o_t) = \mathcal{N}(\mu_\theta(o_t), \operatorname{diag}(\sigma^2))
\]

采样动作：

\[
a_t \sim \pi_\theta(\cdot\mid o_t)
\]

代码：

```python
dist = Normal(mean, std)
actions = dist.sample()
log_prob = dist.log_prob(actions).sum(dim=-1)
```

---

## 11.2 Critic

Critic 估计状态价值：

\[
V_\phi(o_t) \approx \mathbb{E}\left[\sum_{k=0}^{\infty}\gamma^k r_{t+k}\right]
\]

代码：

```python
value = self.critic(obs).squeeze(-1)
```

---

## 11.3 网络结构

对齐原始 `AntPPORunnerCfg`：

```python
actor_hidden_dims = [400, 200, 100]
critic_hidden_dims = [400, 200, 100]
activation = "elu"
```

即：

\[
36 \rightarrow 400 \rightarrow 200 \rightarrow 100 \rightarrow 8
\]

用于 actor。

critic：

\[
36 \rightarrow 400 \rightarrow 200 \rightarrow 100 \rightarrow 1
\]

---

# 12. Rollout 数据结构

每个 PPO iteration 收集：

```python
num_steps_per_env = 32
num_envs = 4096
```

transition 数量：

\[
32 \times 4096 = 131072
\]

buffer 形状：

| 名称 | shape | 含义 |
|---|---|---|
| `obs` | `[32, 4096, 36]` | rollout 观测 |
| `actions` | `[32, 4096, 8]` | policy 动作 |
| `rewards` | `[32, 4096]` | reward |
| `dones` | `[32, 4096]` | 是否终止 |
| `values` | `[32, 4096]` | old critic value |
| `log_probs` | `[32, 4096]` | old policy log prob |
| `advantages` | `[32, 4096]` | GAE advantage |
| `returns` | `[32, 4096]` | critic target |

---

# 13. GAE：Generalized Advantage Estimation

## 13.1 TD residual

定义 TD residual：

\[
\delta_t = r_t + \gamma V(o_{t+1})(1-d_t) - V(o_t)
\]

其中：

- \(d_t=1\)：episode 结束，不 bootstrap
- \(d_t=0\)：episode 未结束，可以 bootstrap

代码：

```python
delta = rewards[step] + gamma * next_values * not_done - values[step]
```

---

## 13.2 GAE advantage

GAE 从后往前递推：

\[
A_t = \delta_t + \gamma\lambda(1-d_t)A_{t+1}
\]

代码：

```python
advantage = delta + gamma * lam * not_done * advantage
```

其中：

```python
gamma = 0.99
lam = 0.95
```

直觉：

- \(\gamma\)：未来 reward 折扣
- \(\lambda\)：bias-variance tradeoff

---

## 13.3 Return

critic 的学习目标：

\[
R_t = A_t + V(o_t)
\]

代码：

```python
returns = advantages + values
```

最后对 advantage 做标准化：

\[
\hat A_t = \frac{A_t - \mu_A}{\sigma_A + \epsilon}
\]

这有助于 PPO 优化稳定。

---

# 14. PPO：Clipped Policy Optimization

PPO 是 on-policy actor-critic 方法。

目标：用当前 rollout 数据更新 policy，但不要让新 policy 偏离旧 policy 太远。

---

## 14.1 Probability ratio

定义：

\[
\rho_t(\theta) = \frac{\pi_\theta(a_t\mid o_t)}{\pi_{\theta_{old}}(a_t\mid o_t)}
\]

代码中用 log prob：

```python
ratio = torch.exp(log_probs - old_log_probs)
```

---

## 14.2 Clipped surrogate objective

PPO actor 目标：

\[
L^{CLIP}(\theta) =
\mathbb{E}_t
\left[
\min
\left(
\rho_t(\theta)\hat A_t,
\operatorname{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\hat A_t
\right)
\right]
\]

其中：

```python
clip_param = 0.2
```

即：

\[
\epsilon = 0.2
\]

优化代码中写成 loss，所以加负号：

```python
surrogate = -advantages * ratio
surrogate_clipped = -advantages * clamp(ratio, 0.8, 1.2)
surrogate_loss = max(surrogate, surrogate_clipped).mean()
```

---

## 14.3 Value loss

critic 目标：

\[
L^V(\phi) = \mathbb{E}_t[(V_\phi(o_t) - R_t)^2]
\]

手搓版使用 clipped value loss：

\[
V_{clip} = V_{old} + \operatorname{clip}(V_\phi - V_{old}, -\epsilon, \epsilon)
\]

\[
L^V = \max\left((V_\phi - R)^2, (V_{clip} - R)^2\right)
\]

---

## 14.4 Total loss

总 loss：

\[
L = L^{policy} + c_v L^V - c_e H[\pi]
\]

代码参数：

```python
value_loss_coef = 1.0
entropy_coef = 0.0
```

所以这里实际是：

\[
L = L^{policy} + L^V
\]

---

## 14.5 PPO update 次数

每个 iteration：

```python
num_learning_epochs = 5
num_mini_batches = 4
```

所以 optimizer step 数：

\[
5 \times 4 = 20
\]

也就是同一批 rollout 数据会被重复使用 20 个 minibatch update。

---

# 15. 一次完整训练 iteration 展开

伪代码：

```python
for iteration in range(max_iterations):

    storage.clear()

    # A. Rollout
    for step in range(num_steps_per_env):
        actions, log_probs, values = actor_critic.act(obs)
        next_obs, rewards, dones, extras = env.step(actions)
        storage.add(obs, actions, rewards, dones, values, log_probs)
        obs = next_obs

    # B. Bootstrap last value
    last_values = critic(obs)

    # C. GAE
    storage.compute_returns(last_values, gamma=0.99, lam=0.95)

    # D. PPO update
    ppo.update(storage)

    # E. Save checkpoint
    if iteration % save_interval == 0:
        save()
```

数学上：

\[
\mathcal{D}_k = \{(o_t, a_t, r_t, d_t, \log\pi_{old}(a_t|o_t), V_{old}(o_t))\}
\]

\[
\mathcal{D}_k \rightarrow \hat A_t, R_t \rightarrow \theta_{k+1}
\]

---

# 16. 关于前向运动学、动力学、积分和 IK 的说明

这部分适合给数学、数值积分或机器人背景的听众解释。

---

## 16.1 这里主要是 forward dynamics，不是 inverse kinematics

在这个 Ant RL 任务中，policy 输出的是 joint effort：

\[
a_t \rightarrow \tau_t
\]

然后仿真器求解：

\[
(q_t, \dot q_t, \tau_t) \rightarrow (q_{t+1}, \dot q_{t+1})
\]

这叫 forward dynamics / 正向动力学。

它不是 IK。

IK 通常是：

\[
x_{ee}^{desired} \rightarrow q
\]

即给定末端目标位置，求关节角。

Ant locomotion 中没有显式求 IK。脚如何摆、关节如何协调，都是 policy 通过 reward 学出来的。

---

## 16.2 Forward kinematics 在哪里出现？

虽然我们没有显式写 FK，但仿真器和 articulation 会隐式维护：

\[
q \rightarrow \text{body poses}
\]

例如：

- root pose
- link poses
- collision geometry poses
- contact points

这些都由 Isaac Sim / PhysX articulation 系统计算。

我们的代码只读取结果：

```python
self.robot.data.root_pos_w
self.robot.data.root_quat_w
self.robot.data.joint_pos
self.robot.data.joint_vel
```

---

## 16.3 接触动力学

Ant locomotion 的关键难点是接触：

\[
J_c(q)\dot q = 0
\]

接触力 \(\lambda\) 不是 policy 直接控制的，而是由物理求解器根据碰撞、摩擦和约束求出：

\[
M(q)\ddot q + C(q,\dot q)\dot q + g(q) = S^\top \tau + J_c(q)^\top\lambda
\]

因此，同一个 action 在不同接触状态下会产生不同运动结果。

这也是为什么 locomotion RL 通常需要大量并行采样。

---

## 16.4 数值积分与 RL 的时间尺度

有两个时间尺度：

| 层级 | 时间步长 | 频率 | 作用 |
|---|---:|---:|---|
| physics step | \(1/120\) s | 120 Hz | 解动力学、碰撞、接触 |
| RL step | \(1/60\) s | 60 Hz | policy 输出 action、计算 reward |

因为：

\[
\Delta t_{RL} = 2\Delta t_{phys}
\]

所以每个 action 会保持两个 physics step。

这在控制中类似 zero-order hold：

\[
\tau(t) = \tau_k, \quad t \in [t_k, t_{k+1})
\]

---

# 17. 会议讲解建议顺序

建议按这个顺序讲，比较顺：

1. **讲目标**
   - 我们把官方 Ant Direct RL 拆成一个手搓文件。
2. **讲整体 pipeline**
   - Sim → Env → Policy → Rollout → GAE → PPO。
3. **讲仿真层**
   - `SimulationContext`
   - `InteractiveScene`
   - `ANT_CFG`
4. **讲 Ant 动作**
   - \(a_t \in \mathbb{R}^8\)
   - \(\tau_t = 7.5a_t\)
5. **讲 step**
   - decimation = 2
   - physics integration
   - scene buffer update
6. **讲 observation**
   - 36 维表格
7. **讲 reward**
   - progress / alive / heading / up / penalties
8. **讲 PPO**
   - rollout shape
   - GAE 公式
   - PPO clipped objective
9. **最后对照官方 IsaacLab**
   - 说明手搓代码和官方代码一一对应。

---

# 18. 常见问题与回答

## Q1：为什么目标点是 `[1000, 0, 0]`？

因为这个任务本质是让 Ant 一直朝 +x 方向跑。目标点设很远，可以近似表示“沿 +x 前进”。

---

## Q2：为什么 progress reward 要除以 `sim_dt`？

原始 Ant 环境这样定义：

\[
\Phi = -\frac{\text{distance}}{\Delta t}
\]

这样 potential 差值近似具有“速度奖励”的含义：

\[
\Phi_t - \Phi_{t-1}
\approx
\frac{d_{t-1} - d_t}{\Delta t}
\]

即朝目标方向的前进速度。

---

## Q3：为什么 observation 里包含 action？

这给 policy 一个关于上一控制输入的信息，有助于学习更平滑的控制策略，也与原始 Ant Direct obs 结构保持一致。

---

## Q4：为什么没有 IK？

因为这是 torque/effort control locomotion。policy 不指定脚的位置，也不解关节角，而是直接输出关节 effort，运动模式由 RL 学出来。

---

## Q5：为什么需要 4096 个并行环境？

PPO 是 on-policy，需要大量新鲜 rollout。并行环境可以在一次 rollout 中得到：

\[
4096 \times 32 = 131072
\]

条 transition，大幅提高训练吞吐。

---

## Q6：PPO 的 clipping 在数学上起什么作用？

它限制新旧 policy 的概率比：

\[
\rho_t = \frac{\pi_\theta}{\pi_{old}}
\]

不要离 1 太远。直观上防止一次 update 把 policy 改得过猛。

---

## Q7：为什么 done 后返回的 obs 已经是 reset 后的？

这是 vectorized env 常见设计。当前 transition 的终止信息由 `done=True` 表示，而下一个 obs 可以直接用于新 episode 的下一步 rollout。

---

# 结尾总结

Isaac-Ant-Direct-v0 的核心可以压缩成一句话：

> policy 根据 36 维观测输出 8 维关节动作，动作被缩放成 joint effort 后交给 Isaac Sim 做正向动力学积分；环境从仿真状态构造 reward、done 和下一帧 observation；PPO 用 4096 个并行环境收集到的 rollout 通过 GAE 和 clipped objective 更新 actor-critic。

最重要的链路是：

\[
o_t \xrightarrow{\pi_\theta} a_t
\xrightarrow{\tau = 7.5a_t} \text{PhysX dynamics}
\xrightarrow{} s_{t+1}
\xrightarrow{} (o_{t+1}, r_t, d_t)
\xrightarrow{\text{GAE/PPO}} \theta_{t+1}
\]
