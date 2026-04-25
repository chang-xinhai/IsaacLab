# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""手搓版 Isaac-Ant-Direct-v0 PPO 训练脚本。

这个文件的目标不是追求工程化，而是把 IsaacLab Ant Direct RL 的核心链路放进一个
standalone 文件里，方便逐行讲解：

1. 用 IsaacLab/IsaacSim API 搭建仿真、地面、灯光、4096 个 Ant 资产。
2. 自己实现 Ant 的 reset / step / observation / reward / done。
3. 自己实现 actor-critic policy、rollout buffer、GAE、PPO update。
4. 自己实现完整 training loop。

刻意不使用：
- DirectRLEnv / ManagerBasedEnv
- Gym task registry
- RSL-RL / rl_games / skrl
- IsaacLab reward/observation/action managers

运行示例：

    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/ant_RL_codex.py \
        --num_envs 64 --max_iterations 10 --headless

接近原始 Isaac-Ant-Direct-v0 配置：

    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/ant_RL_codex.py \
        --num_envs 4096 --max_iterations 1000 --headless
"""

from __future__ import annotations

# =============================================================================
# 0. 启动 Isaac Sim：必须先 AppLauncher，再 import 大部分 IsaacLab/torch 运行逻辑
# =============================================================================

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Hand-written Isaac Ant PPO training loop.")
parser.add_argument("--num_envs", type=int, default=4096, help="并行 Ant 环境数量。")
parser.add_argument("--max_iterations", type=int, default=1000, help="PPO 更新轮数。")
parser.add_argument("--num_steps_per_env", type=int, default=32, help="每个 PPO iteration 中每个环境 rollout 几步。")
parser.add_argument("--seed", type=int, default=42, help="随机种子。")
parser.add_argument("--save_interval", type=int, default=50, help="每隔多少个 iteration 存一次 checkpoint。")
parser.add_argument("--log_dir", type=str, default="logs/hand_rl/ant_codex", help="checkpoint 保存目录。")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Sim 启动后，再 import torch / IsaacLab runtime API。
import torch
import torch.nn as nn
from torch.distributions import Normal

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass
from isaaclab_assets.robots.ant import ANT_CFG


# =============================================================================
# 1. Ant task 配置：对应 AntEnvCfg + AntPPORunnerCfg 中最核心的数字
# =============================================================================


@dataclass
class AntTaskCfg:
    """Isaac-Ant-Direct-v0 的任务参数。

    这里把原本散落在 AntEnvCfg / PPO cfg 里的关键参数集中放在一起，便于讲解。
    """

    # -------------------------
    # 环境 / 动作 / 观测配置
    # -------------------------
    episode_length_s: float = 15.0       # 每个 episode 最长 15 秒
    decimation: int = 2                  # 一个 RL step 内跑 2 个 physics step
    sim_dt: float = 1.0 / 120.0          # physics 仿真步长，120Hz
    action_scale: float = 0.5            # action 转 joint effort 前的缩放
    action_dim: int = 8                  # Ant 有 8 个可控关节
    obs_dim: int = 36                    # policy 输入观测维度
    env_spacing: float = 4.0             # 并行环境之间的间距

    # -------------------------
    # reward / done 配置
    # -------------------------
    heading_weight: float = 0.5          # 朝目标方向的奖励权重
    up_weight: float = 0.1               # 保持身体直立的奖励权重
    energy_cost_scale: float = 0.05      # 能耗惩罚系数
    actions_cost_scale: float = 0.005    # 动作幅度惩罚系数
    alive_reward_scale: float = 0.5      # 未摔倒时每步存活奖励
    dof_vel_scale: float = 0.2           # 关节速度缩放系数
    death_cost: float = -2.0             # 摔倒时 reward 覆盖值
    termination_height: float = 0.31     # root 高度低于它则判定摔倒
    angular_velocity_scale: float = 1.0  # 角速度观测缩放系数

    @property
    def rl_dt(self) -> float:
        """一个 RL step 对应的真实仿真时间。Ant 中是 1/120 * 2 = 1/60 秒。"""
        return self.sim_dt * self.decimation

    @property
    def max_episode_length(self) -> int:
        """15 秒 episode 换算成 RL step 数，约 900 步。"""
        return math.ceil(self.episode_length_s / self.rl_dt)


# =============================================================================
# 2. IsaacLab scene 配置：这里只负责资产，不负责 RL 逻辑
# =============================================================================


@configclass
class AntSceneCfg(InteractiveSceneCfg):
    """最小 Ant 场景：地面 + 灯光 + Ant articulation。"""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                restitution_combine_mode="average",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            )
        ),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )

    # ANT_CFG 来自 isaaclab_assets，只使用资产定义，不使用 IsaacLab 的 Ant task/env。
    robot: ArticulationCfg = ANT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# =============================================================================
# 3. 数学小工具：姿态、局部速度、角度归一化等
# =============================================================================


def normalize_angle(x: torch.Tensor) -> torch.Tensor:
    """把角度映射到 [-pi, pi]。"""
    return torch.atan2(torch.sin(x), torch.cos(x))


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """四元数共轭。IsaacLab root_quat_w 使用 [w, x, y, z]。"""
    return torch.cat((q[..., 0:1], -q[..., 1:]), dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """用四元数 q 把向量 v 从局部坐标旋转到世界坐标。"""
    q_vec = q[..., 1:]
    q_w = q[..., 0:1]
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v + q_w * t + torch.cross(q_vec, t, dim=-1)


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """用 q 的逆旋转：把世界坐标向量 v 转到机器人局部坐标。"""
    return quat_rotate(quat_conjugate(q), v)


def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """从四元数提取 yaw。"""
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def roll_from_quat(q: torch.Tensor) -> torch.Tensor:
    """从四元数提取 roll。"""
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))


def scale_to_minus_one_one(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """把关节位置按 soft joint limits 线性缩放到约 [-1, 1]。"""
    return (2.0 * x - upper - lower) / torch.clamp(upper - lower, min=1.0e-6)


# =============================================================================
# 4. 手写 Ant 环境：reset / step / obs / reward / done 全在这里
# =============================================================================


class HandWrittenAntEnv:
    """一个非常薄的 vectorized Ant env。

    它只依赖 IsaacLab 的 SimulationContext / InteractiveScene / Articulation。
    它不继承 Gym，也不继承 DirectRLEnv。接口故意保持简单：

        obs = env.reset()
        next_obs, rewards, dones, extras = env.step(actions)
    """

    def __init__(self, num_envs: int, device: str, cfg: AntTaskCfg):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        # ---------------------------------------------------------------------
        # 4.1 创建 Isaac Sim 物理上下文
        # ---------------------------------------------------------------------
        sim_cfg = SimulationCfg(dt=cfg.sim_dt, render_interval=cfg.decimation, device=device)
        self.sim = SimulationContext(sim_cfg)
        self.sim.set_camera_view([5.0, 0.0, 3.0], [0.0, 0.0, 0.5])

        # ---------------------------------------------------------------------
        # 4.2 创建并复制 num_envs 个 Ant 场景
        # ---------------------------------------------------------------------
        scene_cfg = AntSceneCfg(num_envs=num_envs, env_spacing=cfg.env_spacing, replicate_physics=True)
        self.scene = InteractiveScene(scene_cfg)
        self.robot: Articulation = self.scene["robot"]

        # 激活物理句柄，并让 scene buffer 拿到初始数据。
        self.sim.reset()
        self.scene.update(cfg.sim_dt)

        # ---------------------------------------------------------------------
        # 4.3 准备 Ant task 需要的 buffer
        # ---------------------------------------------------------------------
        self.joint_ids, _ = self.robot.find_joints(".*")                                  # Ant 的全部关节 id
        self.joint_gears = torch.tensor([15.0] * cfg.action_dim, device=self.device)       # 每个关节的力矩倍率
        self.motor_effort_ratio = torch.ones(cfg.action_dim, device=self.device)           # 能耗惩罚中的关节权重

        self.actions = torch.zeros(num_envs, cfg.action_dim, device=self.device)           # 上一次 policy 输出动作
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)  # 每个 env 当前 episode 步数
        self.reset_terminated = torch.zeros(num_envs, dtype=torch.bool, device=self.device)     # 是否因摔倒终止
        self.reset_time_outs = torch.zeros(num_envs, dtype=torch.bool, device=self.device)      # 是否因超时截断
        self.rew_buf = torch.zeros(num_envs, device=self.device)                           # 每个 env 当前 reward

        # Ant Direct 的目标很简单：每个 Ant 都朝自己 env origin 前方 +x 方向 1000m 处前进。
        self.targets = torch.tensor([1000.0, 0.0, 0.0], device=self.device).repeat(num_envs, 1)
        self.targets += self.scene.env_origins

        # potential 用于 progress_reward：potential = -distance_to_target / dt。
        self.potentials = torch.zeros(num_envs, device=self.device)
        self.prev_potentials = torch.zeros(num_envs, device=self.device)

        self.up_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(num_envs, 1)
        self.forward_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(num_envs, 1)
        self._sim_step_counter = 0

    @property
    def max_episode_length(self) -> int:
        return self.cfg.max_episode_length

    # -------------------------------------------------------------------------
    # 4.4 reset：把 Ant 的 root/joint 状态写回默认状态
    # -------------------------------------------------------------------------

    def reset(self) -> torch.Tensor:
        """重置所有并行环境，并返回第一帧 observation。"""
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_idx(env_ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        return self.compute_observations(update_potential=False)

    def reset_idx(self, env_ids: torch.Tensor):
        """只重置指定 env。训练中 done 的 Ant 会在 step 内部自动调用它。"""
        if len(env_ids) == 0:
            return

        self.robot.reset(env_ids)
        self.scene.reset(env_ids)

        # default_root_state 是以 env local origin 为基准的，需要加上 scene.env_origins 变成世界坐标。
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        self.episode_length_buf[env_ids] = 0
        self.actions[env_ids] = 0.0
        self.reset_terminated[env_ids] = False
        self.reset_time_outs[env_ids] = False

        # reset 后重新初始化 potential，否则 progress reward 会跨 episode 串起来。
        to_target = self.targets[env_ids] - root_state[:, :3]
        to_target[:, 2] = 0.0
        self.potentials[env_ids] = -torch.norm(to_target, dim=-1) / self.cfg.sim_dt
        self.prev_potentials[env_ids] = self.potentials[env_ids]

    # -------------------------------------------------------------------------
    # 4.5 step：policy action -> joint effort -> physics -> reward/done/obs
    # -------------------------------------------------------------------------

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """执行一个 RL step。

        输入：
            actions: [num_envs, 8]

        内部：
            一个 RL step = decimation 个 physics step。
            Ant 中 decimation=2，所以 action 会连续作用两个 1/120s 的物理步。
        """
        self.actions = actions.to(self.device).clone()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1

            # Ant action 是 8 维，每维对应一个关节 effort。
            # 原始 AntEnv 中：forces = action_scale * joint_gears * actions = 0.5 * 15 * actions。
            joint_efforts = self.cfg.action_scale * self.joint_gears * self.actions
            self.robot.set_joint_effort_target(joint_efforts, joint_ids=self.joint_ids)

            self.scene.write_data_to_sim()
            self.sim.step(render=False)

            if is_rendering and self._sim_step_counter % self.cfg.decimation == 0:
                self.sim.render()

            # 从仿真读回 root pose / joint pos / joint vel 等 tensor buffer。
            self.scene.update(self.cfg.sim_dt)

        # 物理推进结束，进入 RL 后处理。
        self.episode_length_buf += 1
        self.reset_terminated, self.reset_time_outs = self.compute_dones()
        dones = self.reset_terminated | self.reset_time_outs
        rewards = self.compute_rewards(self.reset_terminated)

        # IsaacLab DirectRLEnv 的习惯：step 内部自动 reset 已经 done 的 env。
        reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        self.reset_idx(reset_env_ids)

        obs = self.compute_observations(update_potential=False)
        extras = {"time_outs": self.reset_time_outs.clone()}
        return obs, rewards, dones, extras

    # -------------------------------------------------------------------------
    # 4.6 状态读取：把 IsaacLab robot.data 转成 reward/obs 需要的中间量
    # -------------------------------------------------------------------------

    def compute_core_terms(self, update_potential: bool):
        """计算 Ant reward/observation 共用的中间量。

        注意：potential 只能在 reward 时更新一次，否则 progress_reward 会被 observation 的重复调用污染。
        """
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w
        root_lin_vel = self.robot.data.root_lin_vel_w
        root_ang_vel = self.robot.data.root_ang_vel_w
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel

        to_target = self.targets - root_pos
        to_target[:, 2] = 0.0
        target_dir = torch.nn.functional.normalize(to_target, dim=-1, eps=1.0e-6)

        # up_proj: 机器人局部 z 轴和世界 z 轴的对齐程度。越接近 1 说明越直立。
        # heading_proj: 机器人局部 x 轴和目标方向的对齐程度。越接近 1 说明越朝目标。
        up_vec = quat_rotate(root_quat, self.up_axis)
        forward_vec = quat_rotate(root_quat, self.forward_axis)
        up_proj = torch.sum(up_vec * self.up_axis, dim=-1)
        heading_proj = torch.sum(forward_vec * target_dir, dim=-1)

        # 速度转到机器人自身坐标系，和原始 Ant obs 中 vel_loc / angvel_loc 的直觉一致。
        vel_loc = quat_rotate_inverse(root_quat, root_lin_vel)
        angvel_loc = quat_rotate_inverse(root_quat, root_ang_vel)
        yaw = yaw_from_quat(root_quat)
        roll = roll_from_quat(root_quat)
        angle_to_target = torch.atan2(to_target[:, 1], to_target[:, 0]) - yaw

        lower = self.robot.data.soft_joint_pos_limits[0, :, 0]
        upper = self.robot.data.soft_joint_pos_limits[0, :, 1]
        dof_pos_scaled = scale_to_minus_one_one(joint_pos, lower, upper)

        if update_potential:
            self.prev_potentials[:] = self.potentials
            self.potentials[:] = -torch.norm(to_target, dim=-1) / self.cfg.sim_dt

        return root_pos, vel_loc, angvel_loc, yaw, roll, angle_to_target, up_proj, heading_proj, dof_pos_scaled, joint_vel

    # -------------------------------------------------------------------------
    # 4.7 observation：36 维，对齐 Isaac-Ant-Direct-v0 的观测结构
    # -------------------------------------------------------------------------

    def compute_observations(self, update_potential: bool) -> torch.Tensor:
        root_pos, vel_loc, angvel_loc, yaw, roll, angle_to_target, up_proj, heading_proj, dof_pos_scaled, joint_vel = (
            self.compute_core_terms(update_potential=update_potential)
        )

        # 维度展开：
        #   torso height                  1
        #   local linear velocity          3
        #   local angular velocity         3
        #   yaw                            1
        #   roll                           1
        #   angle_to_target                1
        #   up_proj                        1
        #   heading_proj                   1
        #   normalized joint positions     8
        #   scaled joint velocities        8
        #   previous/current actions       8
        # 总计：36
        obs = torch.cat(
            (
                root_pos[:, 2:3],
                vel_loc,
                angvel_loc * self.cfg.angular_velocity_scale,
                normalize_angle(yaw).unsqueeze(-1),
                normalize_angle(roll).unsqueeze(-1),
                normalize_angle(angle_to_target).unsqueeze(-1),
                up_proj.unsqueeze(-1),
                heading_proj.unsqueeze(-1),
                dof_pos_scaled,
                joint_vel * self.cfg.dof_vel_scale,
                self.actions,
            ),
            dim=-1,
        )
        return obs

    # -------------------------------------------------------------------------
    # 4.8 done：摔倒 or 时间到
    # -------------------------------------------------------------------------

    def compute_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_pos = self.robot.data.root_pos_w
        died = root_pos[:, 2] < self.cfg.termination_height
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return died, time_out

    # -------------------------------------------------------------------------
    # 4.9 reward：完整保留 AntEnvCfg 的 reward 项
    # -------------------------------------------------------------------------

    def compute_rewards(self, terminated: torch.Tensor) -> torch.Tensor:
        _, _, _, _, _, _, up_proj, heading_proj, dof_pos_scaled, joint_vel = self.compute_core_terms(
            update_potential=True
        )

        # 朝向奖励：朝目标方向越准越高，超过 0.8 后给满分 heading_weight。
        heading_reward = torch.where(
            heading_proj > 0.8,
            torch.full_like(heading_proj, self.cfg.heading_weight),
            self.cfg.heading_weight * heading_proj / 0.8,
        )

        # 直立奖励：up projection 足够大才给。
        up_reward = torch.where(up_proj > 0.93, torch.full_like(up_proj, self.cfg.up_weight), torch.zeros_like(up_proj))

        # 动作幅度惩罚：鼓励省力、平滑。
        actions_cost = torch.sum(self.actions.square(), dim=-1)

        # 类电费惩罚：动作大且关节速度大，认为耗能高。
        electricity_cost = torch.sum(
            torch.abs(self.actions * joint_vel * self.cfg.dof_vel_scale) * self.motor_effort_ratio.unsqueeze(0), dim=-1
        )

        # 关节极限惩罚：关节位置太接近 limit 会扣分。
        dof_at_limit_cost = torch.sum(dof_pos_scaled > 0.98, dim=-1).float()

        alive_reward = torch.full_like(self.potentials, self.cfg.alive_reward_scale)
        progress_reward = self.potentials - self.prev_potentials

        rewards = (
            progress_reward
            + alive_reward
            + up_reward
            + heading_reward
            - self.cfg.actions_cost_scale * actions_cost
            - self.cfg.energy_cost_scale * electricity_cost
            - dof_at_limit_cost
        )

        # 如果摔倒，直接覆盖为 death_cost。
        rewards = torch.where(terminated, torch.full_like(rewards, self.cfg.death_cost), rewards)
        self.rew_buf[:] = rewards
        return rewards

    def close(self):
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()


# =============================================================================
# 5. 手写 Actor-Critic：对应 RSL-RL 中的 actor/critic 网络
# =============================================================================


class ActorCritic(nn.Module):
    """最小 Gaussian actor + value critic。

    Actor 输入 36 维 obs，输出 8 维 action mean。
    log_std 是可学习参数，和 RSL-RL 常见实现类似。
    """

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.actor = self._mlp(obs_dim, action_dim)                  # actor: obs -> action mean
        self.critic = self._mlp(obs_dim, 1)                          # critic: obs -> state value
        self.log_std = nn.Parameter(torch.zeros(action_dim))         # Gaussian policy 的可学习标准差

    @staticmethod
    def _mlp(input_dim: int, output_dim: int) -> nn.Sequential:
        # 对齐 AntPPORunnerCfg: [400, 200, 100] + ELU。
        return nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.ELU(),
            nn.Linear(400, 200),
            nn.ELU(),
            nn.Linear(200, 100),
            nn.ELU(),
            nn.Linear(100, output_dim),
        )

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(obs)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def act(self, obs: torch.Tensor):
        """rollout 时采样动作，并记录 log_prob/value。"""
        dist = self.distribution(obs)
        actions = dist.sample()
        log_prob = dist.log_prob(actions).sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return actions, log_prob, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """PPO update 时重新计算当前策略下的 log_prob/entropy/value。"""
        dist = self.distribution(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return log_prob, entropy, value


# =============================================================================
# 6. 手写 RolloutStorage：收集 4096 env * 32 steps 的 on-policy 数据
# =============================================================================


class RolloutStorage:
    """PPO rollout buffer。

    形状约定：
        obs      [num_steps, num_envs, obs_dim]
        actions  [num_steps, num_envs, action_dim]
        rewards  [num_steps, num_envs]
        dones    [num_steps, num_envs]
    """

    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, action_dim: int, device: str):
        self.num_steps = num_steps                                             # rollout 时间长度 T
        self.num_envs = num_envs                                               # 并行环境数量 N
        self.obs = torch.zeros(num_steps, num_envs, obs_dim, device=device)     # 观测 buffer
        self.actions = torch.zeros(num_steps, num_envs, action_dim, device=device)  # 动作 buffer
        self.rewards = torch.zeros(num_steps, num_envs, device=device)          # reward buffer
        self.dones = torch.zeros(num_steps, num_envs, device=device)            # done buffer
        self.values = torch.zeros(num_steps, num_envs, device=device)           # critic value buffer
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)        # old policy log_prob
        self.advantages = torch.zeros(num_steps, num_envs, device=device)       # GAE advantage
        self.returns = torch.zeros(num_steps, num_envs, device=device)          # value learning target
        self.step = 0                                                           # 当前写入位置

    def clear(self):
        self.step = 0

    def add(self, obs, actions, rewards, dones, values, log_probs):
        self.obs[self.step].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards)
        self.dones[self.step].copy_(dones.float())
        self.values[self.step].copy_(values)
        self.log_probs[self.step].copy_(log_probs)
        self.step += 1

    def compute_returns(self, last_values: torch.Tensor, gamma: float, lam: float):
        """GAE(lambda)：从后往前计算 advantage 和 return。"""
        advantage = torch.zeros(self.num_envs, device=last_values.device)
        for step in reversed(range(self.num_steps)):
            next_values = last_values if step == self.num_steps - 1 else self.values[step + 1]
            not_done = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * next_values * not_done - self.values[step]
            advantage = delta + gamma * lam * not_done * advantage
            self.advantages[step] = advantage

        self.returns = self.advantages + self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1.0e-8)

    def mini_batches(self, num_mini_batches: int):
        """把 [T, N] 展平成 [T*N]，随机打乱后切 minibatch。"""
        batch_size = self.num_steps * self.num_envs
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(batch_size, device=self.obs.device)

        flat_obs = self.obs.reshape(batch_size, -1)
        flat_actions = self.actions.reshape(batch_size, -1)
        flat_old_log_probs = self.log_probs.reshape(batch_size)
        flat_old_values = self.values.reshape(batch_size)
        flat_returns = self.returns.reshape(batch_size)
        flat_advantages = self.advantages.reshape(batch_size)

        for start in range(0, batch_size, mini_batch_size):
            batch_idx = indices[start : start + mini_batch_size]
            yield (
                flat_obs[batch_idx],
                flat_actions[batch_idx],
                flat_old_log_probs[batch_idx],
                flat_old_values[batch_idx],
                flat_returns[batch_idx],
                flat_advantages[batch_idx],
            )


# =============================================================================
# 7. 手写 PPO：clip surrogate + clipped value loss
# =============================================================================


class PPO:
    """最小 PPO 实现，参数对齐 AntPPORunnerCfg。"""

    def __init__(self, actor_critic: ActorCritic, learning_rate: float = 5.0e-4):
        self.actor_critic = actor_critic
        self.optimizer = torch.optim.Adam(actor_critic.parameters(), lr=learning_rate)

        self.clip_param = 0.2             # PPO ratio/value clip 范围
        self.value_loss_coef = 1.0        # value loss 权重
        self.entropy_coef = 0.0           # entropy bonus 权重
        self.num_learning_epochs = 5      # 每批 rollout 重复训练轮数
        self.num_mini_batches = 4         # 每轮切成几个 minibatch
        self.max_grad_norm = 1.0          # 梯度裁剪上限
        self.gamma = 0.99                 # 折扣因子
        self.lam = 0.95                   # GAE lambda

    def update(self, storage: RolloutStorage):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        num_updates = 0

        # 同一批 rollout 数据会被重复训练 5 个 epoch，每个 epoch 切成 4 个 minibatch。
        for _ in range(self.num_learning_epochs):
            for obs, actions, old_log_probs, old_values, returns, advantages in storage.mini_batches(self.num_mini_batches):
                log_probs, entropy, values = self.actor_critic.evaluate(obs, actions)

                # ratio = pi_new(a|s) / pi_old(a|s)
                ratio = torch.exp(log_probs - old_log_probs)

                # PPO clipped policy loss。
                surrogate = -advantages * ratio
                surrogate_clipped = -advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # clipped value loss。
                value_clipped = old_values + torch.clamp(values - old_values, -self.clip_param, self.clip_param)
                value_losses = (values - returns).square()
                value_losses_clipped = (value_clipped - returns).square()
                value_loss = torch.max(value_losses, value_losses_clipped).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                num_updates += 1

        return mean_value_loss / num_updates, mean_surrogate_loss / num_updates


# =============================================================================
# 8. 主训练循环：对应 train.py + RSL-RL OnPolicyRunner.learn 的教学版
# =============================================================================


def train():
    torch.manual_seed(args_cli.seed)
    device = args_cli.device
    if "cuda" in device:
        torch.cuda.manual_seed_all(args_cli.seed)

    # 1. 创建 task/env/policy/ppo/storage。
    task_cfg = AntTaskCfg()
    env = HandWrittenAntEnv(num_envs=args_cli.num_envs, device=device, cfg=task_cfg)
    actor_critic = ActorCritic(task_cfg.obs_dim, task_cfg.action_dim).to(device)
    ppo = PPO(actor_critic)
    storage = RolloutStorage(args_cli.num_steps_per_env, args_cli.num_envs, task_cfg.obs_dim, task_cfg.action_dim, device)

    log_dir = Path(args_cli.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 2. 初始 reset。随机化 episode_length_buf 是为了避免所有 env 同步 timeout。
    obs = env.reset()
    env.episode_length_buf = torch.randint(0, env.max_episode_length, (env.num_envs,), device=env.device)

    for iteration in range(args_cli.max_iterations):
        storage.clear()
        reward_sum = 0.0
        done_sum = 0

        # ---------------------------------------------------------------------
        # A. rollout：收集 num_envs * num_steps_per_env 条 transition
        # ---------------------------------------------------------------------
        for _ in range(args_cli.num_steps_per_env):
            with torch.no_grad():
                actions, log_probs, values = actor_critic.act(obs)

            next_obs, rewards, dones, _ = env.step(actions)
            storage.add(obs, actions, rewards, dones, values, log_probs)

            obs = next_obs
            reward_sum += rewards.mean().item()
            done_sum += int(dones.sum().item())

        # ---------------------------------------------------------------------
        # B. GAE：用最后一帧 value bootstrap，计算 advantages / returns
        # ---------------------------------------------------------------------
        with torch.no_grad():
            last_values = actor_critic.critic(obs).squeeze(-1)
        storage.compute_returns(last_values, ppo.gamma, ppo.lam)

        # ---------------------------------------------------------------------
        # C. PPO update：用刚收集的 on-policy 数据更新 actor-critic
        # ---------------------------------------------------------------------
        value_loss, policy_loss = ppo.update(storage)

        print(
            f"iter={iteration:04d} "
            f"mean_step_reward={reward_sum / args_cli.num_steps_per_env: .4f} "
            f"resets={done_sum:5d} "
            f"value_loss={value_loss: .4f} "
            f"policy_loss={policy_loss: .4f}"
        )

        # ---------------------------------------------------------------------
        # D. checkpoint
        # ---------------------------------------------------------------------
        if iteration % args_cli.save_interval == 0 or iteration == args_cli.max_iterations - 1:
            torch.save(
                {
                    "iteration": iteration,
                    "model_state_dict": actor_critic.state_dict(),
                    "optimizer_state_dict": ppo.optimizer.state_dict(),
                },
                log_dir / f"model_{iteration}.pt",
            )

    env.close()


if __name__ == "__main__":
    try:
        train()
    finally:
        simulation_app.close()
