# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
# Modified to match paper: 1 hidden layer with 64 neurons
# Added: Paper-style plotting functionality

# ==================== 修复 OpenMP 冲突 (必须放在最开头) ====================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
import time
from dataclasses import dataclass
from collections import defaultdict
import json

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# 使用非交互式后端避免显示问题
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from custom_envs.stowage_gym import StowageEnv


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments (matching paper Table 6, Scenario 5)
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-5  # Paper Sc.5: 0.00003
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 1024  # Paper Sc.5: 1024
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.20  # Paper Sc.5: 0.20
    """the discount factor gamma"""
    gae_lambda: float = 0.98  # Paper Sc.5: 0.98
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.003  # Paper Sc.5: 0.003
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 4.94  # Paper Sc.5: 4.94
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


# ==================== Plotting Utilities ====================

class TrainingLogger:
    """记录训练过程中的指标，用于后续绘图"""

    def __init__(self, algorithm_name="PPO"):
        self.algorithm_name = algorithm_name
        self.data = defaultdict(list)

    def log(self, global_step, **kwargs):
        """记录一个时间步的数据"""
        self.data["global_step"].append(global_step)
        for key, value in kwargs.items():
            self.data[key].append(value)

    def save(self, filepath):
        """保存数据到JSON文件"""
        save_data = {
            "algorithm": self.algorithm_name,
            "data": dict(self.data)
        }
        with open(filepath, 'w') as f:
            json.dump(save_data, f)
        print(f"Training data saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """从JSON文件加载数据"""
        with open(filepath, 'r') as f:
            save_data = json.load(f)
        logger = cls(save_data["algorithm"])
        logger.data = defaultdict(list, save_data["data"])
        return logger


def plot_training_curve(loggers, metric="mean_shifters", title="Evaluation Mean Shifters",
                        ylabel="Evaluation Mean Shifters", save_path=None,
                        figsize=(6, 4), show_std=True):
    """
    绘制单个场景的训练曲线（论文风格）

    Args:
        loggers: dict, {algorithm_name: TrainingLogger} 或 {algorithm_name: list of TrainingLogger}
        metric: str, 要绘制的指标名称
        title: str, 图表标题
        ylabel: str, Y轴标签
        save_path: str, 保存路径
        figsize: tuple, 图表大小
        show_std: bool, 是否显示标准差阴影
    """
    # 论文配色方案
    colors = {
        'QR-DQN': '#00CED1',  # 青色
        'DQN': '#FFD700',  # 金色
        'A2C': '#FF6347',  # 橙红色
        'PPO': '#32CD32',  # 绿色
        'TRPO': '#DA70D6',  # 紫色
    }

    plt.figure(figsize=figsize)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 11

    for algo_name, logger_data in loggers.items():
        color = colors.get(algo_name, '#333333')

        # 如果是多次运行的列表
        if isinstance(logger_data, list):
            # 收集所有运行的数据
            all_steps = []
            all_values = []
            for logger in logger_data:
                if metric in logger.data:
                    all_steps.append(np.array(logger.data["global_step"]))
                    all_values.append(np.array(logger.data[metric]))

            if not all_values:
                continue

            # 找到共同的步数点进行插值
            min_len = min(len(s) for s in all_steps)
            common_steps = all_steps[0][:min_len]

            # 对齐数据
            aligned_values = []
            for steps, values in zip(all_steps, all_values):
                aligned_values.append(values[:min_len])

            aligned_values = np.array(aligned_values)
            mean_values = np.mean(aligned_values, axis=0)
            std_values = np.std(aligned_values, axis=0)

            # 绘制均值线
            plt.plot(common_steps, mean_values, color=color, label=algo_name, linewidth=1.5)

            # 绘制标准差阴影
            if show_std and len(aligned_values) > 1:
                plt.fill_between(common_steps,
                                 mean_values - std_values,
                                 mean_values + std_values,
                                 color=color, alpha=0.2)
        else:
            # 单次运行
            if metric in logger_data.data:
                steps = logger_data.data["global_step"]
                values = logger_data.data[metric]
                plt.plot(steps, values, color=color, label=algo_name, linewidth=1.5)

    plt.xlabel("Total Time Steps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
    plt.grid(True, alpha=0.3)

    # 设置x轴为科学计数法
    ax = plt.gca()
    ax.ticklabel_format(style='sci', axis='x', scilimits=(3, 3))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.close()


def plot_multi_scenario(all_scenario_data, metric="mean_shifters",
                        save_path=None, figsize=(14, 8)):
    """
    绘制多场景对比图（论文Figure 8风格）

    Args:
        all_scenario_data: dict, {scenario_name: {algorithm_name: TrainingLogger or list}}
        metric: str, 要绘制的指标
        save_path: str, 保存路径
        figsize: tuple, 整体图表大小
    """
    colors = {
        'QR-DQN': '#00CED1',
        'DQN': '#FFD700',
        'A2C': '#FF6347',
        'PPO': '#32CD32',
        'TRPO': '#DA70D6',
    }

    n_scenarios = len(all_scenario_data)

    # 确定子图布局
    if n_scenarios <= 3:
        nrows, ncols = 1, n_scenarios
    elif n_scenarios <= 6:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = 3, 3

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_scenarios == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    plt.rcParams['font.size'] = 9

    for idx, (scenario_name, loggers) in enumerate(all_scenario_data.items()):
        ax = axes[idx]

        for algo_name, logger_data in loggers.items():
            color = colors.get(algo_name, '#333333')

            if isinstance(logger_data, list):
                all_steps = []
                all_values = []
                for logger in logger_data:
                    if metric in logger.data:
                        all_steps.append(np.array(logger.data["global_step"]))
                        all_values.append(np.array(logger.data[metric]))

                if not all_values:
                    continue

                min_len = min(len(s) for s in all_steps)
                common_steps = all_steps[0][:min_len]
                aligned_values = np.array([v[:min_len] for v in all_values])
                mean_values = np.mean(aligned_values, axis=0)
                std_values = np.std(aligned_values, axis=0)

                ax.plot(common_steps, mean_values, color=color, label=algo_name, linewidth=1.2)
                if len(aligned_values) > 1:
                    ax.fill_between(common_steps,
                                    mean_values - std_values,
                                    mean_values + std_values,
                                    color=color, alpha=0.2)
            else:
                if metric in logger_data.data:
                    steps = logger_data.data["global_step"]
                    values = logger_data.data[metric]
                    ax.plot(steps, values, color=color, label=algo_name, linewidth=1.2)

        ax.set_xlabel("Total Time Steps")
        ax.set_ylabel("Evaluation Mean Shifters")
        ax.set_title(f"({chr(97 + idx)}) {scenario_name}")
        ax.ticklabel_format(style='sci', axis='x', scilimits=(3, 3))
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for idx in range(len(all_scenario_data), len(axes)):
        axes[idx].set_visible(False)

    # 添加统一图例
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5,
               bbox_to_anchor=(0.5, -0.02), frameon=True, fancybox=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Multi-scenario plot saved to {save_path}")

    plt.close()


# ==================== Network Definition ====================

def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        config = {
            "vessel_shape": (8, 5, 5),
            "yard_shape": (8, 5, 5),
            "num_containers": 200,
            "group_num": 8,
            "group_placement": "random",
            "seed": 4307,
        }

        env = StowageEnv(config)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """
    Agent with 1 hidden layer (64 neurons) matching paper configuration.
    """

    def __init__(self, envs, hidden_size=64, activation_fn=nn.ReLU):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = envs.single_action_space.n

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            activation_fn(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            activation_fn(),
            layer_init(nn.Linear(hidden_size, action_dim), std=0.01),
        )

        total_params = sum(p.numel() for p in self.parameters())
        print(f"Network architecture: 1 hidden layer, {hidden_size} neurons")
        print(f"Total parameters: {total_params:,}")

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def get_masked_action_and_value(self, x, action_mask, action=None):
        logits = self.actor(x)
        masked_logits = logits.clone()
        masked_logits[action_mask == 0] = -1e9
        probs = Categorical(logits=masked_logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


# ==================== Training Function ====================

def train_ppo(args, logger=None):
    """
    PPO训练函数，返回TrainingLogger用于绘图
    """
    if logger is None:
        logger = TrainingLogger("PPO")

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )

    agent = Agent(envs, hidden_size=64, activation_fn=nn.ReLU).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    action_masks = torch.zeros((args.num_steps, args.num_envs, envs.single_action_space.n)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, info = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    mask_np = np.stack(info["action_mask"])
    mask = torch.tensor(mask_np, device=device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            action_masks[step] = mask

            with torch.no_grad():
                action, logprob, _, value = agent.get_masked_action_and_value(next_obs, mask)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

            mask_np = np.stack(infos["action_mask"])
            mask = torch.tensor(mask_np, device=device)

            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                ep_returns = []
                ep_lens = []
                shifters = []
                times = []

                for info in infos["final_info"]:
                    if info and "episode" in info:
                        ep_returns.append(info["episode"]["r"])
                        ep_lens.append(info["episode"]["l"])
                        shifters.append(info.get("total_shifters", 0))
                        times.append(info.get("current_time", 0))

                if ep_returns:
                    mean_return = float(np.mean(ep_returns))
                    mean_len = float(np.mean(ep_lens))
                    mean_shifters = float(np.mean(shifters))
                    mean_time = float(np.mean(times))
                    min_shifters = float(np.min(shifters))
                    max_shifters = float(np.max(shifters))

                    # 记录到logger用于绘图
                    logger.log(
                        global_step,
                        mean_reward=mean_return,
                        mean_shifters=mean_shifters,
                        min_shifters=min_shifters,
                        max_shifters=max_shifters,
                        mean_time=mean_time,
                        ep_len=mean_len
                    )

                    print("-------------------------------")
                    print("| eval/              |        |")
                    print(f"|    ep_len          | {int(mean_len):8d} |")
                    print(f"|    mean_reward     | {mean_return:8.0f} |")
                    print(f"|    mean_shifters   | {mean_shifters:8.0f} |")
                    print(f"|    min_shifters    | {min_shifters:8.0f} |")
                    print(f"|    max_shifters    | {max_shifters:8.0f} |")
                    print(f"|    mean_time       | {mean_time:8.2e} |")
                    print(f"|    iteration       | {iteration:8d} |")
                    print("-------------------------------")

                    writer.add_scalar("eval/ep_len", mean_len, global_step)
                    writer.add_scalar("eval/mean_reward", mean_return, global_step)
                    writer.add_scalar("eval/mean_shifters", mean_shifters, global_step)
                    writer.add_scalar("eval/min_shifters", min_shifters, global_step)
                    writer.add_scalar("eval/max_shifters", max_shifters, global_step)
                    writer.add_scalar("eval/mean_time", mean_time, global_step)

        # Bootstrap and GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_action_masks = action_masks.reshape((-1, envs.single_action_space.n))

        # PPO Update
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_masked_action_and_value(
                    b_obs[mb_inds],
                    b_action_masks[mb_inds],
                    b_actions.long()[mb_inds]
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

    return logger


# ==================== Main ====================

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    logger = train_ppo(args)

    # 绘制单次训练曲线
    plot_training_curve(
        {"PPO": logger},
        metric="mean_shifters",
        title="PPO Training - Evaluation Mean Shifters",
        ylabel="Evaluation Mean Shifters",
        save_path="ppo_training_curve.png"
    )

    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)
    print(f"Plot saved to: ppo_training_curve.png")