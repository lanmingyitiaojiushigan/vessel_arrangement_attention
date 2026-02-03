"""
Attention Model PPO Training for Container Stowage Problem

Based on:
- "Attention, Learn to Solve Routing Problems!" (Kool et al., ICLR 2019)
- PPO algorithm from CleanRL

Paper hyperparameters:
- Encoder: N=3 layers, d_h=128, M=8 heads, d_ff=512
- Batch Normalization
- Logit clipping C=10
"""

# ==================== Fix OpenMP conflict ====================
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
from torch.utils.tensorboard import SummaryWriter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from custom_envs.stowage_gym import StowageEnv
from attention_agent import AttentionAgent


@dataclass
class Args:
    exp_name: str = "attention_ppo"
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
    """whether to capture videos of the agent performances"""

    # PPO arguments (from your paper Table 6, Scenario 5)
    env_id: str = "Stowage-v0"
    """the id of the environment"""
    total_timesteps: int = 300000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-5  # Paper Sc.5: 0.00003
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 512  # Paper Sc.5: 1024
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing"""
    gamma: float = 0.20  # Paper Sc.5: 0.20
    """the discount factor gamma"""
    gae_lambda: float = 0.98  # Paper Sc.5: 0.98
    """the lambda for GAE"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles clipped value loss"""
    ent_coef: float = 0.003  # Paper Sc.5: 0.003
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 4.94  # Paper Sc.5: 4.94
    """the maximum norm for gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Attention Model arguments (memory-efficient, based on Kool et al. paper)
    d_model: int = 32
    """embedding dimension d_h (paper: 128, reduced for memory)"""
    n_layers: int = 1
    """number of encoder layers N (same as paper)"""
    n_heads: int = 4
    """number of attention heads M (paper: 8, reduced for memory)"""
    d_ff: int = 64
    """feed-forward hidden dimension (paper: 512, reduced for memory)"""
    clip_logits: float = 10.0
    """logit clipping C (same as paper)"""

    # Runtime computed
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


# ==================== Logger ====================

class TrainingLogger:
    def __init__(self, algorithm_name="Attention-PPO"):
        self.algorithm_name = algorithm_name
        self.data = defaultdict(list)

    def log(self, global_step, **kwargs):
        self.data["global_step"].append(global_step)
        for key, value in kwargs.items():
            self.data[key].append(value)

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump({"algorithm": self.algorithm_name, "data": dict(self.data)}, f)
        print(f"Training data saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'r') as f:
            save_data = json.load(f)
        logger = cls(save_data["algorithm"])
        logger.data = defaultdict(list, save_data["data"])
        return logger


# ==================== Plotting ====================

def plot_training_curve(loggers, metric="mean_shifters", title="Training Curve",
                        ylabel="Mean Shifters", save_path=None, figsize=(8, 5)):
    colors = {
        'PPO': '#32CD32',
        'Attention-PPO': '#1E90FF',
    }

    plt.figure(figsize=figsize)

    for algo_name, logger_data in loggers.items():
        color = colors.get(algo_name, '#333333')

        if isinstance(logger_data, list):
            # Multiple runs
            all_steps = [np.array(l.data["global_step"]) for l in logger_data if metric in l.data]
            all_values = [np.array(l.data[metric]) for l in logger_data if metric in l.data]

            if all_values:
                min_len = min(len(v) for v in all_values)
                steps = all_steps[0][:min_len]
                values = np.array([v[:min_len] for v in all_values])
                mean_vals = values.mean(axis=0)
                std_vals = values.std(axis=0)

                plt.plot(steps, mean_vals, color=color, label=algo_name, linewidth=1.5)
                plt.fill_between(steps, mean_vals - std_vals, mean_vals + std_vals,
                               color=color, alpha=0.2)
        else:
            if metric in logger_data.data:
                plt.plot(logger_data.data["global_step"], logger_data.data[metric],
                        color=color, label=algo_name, linewidth=1.5)

    plt.xlabel("Total Time Steps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.gca().ticklabel_format(style='sci', axis='x', scilimits=(3, 3))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.close()


# ==================== Environment ====================

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


# ==================== Training ====================

def train_attention_ppo(args, logger=None):
    if logger is None:
        logger = TrainingLogger("Attention-PPO")

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join([f"|{k}|{v}|" for k, v in vars(args).items()]))

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # Environments
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )

    # Attention Agent
    agent = AttentionAgent(
        envs,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        clip_logits=args.clip_logits
    ).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    action_masks = torch.zeros((args.num_steps, args.num_envs, envs.single_action_space.n)).to(device)

    # Initialize
    global_step = 0
    start_time = time.time()
    next_obs, info = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    mask = torch.tensor(np.stack(info["action_mask"]), device=device)

    print(f"\nStarting Attention-PPO training...")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Batch size: {args.batch_size:,}")
    print(f"Iterations: {args.num_iterations:,}")

    for iteration in range(1, args.num_iterations + 1):
        # LR annealing
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # Rollout
        for step in range(args.num_steps):
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
            mask = torch.tensor(np.stack(infos["action_mask"]), device=device)
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward, device=device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(next_done).to(device)

            # Log episodes
            if "final_info" in infos:
                ep_data = [(i["episode"]["r"], i["episode"]["l"],
                           i.get("total_shifters", 0), i.get("current_time", 0))
                          for i in infos["final_info"] if i and "episode" in i]

                if ep_data:
                    returns, lens, shifters, times = zip(*ep_data)
                    mean_return = np.mean(returns)
                    mean_len = np.mean(lens)
                    mean_shifters = np.mean(shifters)
                    min_shifters = np.min(shifters)
                    max_shifters = np.max(shifters)
                    mean_time = np.mean(times)

                    logger.log(global_step,
                        mean_reward=mean_return, mean_shifters=mean_shifters,
                        min_shifters=min_shifters, max_shifters=max_shifters,
                        mean_time=mean_time, ep_len=mean_len)

                    print(f"[Iter {iteration:4d}] Step: {global_step:8d} | "
                          f"Reward: {mean_return:7.0f} | Shifters: {mean_shifters:5.0f} "
                          f"(min:{min_shifters:.0f}, max:{max_shifters:.0f})")

                    writer.add_scalar("eval/mean_reward", mean_return, global_step)
                    writer.add_scalar("eval/mean_shifters", mean_shifters, global_step)
                    writer.add_scalar("eval/min_shifters", min_shifters, global_step)
                    writer.add_scalar("eval/max_shifters", max_shifters, global_step)

        # GAE
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
                mb_inds = b_inds[start:start + args.minibatch_size]

                _, newlogprob, entropy, newvalue = agent.get_masked_action_and_value(
                    b_obs[mb_inds], b_action_masks[mb_inds], b_actions.long()[mb_inds])

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - b_returns[mb_inds]) ** 2).mean()
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

        # Logging
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
    return logger, agent


# ==================== Main ====================

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

    print("=" * 60)
    print("Attention Model PPO for Container Stowage")
    print("=" * 60)
    print(f"Attention: d_model={args.d_model}, N={args.n_layers}, M={args.n_heads}, d_ff={args.d_ff}")
    print(f"PPO: lr={args.learning_rate}, gamma={args.gamma}, ent_coef={args.ent_coef}")
    print("=" * 60)

    logger, agent = train_attention_ppo(args)

    # Save
    logger.save("attention_ppo_training_data.json")
    torch.save(agent.state_dict(), f"attention_ppo_model_{args.seed}.pt")

    plot_training_curve(
        {"Attention-PPO": logger},
        metric="mean_shifters",
        title="Attention-PPO Training",
        ylabel="Mean Shifters",
        save_path="attention_ppo_training_curve.png"
    )

    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)