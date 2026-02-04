import os
import sys
import types
import tempfile

import torch
import gymnasium as gym

import stowage_gym as local_stowage_gym

custom_envs = types.ModuleType("custom_envs")
sys.modules["custom_envs"] = custom_envs
sys.modules["custom_envs.stowage_gym"] = local_stowage_gym

from ppo import Agent, load_checkpoint, prune_checkpoints, save_checkpoint, list_checkpoint_paths


class DummyVectorEnv:
    def __init__(self):
        self.single_observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=float)
        self.single_action_space = gym.spaces.Discrete(2)


def run_checkpoint_flow():
    envs = DummyVectorEnv()
    agent = Agent(envs, hidden_size=8)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmpdir:
        for iteration in range(1, 4):
            save_checkpoint(
                agent,
                optimizer,
                tmpdir,
                iteration=iteration,
                global_step=iteration * 10,
                args={"seed": 123},
            )
            prune_checkpoints(tmpdir, max_checkpoints=2)

        final_path = save_checkpoint(
            agent,
            optimizer,
            tmpdir,
            iteration=4,
            global_step=40,
            args={"seed": 123},
            tag="final",
        )
        prune_checkpoints(tmpdir, max_checkpoints=2)

        checkpoints = list_checkpoint_paths(tmpdir)
        assert os.path.exists(final_path)
        assert len(checkpoints) <= 2

        new_agent = Agent(envs, hidden_size=8)
        load_checkpoint(final_path, new_agent)

        for param_old, param_new in zip(agent.parameters(), new_agent.parameters()):
            assert torch.allclose(param_old, param_new)
            break


if __name__ == "__main__":
    run_checkpoint_flow()
