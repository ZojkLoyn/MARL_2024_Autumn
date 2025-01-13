'''
训练策略
参考自 pytorch 文档 https://pytorch.org/rl/stable/tutorials/multiagent_ppo.html
'''

__all__ = ["CPPO", "MAPPO", "IPPO", "HetIPPO"]

import config
import torch

from torchrl.envs import RewardSum, TransformedEnv

from tensordict.nn import TensorDictModule, NormalParamExtractor
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.collectors import SyncDataCollector

from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

from torchrl.objectives import ClipPPOLoss, ValueEstimators

from tqdm import tqdm
from matplotlib import pyplot as plt

class PPO:
    critic_net_centralised = True
    policy_net_centralised = False
    share_params = True

    def __init__(self):
        env = config.env
        self.env = env = TransformedEnv(
            env,
            RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
        )
        
        # 决策网络
        policy_net = torch.nn.Sequential(
            MultiAgentMLP(
                n_agent_inputs=env.observation_spec[
                    "agents", "observation"].shape[-1],  # n_obs_per_agent
                n_agent_outputs=2 *
                env.action_spec.shape[-1],  # 2 * n_actions_per_agents
                n_agents=env.n_agents,
                centralised=self.policy_net_centralised,
                share_params=self.share_params,
                device=config.device,
                depth=2,
                num_cells=256,
                activation_class=torch.nn.Tanh,
            ),
            torch.nn.Hardtanh(),
            NormalParamExtractor(
            ),  # this will just separate the last dimension into two outputs: a loc and a non-negative scale
        )
        policy_module = TensorDictModule(
            policy_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "loc"), ("agents", "scale")],
        )
        self.policy = ProbabilisticActor(
            module=policy_module,
            spec=env.unbatched_action_spec,
            in_keys=[("agents", "loc"), ("agents", "scale")],
            out_keys=[env.action_key],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": env.unbatched_action_spec[env.action_key].space.low,
                "high": env.unbatched_action_spec[env.action_key].space.high,
            },
            return_log_prob=True,
            log_prob_key=("agents", "sample_log_prob"),
        )  # we'll need the log-prob for the PPO loss
        
        # 评论家网络
        critic_net = MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=1,  # 1 value per agent
            n_agents=env.n_agents,
            centralised=self.critic_net_centralised,
            share_params=self.share_params,
            device=config.device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        )

        self.critic = TensorDictModule(
            module=critic_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "state_value")],
        )
        self.collector = SyncDataCollector(
            env,
            self.policy,
            device=config.vmas_device,
            storing_device=config.device,
            frames_per_batch=config.frames_per_batch,
            total_frames=config.total_frames,
        )
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(
                config.frames_per_batch, device=config.device
            ),  # We store the frames_per_batch collected at each iteration
            sampler=SamplerWithoutReplacement(),
            batch_size=config.minibatch_size,  # We will sample minibatches of this size
        )
        
        loss_module = ClipPPOLoss(
            actor_network=self.policy,
            critic_network=self.critic,
            clip_epsilon=config.clip_epsilon,
            entropy_coef=config.entropy_eps,
            normalize_advantage=False,  # Important to avoid normalizing across the agent dimension
        )
        loss_module.set_keys(  # We have to tell the loss where to find the keys
            reward=env.reward_key,
            action=env.action_key,
            sample_log_prob=("agents", "sample_log_prob"),
            value=("agents", "state_value"),
            # These last 2 keys will be expanded to match the reward shape
            done=("agents", "done"),
            terminated=("agents", "terminated"),
        )


        loss_module.make_value_estimator(
            ValueEstimators.GAE, gamma=config.gamma, lmbda=config.lmbda
        )  # We build GAE
        self.loss_module = loss_module
        self.GAE = loss_module.value_estimator

        self.optim = torch.optim.Adam(loss_module.parameters(), config.lr)
    
    def train(self):
        pbar = tqdm(total=config.n_iters, desc="episode_reward_mean = 0")

        episode_reward_mean_list = []
        for tensordict_data in self.collector:
            tensordict_data.set(
                ("next", "agents", "done"),
                tensordict_data.get(("next", "done"))
                .unsqueeze(-1)
                .expand(tensordict_data.get_item_shape(("next", self.env.reward_key))),
            )
            tensordict_data.set(
                ("next", "agents", "terminated"),
                tensordict_data.get(("next", "terminated"))
                .unsqueeze(-1)
                .expand(tensordict_data.get_item_shape(("next", self.env.reward_key))),
            )
            # We need to expand the done and terminated to match the reward shape (this is expected by the value estimator)

            with torch.no_grad():
                self.GAE(
                    tensordict_data,
                    params=self.loss_module.critic_network_params,
                    target_params=self.loss_module.target_critic_network_params,
                )  # Compute GAE and add it to the data

            data_view = tensordict_data.reshape(-1)  # Flatten the batch size to shuffle data
            self.replay_buffer.extend(data_view)

            for _ in range(config.num_epochs):
                for _ in range(config.frames_per_batch // config.minibatch_size):
                    subdata = self.replay_buffer.sample()
                    loss_vals = self.loss_module(subdata)

                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    loss_value.backward()

                    torch.nn.utils.clip_grad_norm_(
                        self.loss_module.parameters(), config.max_grad_norm
                    )  # Optional

                    self.optim.step()
                    self.optim.zero_grad()

            self.collector.update_policy_weights_()

            # Logging
            done = tensordict_data.get(("next", "agents", "done"))
            episode_reward_mean = (
                tensordict_data.get(("next", "agents", "episode_reward"))[done].mean().item()
            )
            episode_reward_mean_list.append(episode_reward_mean)
            pbar.set_description(f"episode_reward_mean = {episode_reward_mean}", refresh=False)
            pbar.update()
            
        plt.plot(episode_reward_mean_list)
        plt.xlabel("Training iterations")
        plt.ylabel("Reward")
        plt.title("CPPO" + " Episode reward mean")
        plt.show()
            
    def rendering(self):
        with torch.no_grad():
            self.env.rollout(
                max_steps=config.max_steps,
                policy=self.policy,
                callback=lambda env, _: env.render(),
                auto_cast_to_device=True,
                break_when_any_done=False,
            )
            
class CPPO(PPO):
    policy_net_centralised = True
    critic_net_centralised = True
    share_params = False

class MAPPO(PPO):
    policy_net_centralised = False
    critic_net_centralised = True

class IPPO(PPO):
    policy_net_centralised = False
    critic_net_centralised = False

class HetIPPO(IPPO):
    share_params = False