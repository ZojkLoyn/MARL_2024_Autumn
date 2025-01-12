
from config import *

from vmas.simulator.utils import save_video

from net import *
import torch
import time
import numpy as np
import os
from tqdm import tqdm

def get_ppo(name = "CPPO", **kwargs):
    if name == "CPPO":
        return CPPO(**kwargs)
    elif name == "MAPPO":
        return MAPPO(**kwargs)
    elif name == "IPPO":
        return IPPO(**kwargs)
    elif name == 'HetIPPO':
        return HetIPPO(**kwargs)

def ls2tensor(ls):
    return torch.stack(ls, dim=0).to(device)

class PPO:
    critic_net_centralised = True # 评论网络集中化
    policy_net_centralised = True # 策略网络集中化
    share_params = True # Agent之间共享参数，若集中化则无效
    
    @property
    def critic_net(self):
        if self._critic_net is None:
            self._critic_net = CriticNet(centralised=self.critic_net_centralised, share_params=self.share_params)
        return self._critic_net
    
    @property
    def policy_net(self):
        if self._policy_net is None:
            self._policy_net = PolicyNet(centralised=self.policy_net_centralised, share_params=self.share_params)
        return self._policy_net
    
    def __init__(self, 
                 critic_net_load = False,
                 policy_net_load = False,
                 ):
        self._critic_net = None
        self._policy_net = None
        self._critic_net_optim = torch.optim.Adam(self.critic_net.parameters(), lr=lr)
        self._policy_net_optim = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        if critic_net_load and os.path.exists(critic_net_load_path):
            self.critic_net.load_state_dict(torch.load(critic_net_load_path, weights_only=True))
        if policy_net_load and os.path.exists(policy_net_load_path):
            self.policy_net.load_state_dict(torch.load(policy_net_load_path, weights_only=True))
        
    def save(self):
        torch.save(self.critic_net.state_dict(), critic_net_load_path)
        torch.save(self.policy_net.state_dict(), policy_net_load_path)
    
    def train(self):
        state = env.reset()
        state = ls2tensor(state)
        softmax = torch.nn.Softmax(dim=-1)
        get_prob = lambda x: softmax(x)
        for iter in tqdm(range(train_iters)):
            if iter % reset_iters == 0:
                state = env.reset()
                state = ls2tensor(state)
            state_list = [state] # [train_steps+1][n_agent, n_env, state_space]
            action_list = [] # [train_steps][n_agent, n_env, action_space]
            reward_list = [] # [train_steps][n_agent, n_env, 1]
            done_list = [] # [train_steps][n_agent, n_env, 1]
            with torch.no_grad():
                for _ in range(train_steps):
                    action = self.policy_net(state)
                    next_states, rewards, dones, _ = env.step(action)
                    state = ls2tensor(next_states)
                    rewards = ls2tensor(rewards).unsqueeze(-1)
                    dones.unsqueeze_(-1)
                    action_list.append(action)
                    state_list.append(state)
                    reward_list.append(rewards)
                    done_list.append(dones)
            
            action_tensor = torch.stack(action_list, dim=0).permute(1, 0, 2, 3).flatten(1, 2)
            action_prob_tensor = get_prob(action_tensor)
            state_tensor = torch.stack(state_list[:-1], dim=0).permute(1, 0, 2, 3).flatten(1, 2)
            next_states_tensor = torch.stack(state_list[1:], dim=0).permute(1, 0, 2, 3).flatten(1, 2)
            reward_tensor = torch.stack(reward_list, dim=0).permute(1, 0, 2, 3).flatten(1, 2)
            done_tensor = torch.cat(done_list, dim=0)
            
            for _ in range(update_epochs):
                critic = self.critic_net(state_tensor)
                next_critic = self.critic_net(next_states_tensor)
                td_target = reward_tensor + gamma * next_critic * ~done_tensor
                advantage = td_target - critic

                critic_loss = advantage.pow(2).mean() # MSE Loss
                self._critic_net_optim.zero_grad()
                critic_loss.backward()
                self._critic_net_optim.step()
                
                action = self.policy_net(state_tensor)
                action_prob = get_prob(action)
                action_prob = action_prob / action_prob_tensor
                
                A = action_prob * advantage.detach()
                B = torch.clamp(action_prob, 1-clip_epsilon, 1+clip_epsilon) * advantage.detach()
                policy_loss = (torch.min(A, B)).mean() # clip loss
                self._policy_net_optim.zero_grad()
                policy_loss.backward()
                self._policy_net_optim.step()
            
    @torch.no_grad()
    def render(self, 
               record = False,
               record_path = None,
               steps = test_steps):
        state = env.reset()
        state = ls2tensor(state)
        total_rewards = 0
        if record:
            frames = []
        for _ in range(steps):
            # print(state)
            action = self.policy_net(state)
            next_states, rewards, _, _ = env.step(action)
            next_states = ls2tensor(next_states)
            
            total_rewards += ls2tensor(rewards).sum()
            if record:
                frame = env.render("rgb_array", visualize_when_rgb=True)
                frames.append(frame)
            else:
                env.render()
            
            state = next_states
        
        print(f"total rewards: {total_rewards}")
        if record:
            save_video(record_path, frames, fps=1 / env.scenario.world.dt)
        
        return total_rewards
    
class CPPO(PPO):
    critic_net_centralised = True
    policy_net_centralised = True
    share_params = False

class MAPPO(PPO):
    critic_net_centralised = False
    policy_net_centralised = True
    share_params = True

class IPPO(PPO):
    critic_net_centralised = False
    policy_net_centralised = False
    share_params = True
    
class HetIPPO(IPPO):
    share_params = False
    
if __name__ == "__main__":
    ppo = MAPPO()
    for ep in range(train_epochs):
        ppo.train()
        ppo.render(record=True, record_path=f"videos/{time.time()}.mp4")