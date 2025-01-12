from vmas import make_env

import torch
from torch import multiprocessing

method = 'IPPO'
load = False
train = True
render = True
critic_net_load_path = './models/{}_critic_net.pth'.format(method)
policy_net_load_path = './models/{}_policy_net.pth'.format(method)

# Devices
is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)

train_epochs = 3 # 演示轮数
test_steps = 1000 # 每轮演示时长

train_iters = 1000 # 每轮迭代次数
reset_iters = 100 # 每次数重置
train_steps = 10 # 每次迭代步数
update_epochs = 10 # 每次迭代学习轮数
lr = 3e-3

gamma = 0.95  # discount factor
clip_epsilon = 0.2

num_vmas_envs = 64
steps_per_env = 100
scenario_name = "navigation"
n_agents = 1
scenario_kwargs = dict(
    n_agents=n_agents,
)

env = make_env(scenario_name,
               **scenario_kwargs,
               num_envs= num_vmas_envs,
               continuous_actions=True,
               max_steps=steps_per_env,
               device=device,
               )
env.reset()