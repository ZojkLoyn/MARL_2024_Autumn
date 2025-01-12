from config import *
from enum import Enum
import torch
import abc
class StateDealer(Enum):
    CENTRALISED = 0
    SHARED = 1
    INDIVIDUAL = 2
    
class GeneralNet(torch.nn.Module, abc.ABC):
    @property
    @abc.abstractmethod
    def output_dim_basic(self):
        return None
    
    @abc.abstractmethod
    def get_net(self, input_dim, output_dim):
        pass
    
    def __init__(self, centralised = False, share_params = True):
        super().__init__()
        self.state_dim = env.observation_space[0].shape[0]
        if centralised:
            self.deal = StateDealer.CENTRALISED
            input_dim = n_agents * self.state_dim
            output_dim = n_agents * self.output_dim_basic
            self.net = self.get_net(input_dim, output_dim)
        elif share_params:
            self.deal = StateDealer.SHARED
            input_dim = self.state_dim
            output_dim = self.output_dim_basic
            self.net = self.get_net(input_dim, output_dim)
        else:
            self.deal = StateDealer.INDIVIDUAL
            input_dim = self.state_dim
            output_dim = self.output_dim_basic
            self.net = torch.nn.ModuleList([self.get_net(input_dim, output_dim) for _ in range(n_agents)]) 
        
    def forward(self, states):
        # states: [n_agents, n_envs, state_space]
        # return: [n_agents, n_envs, output_dim]
        if self.deal == StateDealer.CENTRALISED:
            # input: [n_envs, n_agents * state_space]
            states = states.permute(1, 0, 2).reshape(-1, n_agents * self.state_dim)
            # output: [n_envs, n_agents * output_dim]
            result = self.net(states)
            result = result.reshape(-1, n_agents, self.output_dim_basic).permute(1, 0, 2)
        elif self.deal == StateDealer.SHARED:
            # input: [n_agents * n_envs, state_space]
            states = states.reshape(-1, self.state_dim)
            # output: [n_agents * n_envs, output_dim]
            result = self.net(states)
            result = result.reshape(n_agents, -1, self.output_dim_basic)
        elif self.deal == StateDealer.INDIVIDUAL:
            # input: [n_agents][n_envs, state_space]
            result = []
            # output: [n_agents][n_envs, output_dim]
            for i in range(n_agents):
                result.append(self.net(states[i]))
            result = torch.stack(result)
        return result
    

class CriticNet(GeneralNet):
    @property
    def output_dim_basic(self):
        return 1

    def get_net(self, input_dim, output_dim):
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, output_dim)
        )
                

class PolicyNet(GeneralNet):
    @property
    def output_dim_basic(self):
        if not hasattr(self, '__output_dim_basic') or self.__output_dim_basic is None:
            self.__output_dim_basic = env.action_space[0].shape[0]
        return self.__output_dim_basic
    
    def get_net(self, input_dim, output_dim):
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, output_dim),
            torch.nn.Hardtanh(),
        )