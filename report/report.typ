// #import "@preview/cuti:0.3.0": show-cn-fakebold
// #show: show-cn-fakebold
// #set text(font: ("Times New Roman", "SimSun"))
// #show emph: text.with(font: ("（西文字体）", "STKaiti"))

#import "@preview/cetz:0.3.1": canvas, draw, tree

#set page(numbering: "1")

#let head_max = 9
#let arr = {
  let (arr, start_str) = (("",), "{2}")
  for i in range(2, head_max+1) {
    arr.push(start_str);start_str += ".{i}".replace("i",str(i+1))
  }
  arr
}
#import "@preview/numbly:0.1.0": numbly
#set heading(numbering: numbly(..arr))
#show heading: it => [
  #set align(center)
  #set text(32pt * (1 - it.level / head_max))
  #it
]

// = 强化学习课程期末项目——VMAS 相关
= Reinforcement Learning Course Final Project - VMAS Related
// #align(center)[21311475 谢文龙]
#align(center)[21311475 Xie Wenlong]

// #show heading: it => [#it; #par(h(0.0em))]
#set par(justify: true, first-line-indent: 2em)

// == 摘要
== Abstract

#emph[
  
// 本项目中，笔者阅读了 VMAS 的源码，并对其进行了整理和解读；VMAS 是一个用于多智能体强化学习（MARL）的仿真环境，其源码结构清晰，易于理解。随后笔者实现了多智能体情况下的多个 PPO 算法，包括 CPPO、MAPPO、IPPO 和 HetIPPO。
In this project, the author read the source code of VMAS and compiled and explained it; VMAS is a simulation environment for Multi-Agent Reinforcement Learning (MARL), its source code structure is clear and easy to understand. Subsequently, the author implemented multiple PPO algorithms under the multi-agent situation, including CPPO, MAPPO, IPPO and HetIPPO.

// 项目代码见 https://github.com/ZojkLoyn/MARL_2024_Autumn ，其中含有 `my_job` 与 `torchrl` 两个分支，分别为笔者使用 `VMAS` 库实现与使用 `torchrl` 库实现的代码。
The project code is available at https://github.com/ZojkLoyn/MARL_2024_Autumn, which contains two branches: `my_job` and `torchrl`, respectively, the code implemented by the author using the `VMAS` library and the code implemented using the `torchrl` library.
]

#show: rest => columns(2, rest)

// == VMAS 源码阅读与整理
== VMAS Source Code Reading and Compilation

// VMAS 为 Vectorized Multi Agent Simulator 的缩写，是一个用于多智能体强化学习（MARL）的仿真环境。本报告将介绍 VMAS 的源码结构，并对其中的关键组件进行详细解读。
VMAS, which stands for Vectorized Multi Agent Simulator, is a simulation environment for Multi-Agent Reinforcement Learning (MARL). This report will introduce the source code structure of VMAS and provide detailed explanations of its key components.

// === 核心概念
=== Core Concepts

// - *环境*：环境的概念来自 OpenAI Gym 库，代表一个马尔可夫决策过程。其中一个环境里有 `n_agents` 个智能体，`num_envs` 个世界，预制的场景与观察空间、奖励、终止条件、动作空间等。
- *Environment*：The concept of an environment comes from the OpenAI Gym library, representing a Markov decision process. One environment contains `n_agents` agents, `num_envs` worlds, pre-set scenarios, observation spaces, rewards, termination conditions, and action spaces, etc.
// - *世界*：一个 VMAS 环境中，每个世界代表一个的 MARL 环境。每个世界都包含 `n_agents` 个智能体，每个智能体有各自的观察空间、动作空间、奖励函数等。此设计是为了充分利用 GPU 的并行计算能力，每个世界都独立运行，互不干扰。
- *World*：In a VMAS environment, each world represents a MARL environment. Each world contains `n_agents` agents, each agent has its own observation space, action space, reward function, etc. This design is to fully utilize the parallel computing capabilities of the GPU, each world runs independently and does not interfere with each other.
// - *智能体*：智能体是 MARL 环境中的主体，每个智能体都有各自的状态、观察空间、动作空间、奖励函数等，会通过动作改变状态、影响环境，并从环境中获取奖励。每个世界都有 `n_agents` 个智能体，也可以理解为每个智能体在每个世界中都有一个副本。
- *Agent*：Agent is the subject in the MARL environment. Each agent has its own state, observation space, action space, reward function, etc. It changes the state and affects the environment through actions, and obtains rewards from the environment. Each world has `n_agents` agents, which can also be understood as each agent has a copy in each world.
// - *动作*：动作是智能体在环境中执行的操作，每个动作都会改变智能体的状态，并影响环境。
- *Action*：Action is the operation that the agent performs in the environment. Each action changes the state of the agent and affects the environment.
// - *状态*：状态是环境中的状态，每个状态都包含智能体的位置、速度、方向等信息。
- *State*：State is the state in the environment. Each state contains the position, velocity, direction, etc. of the agent.
// - *观察*：观察是智能体对环境的感知，即为获取一个或数个智能体的状态信息。
- *Observation*：Observation is the perception of the agent to the environment, which is to obtain the state information of one or more agents.
// - *奖励*：奖励是智能体从环境中获得的奖励，根据智能体的动作和环境的状态，智能体可以获得不同的奖励。
- *Reward*：Reward is the reward obtained by the agent from the environment. According to the agent's actions and the state of the environment, the agent can obtain different rewards.

// === 文件结构
=== File Structure

// 其中 `simulator` 目录为模拟器主要代码目录，其中 `core` 文件为代码核心。
The `simulator` directory is the main code directory of the simulator, and the `core` file is the core of the code.

// `core` 文件分别包含以下类：
The `core` file contains the following classes:

// ==== `TorchVectorizeObject` 类
==== Class `TorchVectorizeObject`

// `TorchVectorizeObject` 是 VMAS 中所有可向量化对象的基类，其 `batch_dim` 属性表示批数量，通常对应一个 VMAS 环境中世界的数量 `num_envs`。
`TorchVectorizeObject` is the base class of all vectorizable objects in VMAS. Its `batch_dim` attribute represents the batch size, which is usually the number of worlds in a VMAS environment `num_envs`.

// 其包含 `to` 方法，用于将对象转移到指定设备。
It contains the `to` method, which is used to transfer the object to the specified device.

// ==== `Shape` 类
==== Class `Shape`

// `Shape` 类用于表示对象形状。
The `Shape` class is used to represent the shape of an object.

// 有子类 `Box, Sphere, Line`，分别表示盒、球、线。
There are subclasses `Box`, `Sphere`, and `Line`, which represent boxes, spheres, and lines, respectively.

// ==== `EntityState` 类
==== Class `EntityState`

// 用于表示实体状态，有 `pos, vel, ang_vel, rot` 四个属性，分别表示位置、线速度、角速度与旋转矩阵。
It is used to represent the state of an entity, with four attributes `pos`, `vel`, `ang_vel`, and `rot`, representing position, linear velocity, angular velocity, and rotation matrix, respectively.

// 其子类 `AgentState` 则专门表示智能体状态，有 `c, force, torque` 四个属性，用于表示交流语言、受力与受力矩。
The subclass `AgentState` specifically represents the state of an agent, with four attributes `c`, `force`, `torque`, used to represent communication language, force, and torque, respectively.

// ==== `Action` 类
==== Class `Action`

// 用于表示单个 `Agent` 的动作，有 `u, c` 两个主要属性，分别表示物理动作和交流动作。
It is used to represent the action of a single `Agent`, with two main attributes `u` and `c`, representing physical actions and communication actions, respectively.

// ==== `Entity` 类
==== Class `Entity`

// 用于表示实体，有质量等物理属性，有可碰撞可移动等性质属性，此外也带有一个 `state` 属性表示位置相关属性，有函数可以设置 `state` 相关信息。
It is used to represent an entity, with physical properties such as mass, and properties such as collision and mobility. In addition, it also has a `state` attribute representing position-related properties, with functions to set `state` related information.

// 其有两个子类
There are two subclasses:

// - `Landmark`：表示预设的标志物，其与 `Entity` 无异。主要用于与 `Agent` 区分。
- `Landmark` represents a preset landmark, which is identical to `Entity`. It is mainly used to distinguish between `Agent` and `Landmark`.
// - `Agent`：表示智能体，有描述对世界的观察、感受能力的属性，有受力的约束属性（会在世界step中处理），有动态与动作相关信息。
- `Agent` represents an agent, with attributes describing the agent's ability to observe and feel the world, with constraints on force (which will be processed in `step` of the world), and dynamics-related and action-related information.

// ==== 其他文件
==== Other Files

// - `Joint` 文件含 `Joint` 关节相关内容，表示实体如何根据动作进行状态变化。
- `Joint` file contains `Joint` joint-related content, representing how entities change their state according to actions.
// - `Dynamics` 文件夹中包含实体动态相关内容，表示实体如何根据动作进行受力状态变化。
- The `Dynamics` folder contains entity dynamics-related content, representing how entities change their force state according to actions.
// - `Environment` 文件夹中包含环境相关内容，表示一个环境，包含一个场景，一个获取信息用的函数，一个 `step` 函数用于输入动作并执行一次环境更新，与一些渲染等功能。
- The `Environment` folder contains environment-related content, representing an environment, containing a scenario, functions for obtaining information, a `step` function for inputting actions and executing an environment update, and some rendering functions.
// - `Scenario` 文件包含场景基类，规定场景应有一个世界，与创建世界、重置指定环境中的世界、智能体对环境的观察、收到的奖励、处理动作等函数。在 `Scenarios` 文件夹中包含具体的预设场景。
- The `Scenario` file contains the base class of the scenario, specifying that the scenario should have a world, with functions for creating a world, resetting the world in a specified environment, the agent's observation of the environment, the received reward, and processing actions. Specific preset scenarios are contained in the `Scenarios` folder.

// === 环境的使用
=== Usage of Environment

// VMAS 的环境类似于 OpenAI Gym 的环境，可以用于训练智能体。只需要掌握些许基础概念即可使用，如下所示。
VMAS's environment is similar to OpenAI Gym's environment and can be used to train agents. You only need to master a few basic concepts to use it, as shown below.

// ==== 初始化环境
==== Initialize Environment

// 使用 `make_env` 函数创建一个环境，其中有如下重要参数
use `make_env` function to create an environment, with the following important parameters

// - `scenario`：场景名称，能够指定使用哪个预设场景。
- `scenario`：Scenario name, which can specify which preset scenario to use.
// - `num_envs`：世界数量，即并行环境数量。
- `num_envs`：Number of worlds, i.e., number of parallel environments.
// - `device`：设备，用于指定使用 GPU 还是 CPU。
- `device`：Device, used to specify whether to use GPU or CPU.
// - 世界相关信息
- World-related information
  // - `max_steps`：每个世界最多执行多少步。
  - `max_steps`：The maximum number of steps each world can execute.
  // - `continuous_actions`：是否在世界结束后自动重置。
  - `continuous_actions`：Whether to automatically reset after the world ends.
// - 场景额外参数
- Additional parameters of the scenario
  // - `n_agents`：智能体数量。在部分预设场景中，智能体数量是固定的，因此无法指定。
  - `n_agents`：Number of agents. In some preset scenarios, the number of agents is fixed and cannot be specified.

// ==== 观察空间与动作空间
==== Observation Space and Action Space

// VMAS 的环境有观察空间与动作空间，分别表示智能体对环境的观察与智能体可以执行的动作。
VMAS's environment has observation space and action space, respectively representing the agent's observation of the environment and the actions that the agent can execute.

// 对于各智能体平等的场景，每个智能体的观察空间与动作空间相同，因此只需要获取一个智能体的空间。可用 ```python env.observation_space[0].shape[0]``` 获取单一智能体在一个世界中的观察空间的张量维度数量 `observation_dim`；可用 ```python env.action_space[0].shape[0]``` 获取单一智能体在一个世界中的动作空间的张量维度数量 `action_dim`。
For scenarios where all agents are equal, the observation space and action space of each agent are the same, so you only need to get the space of one agent.
You can use ```python env.observation_space[0].shape[0]``` to get the number of tensor dimensions of the observation space of a single agent in a world `observation_dim`;
you can use ```python env.action_space[0].shape[0]``` to get the number of tensor dimensions of the action space of a single agent in a world `action_dim`.

// ==== 重置与迭代
==== Reset and step

// VMAS 的环境支持重置与迭代，分别用于初始化环境与执行智能体的动作。
VMAS's environment supports reset and step, respectively used to initialize the environment and execute the agent's actions.

// 使用 ```python env.reset()``` 重置环境，返回一个观察张量 `state`。
use ```python env.reset()``` to reset the environment, returning an observation tensor `state`.

// 使用 ```python env.step(actions)``` 输入动作张量 `action` 并迭代环境，返回一个观察张量 `next_state`，奖励张量 `reward`，终止张量 `done`，信息张量 `info`。
use ```python env.step(actions)``` to input the action tensor `action` and iterate the environment, returning an observation tensor `next_state`, a reward tensor `reward`, a termination tensor `done`, and an information tensor `info`.

// - 动作张量 `action` 用于表示智能体在世界里的动作，格式为智能体索引、世界索引、动作张量，即 $ #"n_agents"_#"tuple" times [#"num_envs", #"action_dim"]_#"tensor"$ 
- action tensor `action` is used to represent the agent's actions in the world, with the format of agent index, world index, and action tensor, i.e. $ #"n_agents"_#"tuple" times [#"num_envs", #"action_dim"]_#"tensor"$
// - 观察张量 `state` 与 `next_state` 表示智能体在世界里的观察，格式均为智能体索引、世界索引、观察张量，即 $ #"n_agents"_#"tuple" times [#"num_envs", #"observation_dim"]_#"tensor" $
- observation tensor `state` and `next_state` represent the agent's observation in the world, with the same format of agent index, world index, and observation tensor, i.e. $ #"n_agents"_#"tuple" times [#"num_envs", #"observation_dim"]_#"tensor" $
// - 奖励张量 `reward` 表示在本次迭代中智能体在世界里的收获，格式为智能体索引、世界索引，即 $ #"n_agents"_#"tuple" times [#"num_envs"]_#"tensor" $
- reward tensor `reward` represents the agent's harvest in the world in this iteration, with the format of agent index, world index, i.e. $ #"n_agents"_#"tuple" times [#"num_envs"]_#"tensor" $
// - 终止张量 `done` 表示迭代后世界是否结束，元素类型为 ```python bool```，格式为世界索引 $ [#"num_envs"]_#"tensor" $
- termination tensor `done` represents whether the world ends after the iteration, with the element type of ```python bool``` and the format of world index $ [#"num_envs"]_#"tensor" $
// - 信息张量 `info` 用于记录奖励张量的具体信息，不同的预设场景有不同的格式，如 `Balance` 场景则为智能体索引、`pos` 奖励与 `ground` 奖励，即 $ #"n_agents"_#"tuple" times cases(\""pos_rew"\": [#"num_envs"]_#"tensor",\  \""ground_rew"\": [#"num_envs"]_#"tensor") $
- information tensor `info` is used to record the specific information of the reward tensor, with different formats for different preset scenarios, such as the `Balance` scenario, which is agent index, `pos` reward and `ground` reward, i.e. $ #"n_agents"_#"tuple" times cases(\""pos_rew"\": [#"num_envs"]_#"tensor",\  \""ground_rew"\": [#"num_envs"]_#"tensor") $

// == 近端策略优化
== Proximal Policy Optimization

// PPO 为 Proximal Policy Optimization 的缩写，是通过一种智能体训练的思想。通过迭代执行智能体的动作，并收集智能体的经验，然后使用这些经验来更新智能体的策略。

PPO stands for Proximal Policy Optimization, which is an agent training idea. By iteratively executing the agent's actions and collecting the agent's experiences, and then using these experiences to update the agent's policy.

// === 单智能体 PPO
=== Single Agent PPO

// PPO 算法基于 Actor-Critic 框架，其中 Actor 为策略网络，Critic 为评论网络。策略网络用于生成智能体的动作，评论网络用于评估智能体的动作的好坏。PPO 算法通过迭代执行智能体的动作，并收集智能体的经验，然后使用这些经验来更新智能体的策略。
PPO algorithm is based on the Actor-Critic framework, where the Actor is the policy network and the Critic is the value network. The policy network is used to generate the agent's actions, and the value network is used to evaluate the good or bad of the agent's actions. The PPO algorithm iteratively executes the agent's actions and collects the agent's experiences, and then uses these experiences to update the agent's policy.

// 其中 Actor 的输入为智能体的观察，输出为智能体的动作。如@single_ppo_actor 所示。
The input of the Actor is the agent's observation, and the output is the agent's action. As shown in @single_ppo_actor.

#let net = (content) => box(pad(content, x:7pt, y: 5pt), inset: .5pt,outset: .5pt, stroke: gray)
#let actor = {
  let I = [$ ["num_envs", "observation_dim"]_"tensor" $]
  let H = [Hidden Layer]
  let O = [$["num_envs", #strong("action_dim")]_"tensor" $]
  (I, (H, O))
}
#let draw_net = (data) => canvas( {
  import draw: *

  // set-style(content: (padding: .2),
  //   fill: gray.lighten(70%),
  //   stroke: gray.lighten(70%))

  tree.tree(data, spread: 2.5, grow: 1.5, draw-node: (node, ..) => {
    // circle((), radius: .45, stroke: none)
    content((), net(node.content))
  }, 
  draw-edge: (from, to, ..) => {
    line((a: from, number: .35, b: to),
         (a: to, number: .4, b: from), mark: (end: ">"))
  },
   name: "tree")

  // Draw a "custom" connection between two nodes
})
// #data
// #figure(draw_net(actor), caption: "单智能体的 Actor 网络结构")
#figure(draw_net(actor), caption: "The structure of the Actor network in single agent")<single_ppo_actor>

// Critic 的输入为智能体的观察和动作，输出为智能体的动作价值。如@single_ppo_critic 所示。
The input of the Critic is the agent's observation and action, and the output is the agent's action value. As shown in @single_ppo_critic.

#let critic = {
  let I = [$ ["num_envs", "observation_dim"]_"tensor" $]
  let H = [Hidden Layer]
  let O = [$["num_envs",#strong(" 1")]_"tensor" $]
  (I, (H, O))
}
// #figure(draw_net(critic), caption: "单智能体的 Critic 网络结构")
#figure(draw_net(critic), caption: "The structure of the Critic network in single agent")<single_ppo_critic>

// 则算法的流程如下：
The algorithm's process is as follows:

// 1. 初始化 Actor 和 Critic 网络，并设置学习率。
1. Initialize the Actor and Critic networks and set the learning rate.

// 2. 在环境中由 Actor 决策执行智能体的动作，并收集智能体的经验。其中经验包括智能体的观察、动作、奖励和下一个状态，即 $(s_t, a_t, r_t, s_(t+1))$。
2. In the environment, the Actor decides the agent's action and collects the agent's experience. The experience includes the agent's observation, action, reward, and next state, that is, $(s_t, a_t, r_t, s_{t+1})$.

// 3. 通过经验计算优势函数 $A_t = r_t + gamma V(s_{t+1}) - V(s_t)$，其中 $gamma$ 为超参数折扣因子，$V(s)$ 为 Critic 网络需要学习的值函数。
3. Calculate the advantage function $A_t = r_t + gamma V(s_{t+1}) - V(s_t)$ through the experience, where $gamma$ is the hyperparameter discount factor, and $V(s)$ is the value function that the Critic network needs to learn.

// 4. 通过优势函数训练 Critic 网络，使其能够更好地估计动作价值。
4. Train the Critic network through the advantage function, so that it can better estimate the action value.

// 5. 使用特殊的目标函数 $r_t(theta) = pi_theta(a_t|s_t) / (pi_(theta,"old")(a_t|s_t))$ 与 clip 损失函数结合，训练 Actor 网络，使其能够更好地估计动作概率。
5. Use the special objective function $r_t(theta) = pi_theta(a_t|s_t) / (pi_(theta,"old")(a_t|s_t))$ combined with the clip loss function to train the Actor network, so that it can better estimate the action probability.

// 6. 重复步骤 3-5，直到达到预定的训练步数或达到预定的收敛条件。
6. Repeat steps 3-5 until the predetermined training steps or convergence conditions are reached.

// 7. 重复步骤 2-6，直到达到预定的训练轮数或达到预定的收敛条件。
7. Repeat steps 2-6 until the predetermined training rounds or convergence conditions are reached.

// === 多智能体 PPO
=== Multiple Agent PPO

// 在多智能体情况下，Actor 网络与 Critic 网络将会更加复杂，负责处理多个智能体的任务。
In the multi-agent case, the Actor network and Critic network will be more complex, responsible for handling tasks for multiple agents.

// ==== 多智能体网络结构
==== Multi-Agent Network Structure

// 对于多智能体，我们将定义整体网络为其输入为所有智能体的状态，输出为所有智能体的信息的网络。
For multi-agents, we will define the overall-network as a network whose input is the state of all agents and whose output is the information of all agents.

// 其网络结构为@multi_ppo_critic。
Its network structure is @multi_ppo_critic.

#let multi_overall = {
  let I = [$ [#strong("n_agents"), "num_envs", "observation_dim"]_"tensor" $]
  let H = [Hidden Layer]
  let O = [$[#strong("n_agents"), "num_envs", "output_dim"]_"tensor" $]
  (I, (H, O))
}

// #figure(draw_net(multi_overall), caption: "多智能体的整体网络结构")
#figure(draw_net(multi_overall), caption: "The structure of the overall-network in multiple agent")<multi_ppo_critic>

// 而其中根据分布的具体设计，隐藏层分为三种类型。
And the hidden layer is divided into three types according to the specific design of the distribution.

// ===== 集中式
===== Centralized <centralised>

// 在集中式中，隐藏层能同时获取处理所有智能体的状态信息，并且为所有智能体的决策。
In the centralized mode, the hidden layer can simultaneously obtain and process the state information of all agents and make decisions for all agents.

// 即将 $"num_env"$ 视作批属性，而将 $"n_agents"*"observation_dim"$ 视作特征属性。
The $"num_env"$ is regarded as the batch attribute, and the $"n_agents"*"observation_dim"$ is regarded as the feature attribute.

// 其网络结构为 @multi_centralised 。
Its network structure is @multi_centralised.

#let multi_centralised = {
  let I = [$["num_envs"] times ["n_agents"* "observation_dim"]_"tensor" $]
  let H = [Hidden Layer]
  let O = [$["num_envs"] times ["n_agents"* "output_dim"]_"tensor" $]
  (I, (H, O))
}

// #figure(draw_net(multi_centralised), caption: "集中式网络结构")
#figure(draw_net(multi_centralised), caption: "The structure of the centralized network")<multi_centralised>

// ===== 共享参数
===== Shared Parameters <shared>

// 在共享参数中，隐藏层只能获取处理单个智能体的状态信息，为单个智能体做决策。但是，其参数会被所有智能体共享。
In the shared parameters, the hidden layer can only obtain and process the state information of a single agent and make decisions for a single agent. However, its parameters are shared by all agents.

// 即将 $"num_env" * "n_agents"$ 视作批属性，而将 $"observation_dim"$ 视作特征属性。
The $"num_env" * "n_agents"$ is regarded as the batch attribute, and the $"observation_dim"$ is regarded as the feature attribute.

// 其网络结构为 @multi_shared 。
Its network structure is @multi_shared.

#let multi_shared = {
  let I = [$["n_agents" * "num_env"] times ["observation_dim"]_"tensor" $]
  let H = [Hidden Layer]
  let O = [$["n_agents" * "num_env"] times [ "output_dim"]_"tensor" $]
  (I, (H, O))
}

// #figure(draw_net(multi_shared), caption: "共享参数网络结构")
#figure(draw_net(multi_shared), caption: "The structure of the shared parameters network")<multi_shared>

// ===== 独立参数
===== Independent Parameters<independent>

// 在独立参数中，隐藏层只能获取处理单个智能体的状态信息，为单个智能体做决策。而且每个智能体独占一个网络。
In the independent parameters, the hidden layer can only obtain and process the state information of a single agent and make decisions for a single agent. And each agent occupies a network independently.

// 即将 $"num_env"$ 视作批属性，而将 $"observation_dim"$ 视作特征属性。
The $"num_env"$ is regarded as the batch attribute, and the $"observation_dim"$ is regarded as the feature attribute.

// 其网络结构为 @multi_independent 。
Its network structure is @multi_independent.

#let multi_independent = {
  let I = [$["num_env"] times ["observation_dim"]_"tensor" $]
  let H = [Hidden Layer]
  let O = [$["num_env"] times ["output_dim"]_"tensor" $]
  (I, (H, O))
}

// #figure(draw_net(multi_independent), caption: "独立参数网络结构")
#let network = [n_agents $times$] + "\n" + box(pad(draw_net(multi_independent), x:7pt, y: 6pt), inset: .5pt,outset: .5pt, stroke: 1pt)
#figure(network, caption: "The structure of the independent parameters network")<multi_independent>

// ==== 多智能体 PPO 算法
==== Multi-agent PPO Algorithm

// 各类多智能体 PPO 算法，其核心思想与单智能体 PPO 算法无异。其差别在于 Actor 网络与 Critic 网络的设计。
The core idea of various multi-agent PPO algorithms is the same as that of the single-agent PPO algorithm. The difference lies in the design of the Actor network and the Critic network.

// 以下为 VMAS 论文原文中提到的四种多智能体 PPO 算法的定义，其取决于 Actor 网络与 Critic 网络的设计。
The following are the definitions of four multi-agent PPO algorithms mentioned in the VMAS paper, which depend on the design of the Actor network and the Critic network.

===== CPPO

- Actor: Centralized [@centralised]
- Critic: Centralized [@centralised]

===== MAPPO

- Actor: Independent [@independent]
- Critic: Centralized [@centralised]

===== IPPO

- Actor: Shared Parameters [@shared]
- Critic: Shared Parameters [@shared]

===== HetIPPO

- Actor: Independent [@independent]
- Critic: Independent [@independent]

// == 项目实现
== Project Implementation

// 在项目实现中，我主要针对 Balance 预设场景进行了实现，并实现了 CPPO、MAPPO、IPPO、HetIPPO 四种算法。
In the project implementation, I mainly implemented the CPPO, MAPPO, IPPO, and HetIPPO algorithms for the Balance scenario.


// 实现代码位于 https://github.com/ZojkLoyn/MARL_2024_Autumn 中。其中包含了 `torchrl` 分支，使用了 `TorchRL` 库实现，其能够训练得到一个比较好的结果；以及 `my_job` 分支，为个人直接使用 `VMAS` 库实现，但是其存在未解决的 bug 。
The implementation code is located at https://github.com/ZojkLoyn/MARL_2024_Autumn. Among them, the `torchrl` branch uses the `TorchRL` library to implement, which can train a good result; and the `my_job` branch is implemented directly using the `VMAS` library, but there are unresolved bugs.

// === 设计细节
=== Design Details

// 其中存在设计细节为，VMAS 要求 Actor 网络输出的动作限制在 $[-1, 1]$ 范围内；而如 ```python torchrl.MultiAgentMLP``` 网络的最后一层为线性层，其可能会输出违规值；因此在网络输出层后应当添加合适的激活层。但是不同激活层的效果不同。
There is a design detail that VMAS requires the output of the Actor network to be limited within the range of $[-1, 1]$; while the last layer of the ```python torchrl.MultiAgentMLP``` network is a Linear Layer, which may output illegal values; therefore, an appropriate activation layer should be added after the network output layer. However, the effect of different activation layers is different.

// - `ReLU` 等激活函数，其值域与 $[-1, 1]$ 不一致；因此依然会产生违规值。
// - `Softmax` 等向量激活函数，其会影响其他动作的输出；因此不适用。
// - `Tanh` 等激活函数，其值域为 $[-1, 1]$，但是其输出值分布不均，且无法达到边界；因此无法产生边界策略，容易出现问题。
// - `HardTanh` 激活函数，其值域为 $[-1, 1]$，且其输出值分布均等，并且能够轻易达到边界；因此其能够产生边界策略，且效果较好。
- `ReLU` activation functions, whose value range is inconsistent with $[-1, 1]$; therefore, illegal values will still be produced.
- `Softmax` and other vector activation functions, which will affect the output of other actions; therefore, they are not applicable.
- `Tanh` and some activation functions, whose value range is $[-1, 1]$, but their output value distribution is uneven, and they cannot reach the boundary; therefore, they cannot produce boundary policies, and problems may easily occur.
- `HardTanh` , whose value range is $[-1, 1]$, and their output value distribution is uniform, and they can easily reach the boundary; therefore, they can produce boundary policies, and the effect is good.

=== Branch `my_job`

// 在 `my_job` 分支中，我使用 `VMAS` 库模拟训练环境，使用 `PyTorch` 库实现 Actor 网络与 Critic 网络。
In the `my_job` branch, I use the `VMAS` library to simulate the training environment and use the `PyTorch` library to implement the Actor network and Critic network.

// 不过其存在 bug，训练没有产生效果，因此此处仅作为工作量证明。
However, there is a bug in it, and the training has no effect, so it is only used as a proof of work here.

=== Branch `torchrl`

// 在 `torchrl` 分支中，我使用 `TorchRL` 库实现了训练环境模拟、神经网络实现以及训练过程。
In the `torchrl` branch, I implemented the training environment simulation, neural network implementation, and training process using the `TorchRL` library.

#figure(image("img/cppo.png"), caption: "CPPO Training Results") <cppo_results>
// #figure(image("img/cppo_render.png"), caption: "它还有进步空间") <cppo_render>
#figure(image("img/cppo_render.png"), caption: "It still has room for improvement") <cppo_render>

// 其能够在 1000 个训练周期内，训练得到一个比较好的结果。训练结果见 @cppo_results。不过其仍会出现一些问题，例如运输路径与目标位置出现偏差，见 @cppo_render。
It can train a good result within 1000 training iters. The training results are shown in @cppo_results. However, it still has some problems, such as the deviation of the transport path and the target location, see @cppo_render.

== Conclusion

// 在本次实验中，我实现了多种 MARL 算法，并对其进行了实验验证。实验结果表明，PPO 算法在多智能体任务中表现良好，能够训练得到一个比较好的结果。但是，我的实现仍存在一些问题，例如运输路径与目标位置出现偏差，需要进一步优化。
In this experiment, I implemented multiple MARL algorithms and verified them through experiments. The experimental results show that the PPO algorithm performs well in multi-agent tasks and can train a good result. However, my implementation still has some problems, such as the deviation of the transport path and the target location, which need further optimization.