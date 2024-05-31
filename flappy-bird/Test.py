import flappy_bird_gymnasium
import gymnasium as gym
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as func


class DQN(nn.Module):
    def __init__(self, num_observations, num_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(num_observations, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, num_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = func.relu(self.layer1(x))
        x = func.relu(self.layer2(x))
        return self.layer3(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)

state, info = env.reset()
num_actions = env.action_space.n  # Number of actions
num_observations = len(state)  # Number of state observations

policy_net = DQN(num_observations, num_actions).to(device)
policy_net.load_state_dict(torch.load("non-lidar-models/FlappyBirdModel-Episode20000.pt"))

state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
env.render()

for t in count():
    action = policy_net(state).max(1).indices.view(1, 1)
    observation, reward, terminated, truncated, _ = env.step(action.item())
    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

    if terminated:
        break
