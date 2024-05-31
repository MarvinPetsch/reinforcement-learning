import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the Actor (Policy) Network
class PPOActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x


class DDPGActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(DDPGActor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


def load_ppo_model(actor_model_path, state_dim, action_dim):
    actor = PPOActor(state_dim, action_dim).to(device)
    actor.load_state_dict(torch.load(actor_model_path))
    actor.eval()  # Set the model to evaluation mode
    return actor


def load_ddpg_model(actor_model_path, state_dim, action_dim, max_action):
    actor = DDPGActor(state_dim, action_dim, max_action).to(device)
    actor.load_state_dict(torch.load(actor_model_path))
    actor.eval()  # Set the model to evaluation mode
    return actor


def run_game(env, actor, max_timesteps=1600):
    state, _ = env.reset()
    total_reward = 0
    for t in range(max_timesteps):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = actor(state_tensor).detach().cpu().numpy().flatten()
        next_state, reward, done, _, _ = env.step(action)
        env.render()
        total_reward += reward
        state = next_state
        if done:
            break
    print(f"Total Reward: {total_reward}")


if __name__ == "__main__":
    env = gym.make('BipedalWalker-v3', render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Load the trained models
    actor_model_path = 'ddpg-models/ddpg_actor_6500.pth'
    actor = load_ddpg_model(actor_model_path, state_dim, action_dim, max_action)

    # Run the game using the loaded models
    run_game(env, actor)
    env.close()
