from collections import deque

import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

episode_durations = []


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


# Actor aka. policy network
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

    # Logarithmic probability of actions under the current policy:
    def log_prob(self, state, action):
        action_mean = self.forward(state)
        action_log_std = torch.zeros_like(action_mean, device=device)  # Assuming std = 1 for simplicity
        action_std = torch.exp(action_log_std)
        dist = torch.distributions.Normal(action_mean, action_std)
        return dist.log_prob(action).sum(-1, keepdim=True)


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


# Define the Critic (Value) Network
class PPOCritic(nn.Module):
    def __init__(self, state_dim):
        super(PPOCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DDPGCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DDPGCritic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        q = torch.relu(self.l1(torch.cat([state, action], 1)))
        q = torch.relu(self.l2(q))
        return self.l3(q)



# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_epsilon=0.2, update_epochs=10):
        self.actor = PPOActor(state_dim, action_dim).to(device)
        self.critic = PPOCritic(state_dim).to(device)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor(state).detach().cpu().numpy().flatten()
        return action

    def compute_returns(self, rewards, dones):
        returns = []
        R = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            R = reward + self.gamma * R * (1 - done)
            returns.insert(0, R)
        return returns

    def update(self, states, actions, log_probs, returns, advantages):
        states = np.array(states)  # Performance improvement
        actions = np.array(actions)  # Performance improvement

        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        log_probs = torch.FloatTensor(log_probs).to(device)
        returns = torch.FloatTensor(returns).to(device)
        advantages = torch.FloatTensor(advantages).to(device)

        for _ in range(self.update_epochs):
            # Compute the ratios
            new_log_probs = self.actor.log_prob(states, actions)
            ratios = torch.exp(new_log_probs - log_probs)

            # Compute the loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            value = self.critic(states)
            critic_loss = (returns - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss

            # Optimize the loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


class OrnsteinUhlenbeckNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action):
        return action + self.evolve_state()


class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = DDPGActor(state_dim, action_dim, max_action).to(device)
        self.actor_target = DDPGActor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = DDPGCritic(state_dim, action_dim).to(device)
        self.critic_target = DDPGCritic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = ReplayBuffer(max_size=1000000)
        self.noise = OrnsteinUhlenbeckNoise(action_dim)

        self.gamma = 0.99
        self.tau = 0.005

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, batch_size):
        state, action, reward, next_state, done = zip(*self.replay_buffer.sample(batch_size))

        state = np.array(state)  # Performance improvement
        action = np.array(action)  # Performance improvement
        next_state = np.array(next_state)  # Performance improvement

        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).reshape(-1, 1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).reshape(-1, 1).to(device)

        target_q = self.critic_target(next_state, self.actor_target(next_state))
        target_q = reward + (1 - done) * self.gamma * target_q
        target_q = target_q.detach()

        current_q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def add_experience(self, experience):
        self.replay_buffer.add(experience)


# Training the PPO Agent
def train_ppo(env, agent, num_episodes=7000, max_timesteps=1600):
    for episode in range(num_episodes):
        state, _ = env.reset()  # Initialize the state
        rewards, states, actions, log_probs, dones = [], [], [], [], []

        for t in range(max_timesteps):
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            log_prob = agent.actor.log_prob(torch.FloatTensor(state).unsqueeze(0).to(device),
                                            torch.FloatTensor(action).unsqueeze(0).to(device))

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob.item())
            dones.append(done)

            state = next_state

            if done:
                episode_durations.append(t + 1)
                break

        states = np.array(states)  # Performance improvement
        returns = agent.compute_returns(rewards, dones)
        advantages = np.array(returns) - agent.critic(torch.FloatTensor(states).to(device)).detach().cpu().numpy().flatten()
        agent.update(states, actions, log_probs, returns, advantages)

        if episode % 10 == 0:
            plot_durations(show_result=True)
            plt.ioff()
            plt.show()

        #if (episode + 1) % 10 == 0:
        #    print(f"Episode {episode + 1}: Reward = {sum(rewards)}")

        #if (episode + 1) % 1000 == 0:
        #    torch.save(agent.actor.state_dict(), f'ppo-models/ppo_actor_{episode + 1}.pth')
        #    print(f"Models saved for episode {episode + 1}")


def train_ddpg(env, agent, num_episodes=6500, max_timesteps=300, batch_size=64):
    for episode in range(num_episodes):
        state = env.reset()[0]
        episode_reward = 0

        env.render()

        for step in range(max_timesteps):
            action = agent.select_action(state)
            action = agent.noise.get_action(action)
            next_state, reward, done, _, _ = env.step(action)

            agent.add_experience((state, action, reward, next_state, float(done)))

            state = next_state
            episode_reward += reward

            if agent.replay_buffer.size() > batch_size:
                agent.train(batch_size)

            if done:
                episode_durations.append(step + 1)
                break

        print(f"Episode {episode + 1}, Reward: {episode_reward}")

        if episode % 500 == 0:
            plot_durations(show_result=True)
            plt.ioff()
            plt.show()

        #if (episode == 10):
        #    torch.save(agent.actor.state_dict(), f'ddpg-models/ddpg_actor_{episode + 1}.pth')

        #if (episode + 1) % 500 == 0:
        #    torch.save(agent.actor.state_dict(), f'ddpg-models/ddpg_actor_{episode + 1}.pth')


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)

    if show_result:
        plt.title('Result (DDPG)')
    else:
        plt.clf()
        plt.title('Training...')

    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

env = gym.make('BipedalWalker-v3', render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

ppo_agent = PPOAgent(state_dim, action_dim)
ddpg_agent = DDPGAgent(state_dim, action_dim, max_action)

train_ddpg(env, ddpg_agent)

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()
