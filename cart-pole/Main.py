# Comment: I took things from the pytorch-dqn-tutorial that also implements cart-pole

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, num_observations, num_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(num_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = func.relu(self.layer1(x))
        x = func.relu(self.layer2(x))
        return self.layer3(x)


env = gym.make("CartPole-v1", render_mode="human")

# Set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

episode_durations = []

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transitions are state changes of the environment, coupled with the change-inducing action and its reward
Transition = namedtuple(
    'Transition',
    ('state', 'action', 'next_state', 'reward')
)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Get samples of previous transitions. Samples are selected randomly to decorrelate transitions that build
        up a batch to stabilize and improve training."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor for future reward estimations --> The lower, the less important estimated future rewards

# EPSILON = The probability of choosing a random action instead of the "best" one, enhancing exploration
# EPSILON_START its starting value
# EPSILON_END its final value
# EPSILON_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay

# TAU is the update rate of the target network
# LEARNING_RATE is the learning rate of the ``AdamW`` optimizer (how much should new learnings count)
BATCH_SIZE = 128
GAMMA = 0.99  #
EPSILON_START = 0.9
EPSILON_END = 0.05  # Exploration decays while the model becomes better at identifying the best action
EPSILON_DECAY = 1000  # I might want to explain this in the presentation
TAU = 0.005
LEARNING_RATE = 1e-4

state, info = env.reset()
num_actions = env.action_space.n  # Number of actions
num_observations = len(state)  # Number of state observations

policy_net = DQN(num_observations, num_actions).to(device)  # The main neural network that approximates the Q-value
target_net = DQN(num_observations, num_actions).to(device)  # A neural network enhancing the policy networks learning
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()  # 0 <= sample <= 1

    # Calculate the current Epsilon
    epsilon_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * steps_done / EPSILON_DECAY)

    steps_done += 1

    if sample > epsilon_threshold:  # Predict the best action to take
        with torch.no_grad():
            # policy_net(state) -> tensor with approximated rewards for action 1 & 2
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:  # Randomize the action to take
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)

    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')

    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    # Take 100 episode averages and integrate them into the plot
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


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)

    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()  # Updates parameters


if torch.cuda.is_available():
    print("Cuda is available")
    num_episodes = 600
else:
    print("Cuda is not available")
    num_episodes = 600

for episode in range(num_episodes):
    print(f"Episode: {episode}")

    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    env.render()

    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # --> Update the target network's weights to the policy network's weights
        #     (not fully tho --> see TAU aka. learning rate)
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)

        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            break

    '''
    match episode:
        case 99:
            torch.save(policy_net.state_dict(), 'models/CartPoleModel-Episode100.pt')
        case 199:
            torch.save(policy_net.state_dict(), 'models/CartPoleModel-Episode200.pt')
        case 299:
            torch.save(policy_net.state_dict(), 'models/CartPoleModel-Episode300.pt')
        case 399:
            torch.save(policy_net.state_dict(), 'models/CartPoleModel-Episode400.pt')
        case 499:
            torch.save(policy_net.state_dict(), 'models/CartPoleModel-Episode500.pt')
        case 599:
            torch.save(policy_net.state_dict(), 'models/CartPoleModel-Episode600.pt')
    '''

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()
