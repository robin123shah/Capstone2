import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# -----------------------------
# CONFIG
# -----------------------------
ACTIONS = ["L45","L22","FW","R22","R45"]

STATE_DIM = 18
ACTION_DIM = 5

GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
BUFFER_SIZE = 50000

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995

TARGET_UPDATE = 1000


# -----------------------------
# Q NETWORK
# -----------------------------
class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_DIM)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# PER BUFFER
# -----------------------------
class PERBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def add(self, transition, td_error=1.0):
        priority = (abs(td_error) + 1e-5) ** self.alpha

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        probs = np.array(self.priorities)
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, torch.FloatTensor(weights)

    def update_priorities(self, indices, td_errors):
        for i, td in zip(indices, td_errors):
            self.priorities[i] = (abs(td) + 1e-5) ** self.alpha


# -----------------------------
# AGENT
# -----------------------------
class DDQNAgent:
    def __init__(self):
        self.q_net = QNetwork()
        self.target_net = QNetwork()
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.buffer = PERBuffer(BUFFER_SIZE)

        self.epsilon = EPS_START
        self.steps = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)

        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_net(state)
        return ACTIONS[q_values.argmax().item()]

    def train_step(self):
        if len(self.buffer.buffer) < BATCH_SIZE:
            return

        samples, indices, weights = self.buffer.sample(BATCH_SIZE)

        states, actions, rewards, next_states, dones = zip(*samples)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        action_idx = torch.tensor([ACTIONS.index(a) for a in actions])

        # current Q
        q_vals = self.q_net(states)
        q_val = q_vals.gather(1, action_idx.unsqueeze(1)).squeeze()

        # DDQN target
        next_actions = self.q_net(next_states).argmax(dim=1)
        next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()

        target = rewards + GAMMA * next_q * (1 - dones)

        td_error = target - q_val

        loss = (weights * td_error.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.buffer.update_priorities(indices, td_error.detach().numpy())

        # epsilon decay
        self.epsilon = max(self.epsilon * EPS_DECAY, EPS_END)

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


# -----------------------------
# TRAIN LOOP (YOU CONNECT ENV)
# -----------------------------
def train(env, episodes=500):
    agent = DDQNAgent()

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for t in range(2000):
            action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.buffer.add((state, action, reward, next_state, done))

            agent.train_step()

            if agent.steps % TARGET_UPDATE == 0:
                agent.update_target()

            state = next_state
            total_reward += reward
            agent.steps += 1

            if done:
                break

        print(f"Episode {ep}, Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

    return agent


# -----------------------------
# SAVE MODEL (IMPORTANT)
# -----------------------------
def save_all(agent, path="model_full.pth"):
    torch.save({
        "model_state_dict": agent.q_net.state_dict(),
        "target_state_dict": agent.target_net.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "epsilon": agent.epsilon
    }, path)


def save_weights(agent, path="weights.pth"):
    torch.save(agent.q_net.state_dict(), path)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    from obelix import OBELIX   # YOU already have this

    env = OBELIX()

    agent = train(env, episodes=500)

    save_all(agent)      # full training state
    save_weights(agent)  # for submission