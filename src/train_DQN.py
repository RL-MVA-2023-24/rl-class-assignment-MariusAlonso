from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn
import random
import numpy as np

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

PATH = "model.pt"

CONFIG = {
    "gamma": 0.99,
    "batch_size": 64,
    "nb_actions": 4,
    "buffer_size": 10000,
    "epsilon_max": 1.0,
    "epsilon_min": 0.1,
    "epsilon_decay_period": 1000,
    "epsilon_delay_decay": 1000,
    "learning_rate": 0.01,
}


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity  # capacity of the buffer
        self.data = []
        self.index = 0  # index of the next cell to be filled
        # self.device = device

    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(
            map(lambda x: torch.Tensor(np.array(x)), list(zip(*batch)))
        )

    def __len__(self):
        return len(self.data)


def greedy_action(network, state):
    device = "cpu"  # "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()


class ProjectAgent:
    def __init__(self):
        device = "cpu"  # "cuda" if next(model.parameters()).is_cuda else "cpu"
        self.gamma = CONFIG["gamma"]
        self.batch_size = CONFIG["batch_size"]
        self.nb_actions = CONFIG["nb_actions"]
        self.memory = ReplayBuffer(CONFIG["buffer_size"])
        self.epsilon_max = CONFIG["epsilon_max"]
        self.epsilon_min = CONFIG["epsilon_min"]
        self.epsilon_stop = CONFIG["epsilon_decay_period"]
        self.epsilon_delay = CONFIG["epsilon_delay_decay"]
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_stop
        self.model = torch.nn.Sequential(
            nn.Linear(6, 24),
            nn.ReLU(),
            # nn.Linear(24, 24),
            # nn.ReLU(),
            nn.Linear(24, 4),
            nn.Softmax(dim=1),
        ).to(device)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=CONFIG["learning_rate"]
        )

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.model(Y).max(1)[0].detach()
            # update = torch.addcmul(R, self.gamma, 1-D, QYmax)
            update = torch.addcmul(R, 1 - D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0

        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)

            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)

            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            # train
            self.gradient_step()

            # next transition
            step += 1
            if done or trunc:
                print("done", done, "trunc", trunc)
                episode += 1
                print(
                    "Episode ",
                    "{:3d}".format(episode),
                    ", epsilon ",
                    "{:6.2f}".format(epsilon),
                    ", batch size ",
                    "{:5d}".format(len(self.memory)),
                    ", episode return ",
                    "{:4.1f}".format(episode_cum_reward / 1000000),
                    sep="",
                )
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state

        return episode_return

    def act(self, observation, use_random=False):
        if use_random:
            return env.action_space.sample()
        else:
            return greedy_action(self.model, observation)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self):
        self.model.load_state_dict(torch.load(PATH))
        self.model.eval()


if __name__ == "__main__":
    agent = ProjectAgent()
    episode_return = agent.train(env, 100)
    agent.save(PATH)
    print(episode_return)
    env.close()
