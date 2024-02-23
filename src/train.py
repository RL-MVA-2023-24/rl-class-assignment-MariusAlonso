from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import pickle
import numpy as np
from tqdm import tqdm

PATH = "model.pkl"

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
def greedy_action(Q, s, nb_actions):
    Qsa = []
    for a in range(nb_actions):
        sa = np.append(s, a).reshape(1, -1)
        Qsa.append(Q.predict(sa))
    return np.argmax(Qsa)


class ProjectAgent:
    def __init__(self):
        self.gamma = 0.98
        self.S = None
        self.A = None
        self.R = None
        self.S2 = None
        self.D = None
        self.Qfunctions = None

    def collect_samples(self, env, horizon):
        s, _ = env.reset()
        # dataset = []
        S = []
        A = []
        R = []
        S2 = []
        D = []
        for _ in tqdm(range(horizon)):
            a = env.action_space.sample()
            s2, r, done, trunc, _ = env.step(a)
            # dataset.append((s,a,r,s2,done,trunc))
            S.append(s)
            A.append(a)
            R.append(r)
            S2.append(s2)
            D.append(done)
            if done or trunc:
                s, _ = env.reset()
            else:
                s = s2
        self.S = np.array(S)
        self.A = np.array(A).reshape((-1, 1))
        self.R = np.array(R)
        self.S2 = np.array(S2)
        self.D = np.array(D)

    def rf_fqi(self, iterations):
        nb_samples = self.S.shape[0]
        Qfunctions = []
        SA = np.append(self.S, self.A, axis=1)
        for iter in tqdm(range(iterations)):
            if iter == 0:
                value = self.R.copy()
            else:
                Q2 = np.zeros((nb_samples, 4))
                for a2 in range(4):
                    A2 = a2 * np.ones((self.S.shape[0], 1))
                    S2A2 = np.append(self.S2, A2, axis=1)
                    Q2[:, a2] = Qfunctions[-1].predict(S2A2)
                max_Q2 = np.max(Q2, axis=1)
                value = self.R + self.gamma * (1 - self.D) * max_Q2
            Q = ExtraTreesRegressor(n_estimators=40, max_depth=10)
            Q.fit(SA, value)
            Qfunctions.append(Q)

        # Display evolution of bellman residual
        bellman_residuals = []
        for iter in range(iterations):
            Q = Qfunctions[iter]
            Q2 = np.zeros((nb_samples, 4))
            for a2 in range(4):
                A2 = a2 * np.ones((self.S.shape[0], 1))
                S2A2 = np.append(self.S2, A2, axis=1)
                Q2[:, a2] = Q.predict(S2A2)
            max_Q2 = np.max(Q2, axis=1)
            bellman_residuals.append(np.mean((self.R + self.gamma * (1 - self.D) * max_Q2 - Q.predict(SA)) ** 2))

        import matplotlib.pyplot as plt

        plt.plot(bellman_residuals)
        plt.title("Bellman Residuals")
        plt.show()

        # Display reward of greedy agent reward
        rs = []
        for _ in range(10):
            print("Episode ", _)
            s, _ = env.reset()
            cum_reward = 0
            t = 0
            done = False
            trunc = False
            while not done and not trunc:
                a = greedy_action(Qfunctions[-1], s, env.action_space.n)
                s, r, done, trunc, _ = env.step(a)
                cum_reward += self.gamma ** t * r
                t += 1
            rs.append(cum_reward)

        print("Mean reward of greedy agent: ", np.mean(rs) / 1000000)
        print("Standard deviation of greedy agent: ", np.std(rs) / 1000000)     

        self.Q = Qfunctions[-1]

    def train(self, env, nb_samples, nb_iterations):

        self.collect_samples(env, nb_samples)

        self.rf_fqi(nb_iterations)

    def act(self, observation, use_random=False):
        if use_random:
            return env.action_space.sample()
        else:
            return greedy_action(self.Q, observation, env.action_space.n)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.Q, f)

    def load(self):
        with open(PATH, "rb") as f:
            self.Q = pickle.load(f)


if __name__ == "__main__":
    agent = ProjectAgent()
    episode_return = agent.train(env, 2000, 200)
    agent.save(PATH)
    print(episode_return)
    env.close()
