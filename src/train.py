from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import pickle
import numpy as np
from tqdm import tqdm
import random
import os

# evaluate the agent
from evaluate import evaluate_HIV, evaluate_HIV_population

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

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


class ProjectAgent:
    def __init__(self):
        self.gamma = 0.98
        self.S = []
        self.A = []
        self.R = []
        self.S2 = []
        self.D = []
        self.Q = None

    def collect_samples(self, env, horizon, greedy_Q=None, greedy_prob=0.85):

        s, _ = env.reset()
        for _ in tqdm(range(horizon)):
            if greedy_Q is not None:
                if np.random.rand() < greedy_prob:
                    a = greedy_action(greedy_Q, s, env.action_space.n)
                else:
                    a = np.random.randint(env.action_space.n)
            else:
                a = np.random.randint(env.action_space.n)
            s2, r, done, trunc, _ = env.step(a)
            # dataset.append((s,a,r,s2,done,trunc))
            self.S.append(s)
            self.A.append(a)
            self.R.append(r)
            self.S2.append(s2)
            self.D.append(done)
            if done or trunc:
                s, _ = env.reset()
            else:
                s = s2

    def rf_fqi(self, iterations, last_always_taken, augmented_random=None):

        idx_samples = np.arange(len(self.S))
        if augmented_random is not None:
            idx_samples = np.concatenate([idx_samples[-last_always_taken:], np.random.choice(idx_samples[:-last_always_taken], augmented_random, replace=False)])
        else:
            idx_samples = idx_samples[-last_always_taken:]

        nb_samples = len(idx_samples)

        S = np.array(self.S)[idx_samples, :]
        A = np.array(self.A).reshape((-1, 1))[idx_samples, :]
        R = np.array(self.R)[idx_samples]
        S2 = np.array(self.S2)[idx_samples, :]
        D = np.array(self.D)[idx_samples]
        SA = np.append(S, A, axis=1)

        Q = self.Q
        for iter in tqdm(range(iterations)):
            if iter == 0 and Q is None:
                value = R.copy() # Q.predict
            else:
                Q2 = np.zeros((nb_samples, 4))
                for a2 in range(4):
                    A2 = a2 * np.ones((S.shape[0], 1))
                    S2A2 = np.append(S2, A2, axis=1)
                    Q2[:, a2] = Q.predict(S2A2)
                max_Q2 = np.max(Q2, axis=1)
                value = R + self.gamma * (1 - D) * max_Q2

            Q = ExtraTreesRegressor(n_estimators=50, n_jobs=-1)
            Q.fit(SA, value)

        self.Q = Q

    def train(self, env, nb_samples, nb_iterations, nb_collects, nb_samples_first_collect=None, nb_iterations_first_collect=None, augmented_random=None):

        if nb_samples_first_collect is not None and nb_iterations_first_collect is not None:
            self.collect_samples(env, nb_samples_first_collect)
            self.rf_fqi(nb_iterations_first_collect, last_always_taken=nb_samples_first_collect, augmented_random=None)
        else:
            self.collect_samples(env, nb_samples)
            self.rf_fqi(nb_iterations, last_always_taken=nb_samples, augmented_random=None)

        print(0, evaluate_HIV(agent=self, nb_episode=5) / 1000000)

        # Make an auto-save
        self.save(PATH[:-4] + "_autosave_0.pkl")

        for _ in range(nb_collects-1):
            self.collect_samples(env, nb_samples, self.Q, greedy_prob=0.85)
            self.rf_fqi(nb_iterations, last_always_taken=nb_samples, augmented_random=augmented_random)

            seed_everything(seed=42)
            print(_+1, evaluate_HIV(agent=self, nb_episode=5) / 1000000)
            seed_everything(seed=999)

            # Make an auto-save
            self.save(PATH[:-4] + f"_autosave_{_+1}.pkl")

    def act(self, observation, use_random=False):
        if use_random:
            return np.random.randint(env.action_space.n)
        else:
            return greedy_action(self.Q, observation, env.action_space.n)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.Q, f)

    def load(self):
        with open(PATH, "rb") as f:
            self.Q = pickle.load(f)


if __name__ == "__main__":

    # # seed for reproducibility
    seed_everything(seed=999)

    # train the agent
    agent = ProjectAgent()
    episode_return = agent.train(
        env,
        nb_samples=2000,
        nb_iterations=300,
        nb_collects=50,
        nb_samples_first_collect=16000,
        nb_iterations_first_collect=800,
        augmented_random=4000,
    )
    agent.save(PATH)
    print(episode_return)
    env.close()

    # seed for reproducibility
    seed_everything(seed=42)

    print("Evaluation of the agent without domain randomization")
    print(evaluate_HIV(agent=agent, nb_episode=5) / 1000000)
    # print("Evaluation of the agent with domain randomization")
    # print(evaluate_HIV_population(agent=agent, nb_episode=5) / 1000000)
