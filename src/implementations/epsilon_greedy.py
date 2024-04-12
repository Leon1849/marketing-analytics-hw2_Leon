import csv
import random
from tqdm import tqdm
from src.generalization.bandit import Bandit


class EpsilonGreedy(Bandit):
    def __init__(self, p, initial_epsilon):
        super().__init__(p)
        self.p = p
        self.rewards = []  # Creating list to store rewards
        self.regrets = [] # Same, but with regrets
        self.epsilon = initial_epsilon  # Initializing epsilon
        self.q_values = [0] * len(p)  # Initializing q-values
        self.action_counts = [0] * len(p)  # Initializing action counts

    def __repr__(self):
        return 'EpsilonGreedy'

    def pull(self):
        if random.random() < self.epsilon:
            return random.randint(0, len(self.p) - 1)
        else:
            return self.q_values.index(max(self.q_values))

    def update(self, arm, reward):
        self.action_counts[arm] += 1  # Incrementing action count
        self.q_values[arm] += (reward - self.q_values[arm]) / self.action_counts[arm]  # Updating q-value
        self.epsilon = 1 / (sum(self.action_counts) + 1)  # Decay epsilon

    def experiment(self,  num_trials):
        # Running an experiment with num_trials iterations
        for _ in tqdm(range(num_trials)):
            arm = self.pull()
            # print(self.rewards)
            reward = self.p[arm]
            self.rewards.append(reward)  # Recording the reward
            regret = max(self.p) - reward  # Calculating regret
            self.regrets.append(regret)  # Recording regret
            self.update(arm, reward)  # Updating bandit's state

    def report(self, algorithm):
        # Reporting results of the experiment
        with open(f'{algorithm}_results.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Bandit', 'Reward', 'Algorithm'])
            for i in range(len(self.rewards)):
                writer.writerow([i, self.rewards[i], algorithm])
        avg_reward = sum(self.rewards) / len(self.rewards)
        avg_regret = sum(self.regrets) / len(self.regrets)
        print(f'Average Reward for {algorithm}: {avg_reward}')
        print(f'Average Regret for {algorithm}: {avg_regret}')