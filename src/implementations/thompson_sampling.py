import csv
import random
import logging
from tqdm import tqdm
from src.generalization.bandit import Bandit


class ThompsonSampling(Bandit):
    def __init__(self, p, precision):
        super().__init__(p)
        self.p = p
        self.precision = precision  # Setting precision
        self.alpha = [1.0] * len(p)  # Initializing alpha values
        self.beta = [1.0] * len(p)  # Initializing beta values
        self.rewards = []  # Creating list to store rewards
        self.regrets = [] # Same, but with regrets

    def __repr__(self):
        return 'ThompsonSampling'

    def pull(self):
        samples = [random.betavariate(self.alpha[i], self.beta[i]) for i in range(len(self.p))]
        return samples.index(max(samples))

    def update(self, arm, reward):
        self.alpha[arm] += reward
        self.beta[arm] += (1 - reward)
        if self.alpha[arm] <= 0 or self.beta[arm] <= 0:
            self.alpha[arm] = 1.0  # Resetting alpha if it becomes non-positive
            self.beta[arm] = 1.0  # Resetting beta if it becomes non-positive
        logging.debug(f'ThompsonSampling - Arm {arm} selected, Reward: {reward}')

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
