import matplotlib.pyplot as plt


class Visualization():
    @staticmethod
    def plot1(epsilon_greedy_rewards, thompson_rewards):
        plt.plot(epsilon_greedy_rewards, label='Epsilon Greedy')
        plt.plot(thompson_rewards, label='Thompson Sampling')
        plt.xlabel('Trials')
        plt.ylabel('Reward')
        plt.title('Learning Process')
        plt.legend()
        plt.show()

    @staticmethod
    def plot2(e_greedy_rewards, thompson_rewards):
        cumulative_e_greedy_rewards = [sum(e_greedy_rewards[:i + 1]) for i in range(len(e_greedy_rewards))]
        cumulative_thompson_rewards = [sum(thompson_rewards[:i + 1]) for i in range(len(thompson_rewards))]
        plt.plot(cumulative_e_greedy_rewards, label='Epsilon Greedy')
        plt.plot(cumulative_thompson_rewards, label='Thompson Sampling')
        plt.xlabel('Trials')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Rewards')
        plt.legend()
        plt.show()
