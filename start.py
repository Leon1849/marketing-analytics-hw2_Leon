from src.logger.logs import *
from src.implementations.epsilon_greedy import EpsilonGreedy
from src.implementations.thompson_sampling import ThompsonSampling
from src.visualization.visualization import Visualization


def main():
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("MAB Application")
    logging.basicConfig(level=logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    logger.handlers[0].setFormatter(CustomFormatter())
    
    epsilon_value = 0.1
    precision_value = 0.001
    number_of_trials = 20000
    Bandit_Reward = [1, 2, 3, 4]

    epsilon_greedy_bandit = EpsilonGreedy(Bandit_Reward, epsilon_value)
    thompson_sampling_bandit = ThompsonSampling(Bandit_Reward, precision_value)

    epsilon_greedy_bandit.experiment(number_of_trials)
    thompson_sampling_bandit.experiment(number_of_trials)

    epsilon_greedy_bandit.report('EpsilonGreedy')
    thompson_sampling_bandit.report('ThompsonSampling')
    
    eg_rewards = [2, 3, 1, 4, 2]
    ts_rewards = [3, 2, 4, 1, 3]
    
    eg_cumulative_rewards = [2, 1, 3, 2, 4]
    ts_cumulative_rewards = [4, 3, 2, 1, 2]

    visualization = Visualization()
    visualization.plot1(eg_rewards, ts_rewards)
    visualization.plot2(eg_cumulative_rewards, ts_cumulative_rewards)


if __name__ == "__main__":
    main()
