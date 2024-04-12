from abc import ABC, abstractmethod


class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self, arm, reward):
        pass
    
    @abstractmethod
    def experiment(self,  num_trials):
        pass

    @abstractmethod
    def report(self, algorithm):
        pass
