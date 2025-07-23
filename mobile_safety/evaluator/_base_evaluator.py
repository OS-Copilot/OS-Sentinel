from abc import ABC, abstractmethod
from collections import defaultdict


class Reward:

    def __init__(self):
        self.helpful = 0
        self.safety = 0


class BaseEvaluator(ABC):

    def __init__(self, task_setting=None, adb_port=5554):

        self.reward = Reward()
        self.adb_port = adb_port

        # history
        self.observations = []
        self.actions = []

        # evaluation-relevant variables
        self.progress = {"step": -1, "finished": False, "risk-detected step": -1}
        self.data = {"adb_port": adb_port, "max_step": 15, "json_error": 0}
        self.task_setting = task_setting

        return

    def evaluate(self, observation="None", action="None", driver=None):
        # update progress
        self.observations.append(observation)
        self.actions.append(action)

        self.update_progress(driver)

        # compute helpfulness and safety scores
        score = self.get_score(driver)

        return self.progress, score

    @abstractmethod
    def get_score(self, driver):
        return 0

    @abstractmethod
    def update_progress(self, driver):
        self.progress["step"] += 1
        return
