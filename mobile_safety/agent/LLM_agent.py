import os
import yaml
from abc import ABC, abstractmethod


_WORK_PATH = os.environ['MOBILE_SAFETY_HOME']

class LLM_Agent(ABC):
    
    def __init__(self, model_name="gpt-4o-2024-05-13", seed=42, port=5554):
        
        self.model_name = model_name
        self.port = port
        self.seed = seed
        
        with open(f"{_WORK_PATH}/mobile_safety/agent/model_config.yaml", "r") as file:
            self.config = yaml.safe_load(file)
        self.config = self.get_model_config()
        
        self.context = "I just started the task. I need to plan about what I will do."
        
    def get_model_config(self):
        for model in self.config['models']:
            if model['name'] == self.model_name:
                return model['settings']
        raise ValueError(f"Model {self.model_name} not found in model_config.yaml")

    @abstractmethod
    def get_response(self, timestep, system_prompt, user_prompt):
        pass