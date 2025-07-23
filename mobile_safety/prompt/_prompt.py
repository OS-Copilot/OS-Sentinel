import regex as re
import importlib
from typing import Any, NamedTuple


class Prompt(NamedTuple):
    system_prompt: str
    user_prompt: str

    def __getitiem__(self, attr):
        return getattr(self, attr)


class PromptBuilder:
    def __init__(self, env):

        self.prompt_mode = env.prompt_mode
        self.task_tag = env.task_tag
        self.instruction = env.instruction

        self.width = env.width
        self.height = env.height

        self.base_prompt = self.load_base_prompt()

    def build(self, parsed_obs, action_history, action_error):
        # error warning
        if action_error == 1:
            instruction_prompt = (
                self.base_prompt.INSTRUCTION_PROMPT[:-1]
                + "You have made a mistake on the format once. Please make sure to respond in the provided format.\n\n"
            )
        elif action_error == 2:
            instruction_prompt = (
                self.base_prompt.INSTRUCTION_PROMPT[:-1]
                + "You have made a mistake on the format twice. Please make sure to respond in the provided format. If you make another mistake in the format, it may affect the score negatively.\n\n"
            )
        else:
            instruction_prompt = self.base_prompt.INSTRUCTION_PROMPT

        # build prompt
        system_prompt = self.base_prompt.SYSTEM_PROMPT
        user_prompt = ""
        user_prompt += instruction_prompt
        user_prompt += self.base_prompt.INFORMATION_PROMPT
        user_prompt = re.sub(r"<action_history>", str(action_history), user_prompt)
        user_prompt = re.sub(r"<current_observation>", str(parsed_obs), user_prompt)

        return Prompt(system_prompt, user_prompt)

    def load_base_prompt(self):
        # import base prompt
        if self.prompt_mode == "" or self.prompt_mode == "basic":
            prompt_base_module = f"mobile_safety.prompt._base"  # basic
        else:
            prompt_base_module = f"mobile_safety.prompt._base_{self.prompt_mode}"
        base_prompt = importlib.import_module(prompt_base_module)
        importlib.reload(base_prompt)

        # instruction setting
        base_prompt.INFORMATION_PROMPT = re.sub(
            r"<goal_instruction>", self.instruction, base_prompt.INFORMATION_PROMPT
        )

        # import task specific prompt
        prompt_task_module = f"mobile_safety.prompt.{self.task_tag}"
        prompt_task = importlib.import_module(prompt_task_module)

        # additonal actions
        additional_actions = getattr(prompt_task, "ADDITIONAL_ACTIONS", "")
        base_prompt.INSTRUCTION_PROMPT = base_prompt.INSTRUCTION_PROMPT.replace(
            "<additional_actions>", additional_actions
        )

        return base_prompt
