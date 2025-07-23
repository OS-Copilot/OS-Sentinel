import os
import re
import cv2

from PIL import Image
from datetime import datetime

import google.generativeai as genai

from mobile_safety.agent.LLM_agent import LLM_Agent
from mobile_safety.agent.utils import parse_response


class GeminiAgent(LLM_Agent):
    def __init__(
        self, model_name="gemini-1.5-pro-001", seed=42, port=5554, safety_settings=None
    ):

        super().__init__(model_name, seed, port)

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])

        generation_config = {
            "temperature": self.config["temperature"],
            "top_p": self.config["top_p"],
            # "top_k": 42,
            "max_output_tokens": self.config["max_tokens"],
        }

        if safety_settings is None:
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": self.config["HARM_CATEGORY_HARASSMENT"],
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": self.config["HARM_CATEGORY_HATE_SPEECH"],
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": self.config["HARM_CATEGORY_SEXUALLY_EXPLICIT"],
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": self.config["HARM_CATEGORY_DANGEROUS_CONTENT"],
                },
            ]

        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

        return

    def get_response(self, timestep=None, system_prompt=None, user_prompt=None):

        # save image as png
        img_pil = self.save_image(timestep)

        # add context to user prompt
        user_prompt = re.sub(r"<context>", self.context, user_prompt)

        # build prompt
        prompt = ""
        prompt += system_prompt + "\n"
        prompt += user_prompt

        start_time = datetime.now()

        # get response from the agent
        response = self.model.generate_content([prompt, img_pil])

        end_time = datetime.now()
        time_elapsed = end_time - start_time
        print(f"Time elapsed: {time_elapsed.total_seconds()}")

        # parse response
        try:
            response = response.text

            # parse response
            response_dict = parse_response(response)
            if response_dict["action"] == None:
                print("Error in response")
            if response_dict["context"] != "":
                self.context = response_dict["context"]
        except:  # mostly blocked
            response_dict = {"action": None, "context": ""}
            response_dict["raw_response"] = response

        # return used prompt for logging
        prompt = ""
        prompt += "<system_prompt>\n" + system_prompt + "</system_prompt>\n\n"
        prompt += "<user_prompt>\n" + user_prompt + "</user_prompt>\n\n"

        return response_dict, prompt

    def save_image(self, timestep=None):
        img_obs = timestep.curr_obs["pixel"]
        img_cv = cv2.resize(img_obs, dsize=(1024, 2048), interpolation=cv2.INTER_AREA)
        img_pil = Image.fromarray(img_cv)

        return img_pil
