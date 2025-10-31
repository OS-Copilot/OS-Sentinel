import os
import re
import cv2

from PIL import Image
from openai import OpenAI
from datetime import datetime

from mobile_safety.agent.LLM_agent import LLM_Agent
from mobile_safety.agent.utils import encode_image, parse_response


_WORK_PATH = os.environ["MOBILE_SAFETY_HOME"]


class GPTAgent(LLM_Agent):
    def __init__(self, model_name="gpt-4o-2024-05-13", seed=42, port=5554):

        super().__init__(model_name, seed, port)

        self.model = OpenAI()
        self.time_list = []

        return

    def get_response(self, timestep=None, system_prompt=None, user_prompt=None):

        img_pil_path = self.save_image(timestep)

        user_prompt = re.sub(r"<context>", self.context, user_prompt)

        start_time = datetime.now()

        completion = self.model.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encode_image(img_pil_path)}"
                            },
                        },
                    ],
                },
            ],
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"],
            top_p=self.config["top_p"],
            seed=self.seed,
        )
        response = completion.choices[0].message.content

        end_time = datetime.now()
        time_elapsed = end_time - start_time
        self.time_list.append(time_elapsed.total_seconds())
        print(f"Time elapsed: {time_elapsed.total_seconds()}")

        # parse response
        response_dict = parse_response(response)
        if response_dict["action"] == None:
            print("Error in response")
        if response_dict["context"] != "":
            self.context = response_dict["context"]

        # return used prompt for logging
        prompt = ""
        prompt += "<system_prompt>\n" + system_prompt + "</system_prompt>\n\n"
        prompt += "<user_prompt>\n" + user_prompt + "</user_prompt>\n\n"

        return response_dict, prompt

    def save_image(self, timestep=None):
        img_obs = timestep.curr_obs["pixel"]
        img_cv = cv2.resize(img_obs, dsize=(1024, 2048), interpolation=cv2.INTER_AREA)
        img_pil = Image.fromarray(img_cv)
        img_pil_path = f"{_WORK_PATH}/logs/tmp_{self.port}.png"
        img_pil.save(img_pil_path)

        return img_pil_path
