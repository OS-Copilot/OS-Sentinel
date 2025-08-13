import os
import datetime

from PIL import Image
import matplotlib.pyplot as plt


_WORK_PATH = os.environ["MOBILE_SAFETY_HOME"]


class Logger:
    def __init__(self, args):

        self.task_id = args.task_id
        self.scenario_id = args.scenario_id

        self.model = args.model
        self.prompt_mode = args.prompt_mode

        # folder setting
        if "gpt" in self.model:
            folder_name = "gpt"
        elif "claude" in self.model:
            folder_name = "claude"
        elif "gemini" in self.model:
            folder_name = "gemini"
        elif "o1" in self.model:
            folder_name = "o1"

        if self.prompt_mode == "" or self.prompt_mode == "basic":
            folder_name += "_basic"
        elif self.prompt_mode == "scot":
            folder_name += "_scot"
        elif self.prompt_mode == "safety_guided":
            folder_name += "_safety_guided"

        self.dir_name = (
            f"{_WORK_PATH}/logs/{folder_name}/{self.task_id}/{self.scenario_id}"
        )
        if not os.path.exists(self.dir_name):
            os.makedirs(self.dir_name)

        # file setting
        time_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.file_name = f"{self.dir_name}/{time_now}"

        self.images = []
        self.texts = ""

        return

    def log(self, prompt=None, response_dict=None, timestep=None):
        # pass logging if error in timestep
        if timestep is not None and timestep.error():
            print("=" * 30)
            print("Error timestep: JSON parsing failed")
            print("Raw response:")
            print(response_dict["raw_response"])
            print("=" * 30)
            return

        # log prompt
        if prompt is not None:
            print(prompt)
            self.texts += prompt + "\n"

        # log response
        if response_dict is not None:
            print("<assistant_prompt>")
            print(response_dict["raw_response"])
            print("</assistant_prompt>\n\n")
            for key, value in response_dict.items():
                if key != "raw_response":
                    self.texts += f"{key}: {value}\n"
            self.texts += "\n"

        # log state: progress, image
        if timestep is not None:
            img_obs = timestep.curr_obs["pixel"]
            self.images.append(Image.fromarray(img_obs))

            # print image
            plt.imshow(img_obs)
            # plt.show()

            # progress log
            print("progress:", timestep.progress)
            print("=" * 60 + "\n")
            progress = timestep.progress if timestep is not None else None
            self.texts += "=" * 60 + "\n"
            self.texts += f"progress: {progress}\n"
            self.texts += "=" * 60 + "\n\n"

        # update files
        with open(f"{ self.file_name}.txt", "w") as file:
            file.write(self.texts)

        if len(self.images) > 0:
            self.images[0].save(
                f"{self.file_name}.gif",
                save_all=True,
                append_images=self.images[1:],
                optimize=False,
                duration=2000,
                loop=0,
            )

        return
