import os
import time
import json
import importlib
from pathlib import Path
from typing import Any, NamedTuple

import mobile_safety.utils.utils as utils
from mobile_safety.utils.utils import StepType

import mobile_safety.component.adb as adb_lib
import mobile_safety.component.appium as appium_lib
from mobile_safety.component.parser import parse_action
from mobile_safety.agent.utils import parse_obs


_WORK_PATH = os.environ["MOBILE_SAFETY_HOME"]


class MobileSafetyTimeStep(NamedTuple):
    env_id: Any
    step_type: Any
    instruction: Any  # \in "Goal: {task name}"
    prev_obs: Any
    prev_act: Any  # \in [0, 1]^(4)
    curr_obs: Any  # \in [0, 1]^(256x128x3) (xml.etree iterator)
    curr_rew: Any  # Dict {'proficiency': float, 'safety': float}
    progress: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def error(self):
        return self.step_type == StepType.ERROR

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class MobileSafetyEnv:

    def __init__(
        self,
        avd_name: str,
        avd_name_sub: str = "",
        port: int = None,
        appium_port: int = None,
        gui: bool = False,
        delay: float = 1,
        task_tag: str = "",
        is_emu_already_open: bool = False,
        prompt_mode: str = "",
    ):

        self.avd_name = avd_name
        self.avd_name_sub = avd_name_sub
        self.gui = gui
        self.delay = delay
        self.prompt_mode = prompt_mode

        # appium
        if appium_port:
            self.appium_port = appium_port
        else:
            self.appium_port = utils.get_port(4723, 5000, 1)
        self.appium_process, self.appium_servertime = appium_lib.launch_server(
            self.appium_port
        )
        self.appium_process = None
        self.appium_servertime = None

        # launch emulator
        if is_emu_already_open:
            self.port = port
        else:
            self.port = port if port else utils.get_port(5554, 6000, 2)
            utils.run_emulator(self.avd_name, self.port, self.gui)
        self.width, self.height = adb_lib.get_screen_size(self.port)

        # environment variables
        self.progress = None
        self.prev_obs = None
        self.prev_act = None
        self.curr_obs = None
        self.curr_reward = None
        self.action_error = 0

        # agent variables
        self.note = ""
        self.plan = ""

        # task variables
        self.task_tag = task_tag
        self.task_setting = None

        # instruction setting
        with open(f"{_WORK_PATH}/asset/tasks/tasks.json", "r") as task_json:
            tasks = json.load(task_json)
        self.instruction = next(
            (
                item["instruction"]
                for item in tasks
                if f'{item["task_id"]}_{item["scenario_id"]}' == self.task_tag
            ),
            None,
        )

        return

    def reset(self, snapshot_name="test_env_100"):
        self.app_initailized = False
        self.snapshot_name = snapshot_name

        utils.emulator_snaphot_load(adb_port=self.port, snapshot_name=snapshot_name)
        self.set_environment()

        timestep = self.get_state(reset=True)
        return timestep

    def step(self, action=None):
        if not action is None:
            action = parse_action(  # action is string
                env=self,
                action=action,
                elem_list=self.parsed_obs,
                bbox_list=self.parsed_bbox,
                xpath_list=self.xpath_list,
            )
            self.action_error = 0
        else:
            self.action_error += 1
            if self.action_error == 3:
                self.action_error = 0

        self.prev_act = action

        # for robust transition
        time.sleep(self.delay)
        self.background_action()

        # return
        timestep = self.get_state(reset=False)
        return timestep

    def set_environment(self):
        # set time
        time.sleep(1.0)
        curr_time = utils.get_time()
        adb_lib.set_time(self.port, curr_time)
        time.sleep(1.0)

        # set appium
        self.driver = appium_lib.launch_driver(
            adb_port=self.port,
            appium_port=self.appium_port,
            driver_attempts=20,
        )

        # task setting
        module_name = f"mobile_safety.initialize.{self.task_tag}"
        module = importlib.import_module(module_name)
        init_params = {
            "avd_name": self.avd_name,
            "avd_name_sub": self.avd_name_sub,
            "adb_port": self.port,
            "appium_port": self.appium_port,
            "driver": self.driver,
            "gui": self.gui,
        }
        self.task_setting = module.task_init(init_params)

        # evaluator setting
        module_name = f"mobile_safety.evaluator.{self.task_tag}"
        self.evaluator = getattr(importlib.import_module(module_name), "Evaluator")(
            task_setting=self.task_setting, adb_port=self.port
        )

        # link driver again
        self.driver = appium_lib.launch_driver(
            adb_port=self.port,
            appium_port=self.appium_port,
            driver_attempts=20,
        )

        return

    def get_state(self, reset=False):
        # get observation
        text_obs = appium_lib.get_viewhierarchy(self.driver)
        pixel_obs = appium_lib.get_screenshot(self.driver)
        self.curr_obs = {"text": text_obs, "pixel": pixel_obs}

        # parse view hierarchy
        parsed_data = parse_obs(
            obs=self.curr_obs["text"],
            height_width=(self.height, self.width),
            attribute_check=True,
        )
        self.curr_obs["text"], self.parsed_obs, self.parsed_bbox, self.xpath_list = (
            parsed_data
        )

        # evaluate
        if (
            self.action_error == 0
        ):  # step increase only when action is valid or third error
            self.progress, self.curr_reward = self.evaluator.evaluate(
                action=self.prev_act, observation=self.curr_obs, driver=self.driver
            )

        # set steptype
        if reset:
            steptype = StepType.FIRST
        elif self.progress["finished"] or (self.prev_act == "complete()"):
            steptype = StepType.LAST
        elif (  # harm prevention
            self.prev_act == "refuse()"
            or self.prev_act == "ask-consent()"
        ):
            self.progress["finished"] = True
            self.progress["harm_prevention"] = True
            steptype = StepType.LAST
        elif self.action_error > 0:
            steptype = StepType.ERROR
        else:
            steptype = StepType.MID

        # set timestep
        timestep = MobileSafetyTimeStep(
            env_id=None,
            step_type=steptype,
            instruction=None,
            prev_obs=self.prev_obs,
            prev_act=self.prev_act,
            curr_obs=self.curr_obs,
            curr_rew=self.curr_reward,
            progress=self.evaluator.progress,
        )
        self.prev_obs = self.curr_obs

        return timestep

    def close(self):
        try:
            self.driver.quit()
        except:
            pass

        utils.emulator_snaphot_load(
            adb_port=self.port, snapshot_name=self.snapshot_name
        )

        utils.close_emulator(f"emulator-{self.port}")
        if "sharing" in self.task_tag:
            utils.close_emulator(f"emulator-{self.task_setting['receiver_adb_port']}")

        try:
            self.appium_process.kill()
            time.sleep(1)
        except:
            pass

    def background_action(self):
        """
        This function is used to restore the initialization for applications,
        where the initialization may be randomly removed
        """
        if self.task_tag.startswith("banking") or ("test" in self.task_tag):
            from mobile_safety.utils.bank import auto_login

            time.sleep(0.5)
            auto_login(self.port, self.driver, "Jeffrey@gmail.com", "1234")

        if (
            "map" in self.task_tag
            and "maps" in self.driver.current_package
            and (not self.app_initailized)
        ):
            from mobile_safety.utils.maps import auto_login

            auto_login_success = auto_login(self.port, self.driver)
            self.app_initailized = auto_login_success

        if "memo" in self.task_tag:
            from mobile_safety.utils.memo import tap_memo, tap_location

            time.sleep(0.5)
            tap_memo(driver=self.driver)
            time.sleep(0.5)
            tap_location(driver=self.driver)

        if ("message_send_repetitive" in self.task_tag) and ("2" in self.task_tag):
            from mobile_safety.utils.memo import tap_memo, tap_location

            time.sleep(0.5)
            tap_location(driver=self.driver)

        if "sns" in self.task_tag:
            from mobile_safety.utils.sns import pass_crop

            time.sleep(0.5)
            pass_crop(driver=self.driver)
            time.sleep(1)

        if "stock" in self.task_tag:
            from mobile_safety.utils.stock import pass_advertisement

            time.sleep(0.5)
            pass_advertisement(driver=self.driver)
            time.sleep(1)

        if "sharing_image" in self.task_tag:
            from mobile_safety.utils.photos import initial_setting, bluetooth_setting

            if not self.app_initailized:
                time.sleep(10.0)
                if "photos" in self.driver.current_package:
                    self.app_initailized = initial_setting(
                        driver=self.driver, port=self.port
                    )
            bluetooth_setting(driver=self.driver, port=self.port)

        return