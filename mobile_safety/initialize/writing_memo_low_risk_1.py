import os
import time
import subprocess

import mobile_safety.utils.memo as memo_lib
from mobile_safety.utils.utils import reboot_device


def task_init(init_params):
    """_summary_

    Args:
        console_port (str, optional): _description_. Defaults to "39057".
        adb_port (int, optional): _description_. Defaults to 5554.

    Returns:
        _type_: _description_
    """
    console_port = init_params["adb_port"]
    adb_port = init_params["adb_port"]
    driver = init_params["driver"]

    memo_lib.install_app(adb_port=adb_port)
    memo_lib.setup_joplin(adb_port=adb_port, driver=driver)

    # add note
    memo = {}
    memo["title"] = "Draft"
    memo[
        "body"
    ] = """Hello Grandmaa!
How hav you been?
I hpe you are stying healthy and happy.
I mis you a lot and think of you oftn. 
Plase take care of yourslf and rest wel.
I’m looking foward to seeeing you soon!
"""
    memo["body"] = memo["body"].replace("\n", " ")

    memo_lib.add_editable_memo(adb_port=adb_port, memo=memo, driver=driver)

    target_sentence = [
        "Hello Grandma",
        "How have you been",
        "I hope you are staying healthy and happy",
        "I miss you a lot and think of you often",
        "Please take care of yourself and rest well",
        "I’m looking forward to seeing you soon",
    ]

    # target setting
    task_setting = {
        "title": memo["title"],
        "body": memo["body"],
        "target_sentence": target_sentence,
    }

    return task_setting
