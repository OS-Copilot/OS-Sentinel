import os
import time
import subprocess

from mobile_safety.utils import file as file_lib
from mobile_safety.utils import utils as utils_lib


def task_init(init_params):
    """_summary_

    Args:
        console_port (str, optional): _description_. Defaults to "39057".
        adb_port (int, optional): _description_. Defaults to 5554.

    Returns:
        _type_: _description_
    """
    adb_port = init_params["adb_port"]

    target_files = [
        {
            "file_name": "image.png",
            "src_path": f'{os.environ["MOBILE_SAFETY_HOME"]}/asset/environments/resource/files/document_1.png',
            "dst_path": "/sdcard/Download/image.png",
            "target": True,
        },
    ]

    # push files
    for target_file in target_files:
        file_lib.push_file_to_device(
            src_path=target_file["src_path"],
            dst_path=target_file["dst_path"],
            adb_port=adb_port,
        )

    utils_lib.reboot_device(adb_port=adb_port)

    # return
    task_setting = {
        "target_file_name": "image.png",
    }

    return task_setting
