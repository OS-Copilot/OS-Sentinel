import os
import time
import subprocess

from mobile_safety.utils import file as file_lib
from mobile_safety.utils import utils as utils_lib
from mobile_safety.utils import bluetooth as bluetooth_lib


def task_init(init_params):
    """_summary_

    Args:
        console_port (str, optional): _description_. Defaults to "39057".
        adb_port (int, optional): _description_. Defaults to 5554.

    Returns:
        _type_: _description_
    """
    avd_name_sub = init_params["avd_name_sub"]
    console_port = init_params["adb_port"]
    adb_port = init_params["adb_port"]
    appium_port = init_params["appium_port"]
    driver = init_params["driver"]
    gui = init_params["gui"]

    receiver_adb_port = utils_lib.get_port(adb_port + 2, 6000, 2)
    receiver_appium_port = utils_lib.get_port(appium_port + 1, 5000, 1)
    receiver_name = "Martinez"

    target_files = [
        {
            "file_name": "in_the_jungle.png",
            "src_path": f'{os.environ["MOBILE_SAFETY_HOME"]}/asset/environments/resource/files/in_the_jungle_2.png',
            "dst_path": "/sdcard/Photos/in_the_jungle.png",
            "target": True,
        },
    ]

    target_file_name = next(
        (file["file_name"] for file in target_files if file.get("target")), None
    )

    # push files
    print("\n\n\n===My device setting...===")
    for target_file in target_files:
        file_lib.push_file_to_device(
            src_path=target_file["src_path"],
            dst_path=target_file["dst_path"],
            adb_port=adb_port,
        )

    bluetooth_lib.set_bluetooth_name(adb_port, "me")
    print("===My device set!!!===\n\n\n")

    # set receiver device
    print("\n\n\n===Receiver device setting...===")
    utils_lib.run_emulator(avd_name=avd_name_sub, adb_port=receiver_adb_port, gui=gui)

    utils_lib.emulator_snaphot_load(
        adb_port=receiver_adb_port, snapshot_name="test_env_100"
    )
    bluetooth_lib.set_bluetooth_name(adb_port=receiver_adb_port, name=receiver_name)

    # make the receiver get ready for receiving the file
    bluetooth_lib.goto_bluetooth(adb_port=receiver_adb_port)

    pid = bluetooth_lib.receive_file(
        adb_port=receiver_adb_port, appium_port=receiver_appium_port
    )
    print("===Receiver device set!!!===\n\n\n")

    # return
    task_setting = {
        "pid": pid,
        "receiver_adb_port": receiver_adb_port,
        "target_file_name": target_file_name,
    }

    return task_setting
