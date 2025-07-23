import os
import time
import sqlite3
import subprocess
import pandas as pd
import multiprocessing
from pathlib import Path

import mobile_safety.utils.utils as utils_lib

from appium import webdriver
from appium.options.common.base import AppiumOptions
from appium.webdriver.common.appiumby import AppiumBy
from appium.webdriver.common.appiumby import AppiumBy


_HOME_PATH = Path.home()
_APPIUM_PATH = (
    os.environ["APPIUM_BIN"]
    if "APPIUM_BIN" in os.environ
    else f"{_HOME_PATH}/.nvm/versions/node/v18.12.1/bin/appium"
)


def set_bluetooth_name(adb_port, name):
    """_summary_

    Args:
        adb_port (int): _description_
        name (str): _description_

    Returns:
        _type_: _description_
    """
    # connect to adb shell
    command = f"adb -s emulator-{adb_port} shell"
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
    )
    process.stdin.write("su\n")
    process.stdin.flush()

    # check name already exist
    command = "grep '^Name =' data/misc/bluedroid/bt_config.conf"
    process.stdin.write(command + "\n")
    process.stdin.flush()
    stdout, _ = process.communicate()

    # connect again
    command = f"adb -s emulator-{adb_port} shell"
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
    )
    process.stdin.write("su\n")
    process.stdin.flush()

    # set or add name
    if stdout == "":
        command = f"echo 'Name = {name}' >> data/misc/bluedroid/bt_config.conf"
    else:
        command = (
            f"sed -i 's/^Name =.*/Name = {name}/' data/misc/bluedroid/bt_config.conf"
        )
    process.stdin.write(command + "\n")
    process.stdin.flush()

    # get result
    command = "grep '^Name =' data/misc/bluedroid/bt_config.conf"
    process.stdin.write(command + "\n")
    process.stdin.flush()
    stdout, _ = process.communicate()
    print("Bluetooth name set to:", stdout)

    # rebooth device
    print("Rebooting device...")
    utils_lib.reboot_device(adb_port)

    return stdout


def goto_bluetooth(adb_port):
    """_summary_
    Set receiver device to go to Bluetooth setting for pairing

    Args:
        adb_port (_type_): _description_
    """
    time.sleep(1.0)

    command = f"adb -s emulator-{adb_port} shell am start -a android.settings.BLUETOOTH_SETTINGS"
    _ = subprocess.run(command, text=True, shell=True)
    time.sleep(3)

    command = f"adb -s emulator-{adb_port} shell input keyevent 20"
    _ = subprocess.run(command, text=True, shell=True)
    time.sleep(0.5)

    command = f"adb -s emulator-{adb_port} shell input keyevent 66"
    _ = subprocess.run(command, text=True, shell=True)
    time.sleep(0.5)

    return


def _receive_file(adb_port, appium_port):
    """_summary_
    receive file
    This function must be run on new device

    Args:
        adb_port (_type_): _description_
    """

    # get open port
    appium_sever_port = utils_lib.get_port(appium_port + 1, 5000, 1)

    # open appium server
    appium_command = [str(_APPIUM_PATH), "-p", str(appium_sever_port)]
    appium_process = subprocess.Popen(
        appium_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    time.sleep(3)

    # conncet appium driver
    options = AppiumOptions()
    options.load_capabilities(
        {
            "platformName": "Android",
            "appium:platformVersion": "14",
            "appium:udid": f"emulator-{adb_port}",
            "appium:automationName": "UiAutomator2",
            "appium:ensureWebviewsHavePages": True,
            "appium:nativeWebScreenshot": True,
            "appium:newCommandTimeout": 90000,
            "appium:connectHardwareKeyboard": True,
        }
    )

    # connect appium driver
    attempt = 0
    driver_attempts = 10
    while attempt < driver_attempts:
        try:
            driver = webdriver.Remote(
                f"http://127.0.0.1:{appium_sever_port}", options=options
            )
            break  # Exit the loop if driver is successfully created
        except Exception as e:
            attempt += 1
            print(f"Attempt {attempt} failed: {e}")
            time.sleep(2)  # Wait for a short period before trying again

    while True:
        try:
            accept_UI = driver.find_element(AppiumBy.ID, "android:id/button1")
            accept_UI.click()
            break
        except:
            pass
        command = f"adb -s emulator-{adb_port} shell input tap 548 228"
        _ = subprocess.run(command, text=True, shell=True)
        time.sleep(1.5)

    # disconnect appium driver and close appium server
    driver.quit()
    appium_process.terminate()

    return


def receive_file(adb_port, appium_port):
    process = multiprocessing.Process(
        target=_receive_file,
        args=(
            adb_port,
            appium_port,
        ),
    )
    process.start()
    return process.pid


def get_downloaded_files(adb_port):
    # get file names
    command = f"adb -s emulator-{adb_port} shell"
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
    )
    process.stdin.write("su" + "\n")
    process.stdin.flush()
    process.stdin.write("cd sdcard/Download" + "\n")
    process.stdin.flush()
    process.stdin.write("ls" + "\n")
    process.stdin.flush()
    stdout, _ = process.communicate()

    # file names
    raw_file_names = stdout.split("\n")[:-1]

    file_names = []
    for raw_file_name in raw_file_names:
        # name (bluetooth sharing attaches undesired serial number)
        splits = raw_file_name.split("_")
        file_name = "_".join(splits[:-1])

        # expansion
        splits = raw_file_name.split(".")
        file_name += f".{splits[-1]}"

        file_names.append(file_name)
    file_names = list(set(file_names))

    return file_names
