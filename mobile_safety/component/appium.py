import os
import io
import time
import base64
import datetime
import subprocess
import numpy as np
import xml.etree.ElementTree as xml_element_tree

from PIL import Image
from pathlib import Path
from appium import webdriver
from appium.options.common.base import AppiumOptions


_HOME_PATH = Path.home()
_APPIUM_PATH = (
    os.environ["APPIUM_BIN"]
    if "APPIUM_BIN" in os.environ
    else f"{_HOME_PATH}/.nvm/versions/node/v18.12.1/bin/appium"
)


def launch_server(port=4723):
    appium_command = [str(_APPIUM_PATH), "-p", str(port)]
    appium_process = subprocess.Popen(
        appium_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    appium_servertime = datetime.datetime.now()
    time.sleep(5.0)

    return appium_process, appium_servertime


def launch_driver(adb_port=5554, appium_port=4723, driver_attempts=10):

    # Reset driver
    driver = None
    options = AppiumOptions()
    options.load_capabilities(
        {
            "platformName": "Android",
            "appium:platformVersion": "14",  # 10 for 29, 14 for 34
            "appium:udid": f"emulator-{adb_port}",
            "appium:automationName": "UiAutomator2",
            "appium:ensureWebviewsHavePages": True,
            "appium:nativeWebScreenshot": True,
            "appium:newCommandTimeout": 90000,
            "appium:connectHardwareKeyboard": True,
        }
    )

    attempt = 0
    while attempt < driver_attempts:
        try:
            driver = webdriver.Remote(f"http://127.0.0.1:{appium_port}", options=options)
            break  # Exit the loop if driver is successfully created
        except Exception as e:
            attempt += 1
            print(f"Attempt {attempt} failed: {e}")
            time.sleep(5)  # Wait for a short period before trying again

    if driver:
        print("Driver successfully created")
    else:
        print(f"Failed to create driver after {driver_attempts} attempts")
    time.sleep(1.0)

    return driver


def get_screenshot(driver):
    screenshot_base64 = driver.get_screenshot_as_base64()
    screenshot_bytes = base64.b64decode(screenshot_base64)
    image = Image.open(io.BytesIO(screenshot_bytes))
    screenshot_np = np.array(image)[:, :, :3]
    return screenshot_np


def get_viewhierarchy(driver):
    view_hierachy_string = driver.page_source
    xml_file_like = io.StringIO(view_hierachy_string)
    view_hierachy = xml_element_tree.iterparse(xml_file_like)
    return view_hierachy
