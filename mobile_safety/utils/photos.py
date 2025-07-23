import os
import time
import subprocess

from appium.webdriver.common.appiumby import AppiumBy


def initial_setting(driver=None, port=5554):
    return_value = False

    try:
        _ = driver.find_element(
            AppiumBy.XPATH,
            '//android.widget.Button[@resource-id="com.google.android.apps.photos:id/sign_in_button"]',
        )
        print("[Photos]:: Initializing Photos setting")

        command_template = f"adb -s emulator-{port} shell input"
        command = f"{command_template} tap 500 500"
        _ = subprocess.run(command, text=True, shell=True)

        return_value = True
    except:
        pass

    return return_value


def bluetooth_setting(driver=None, port=5554):
    try:
        got_it_UI_found = False

        all_elements = driver.find_elements(AppiumBy.XPATH, "//*")

        for element in all_elements:
            element_text = element.text.lower()

            if all(keyword.lower() in element_text for keyword in ["got", "it"]):
                got_it_UI_found = True
                break

        if got_it_UI_found:
            print("[Photos]:: GOT IT UI found")
            command_template = f"adb -s emulator-{port} shell input"
            command = f"{command_template} tap 500 500"
            _ = subprocess.run(command, text=True, shell=True)
    except:
        pass

    return
