import os
import time
import sqlite3
import subprocess
import pandas as pd

from appium.webdriver.common.appiumby import AppiumBy


def auto_login(adb_port=5554, driver=None):
    return_value = False

    # skip log-in
    try:
        skip_UI = driver.find_element(
            AppiumBy.XPATH,
            f'//android.widget.Button[@text="SKIP"]',
        )
        skip_UI.click()
        print("[Maps]:: Log-in UI clicked")

        time.sleep(5.0)
    except:
        print("[Maps]:: Log-in UI not found")
        pass

    # activate current location
    try:
        current_UI = driver.find_element(
            AppiumBy.XPATH,
            f'(//android.widget.FrameLayout[@resource-id="com.google.android.apps.maps:id/fab_icon"])[3]/android.widget.FrameLayout[2]/android.widget.ImageView',
        )
        current_UI.click()
        print("[Maps]:: Current location UI clicked")

        time.sleep(5.0)
    except:
        print("[Maps]:: Current location UI not found")
        pass

    # activate allow while using this app
    try:
        allow_UI = driver.find_element(
            AppiumBy.ID,
            f"com.android.permissioncontroller:id/permission_allow_foreground_only_button",
        )
        allow_UI.click()
        return_value = True
        print("[Maps]:: Allow while using this app UI clicked")

        time.sleep(5.0)
    except:
        print("[Maps]:: Allow while using this app UI not found")
        pass

    return return_value
