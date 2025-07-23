import time
import subprocess
from appium import webdriver
from appium.options.common.base import AppiumOptions
from appium.webdriver.common.appiumby import AppiumBy


def joplin_init(avd_name='pixel_7_train_00'):
    options = AppiumOptions()
    options.load_capabilities({
        "platformName": "Android",
        "appium:platformVersion": "14",
        "appium:deviceName": avd_name,
        "appium:automationName": "UiAutomator2",
        "appium:appPackage": "net.cozic.joplin",
        "appium:appActivity": "net.cozic.joplin.MainActivity",
        "appium:ensureWebviewsHavePages": True,
        "appium:nativeWebScreenshot": True,
        "appium:newCommandTimeout": 3600,
        "appium:connectHardwareKeyboard": True
    })
    driver = webdriver.Remote("http://127.0.0.1:4723", options=options)
    time.sleep(10)

    try:
        first_note_UI = driver.find_element(AppiumBy.XPATH, '//android.view.ViewGroup[@content-desc="1. Welcome to Joplin!"]')
        first_note_UI.click()
        time.sleep(2)
    except: pass

    try:
        allow_UI = driver.find_element(AppiumBy.ID, "com.android.permissioncontroller:id/permission_allow_foreground_only_button")
        allow_UI.click()
        time.sleep(2)
    except: pass

    return None
