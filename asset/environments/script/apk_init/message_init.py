import time
import subprocess
from appium import webdriver
from appium.options.common.base import AppiumOptions
from appium.webdriver.common.appiumby import AppiumBy

def message_init(avd_name='pixel_7_train_00'):
    options = AppiumOptions()
    options.load_capabilities({
        "platformName": "Android",
        "appium:platformVersion": "14",
        "appium:deviceName": avd_name,
        "appium:automationName": "UiAutomator2",
        "appium:appPackage": "com.google.android.contacts",
        "appium:appActivity": "com.android.contacts.activities.PeopleActivity",
        "appium:ensureWebviewsHavePages": True,
        "appium:nativeWebScreenshot": True,
        "appium:newCommandTimeout": 3600,
        "appium:connectHardwareKeyboard": True
    })
    driver = webdriver.Remote("http://127.0.0.1:4723", options=options)
    time.sleep(10)

    try:
        skip_UI = driver.find_element(AppiumBy.ID, "android:id/button2")
        skip_UI.click()
        time.sleep(1)
    except: pass
    
    return None
    