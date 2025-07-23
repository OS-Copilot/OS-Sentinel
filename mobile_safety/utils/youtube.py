import time
import subprocess

from appium.webdriver.common.appiumby import AppiumBy


def init_youtube(port, driver):
    command = f"adb -s emulator-{port} shell am start -n com.google.android.youtube/com.google.android.youtube.HomeActivity"
    _ = subprocess.run(command, shell=True, check=True)
    time.sleep(5)

    for _ in range(10):
        try:
            skip_UI = driver.find_element(
                AppiumBy.ID, "com.android.permissioncontroller:id/permission_deny_button"
            )
            skip_UI.click()
            print("Init Youtube")
            break
        except:
            time.sleep(1)
            
    time.sleep(3)    
    command = f"adb -s emulator-{port} shell am force-stop com.google.android.youtube"
    _ = subprocess.run(command, shell=True, check=True)
    time.sleep(3)   