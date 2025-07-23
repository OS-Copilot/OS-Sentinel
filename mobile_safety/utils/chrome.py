import time
import subprocess

from appium.webdriver.common.appiumby import AppiumBy


def chrome_activated(port):
    """_summary_

    Args:
        port (int): _description_

    Returns:
        bool: _description_
    """
    try:
        output = subprocess.check_output(
            f"adb -s emulator-{port} shell dumpsys activity activities | grep com.android.chrome",
            shell=True,
        )
        if "mFocusedApp" in str(output):
            return True
        else:
            return False
    except:
        return False
    
    
def chrome_init(port, driver):
    command = f"adb -s emulator-{port} shell am start -n com.android.chrome/com.google.android.apps.chrome.Main"
    _ = subprocess.run(command, shell=True, check=True)
    time.sleep(5)
    
    for _ in range(10):
        try:
            skip_UI = driver.find_element(
                AppiumBy.ID, "com.android.chrome:id/signin_fre_dismiss_button"
            )
            skip_UI.click()
            break
        except:
            time.sleep(1)
            
    for _ in range(10):
        try:
            skip_UI = driver.find_element(
                AppiumBy.ID, "com.android.chrome:id/negative_button"
            )
            skip_UI.click()
            print("Init Chrome")
            break
        except:
            time.sleep(1)
       
    time.sleep(3)    
    command = f"adb -s emulator-{port} shell am force-stop com.android.chrome"
    _ = subprocess.run(command, shell=True, check=True)
    time.sleep(3)    
    
    command = (
		f"adb -s emulator-{port} "
		"shell am start -a android.intent.action.VIEW -d "
		"'https://www.google.com' com.android.chrome"
	)
    _ = subprocess.run(command, text=True, shell=True)

    time.sleep(10)

    try:
        notification_UI = driver.find_element(
                AppiumBy.XPATH, '//android.widget.Button[@resource-id="com.android.chrome:id/negative_button"]'
            )
        notification_UI.click()
    except:
        pass

    command = (
            f"adb -s emulator-{port} "
            "shell am start -a android.intent.action.VIEW -d "
            "'https://www.google.com' com.android.chrome"
        )
    _ = subprocess.run(command, text=True, shell=True)

    time.sleep(5)

    try:
        pass_UI = driver.find_element(
                AppiumBy.XPATH, '//android.view.View[@resource-id="gb-main"]/android.view.View[1]'
            )
        pass_UI.click()
    except:
        pass
    
    command = (
		f"adb -s emulator-{port} "
		"shell am force-stop com.android.chrome"
	)
    _ = subprocess.run(command, text=True, shell=True)
    

def open_chrome(port):
    
    command = (
		f"adb -s emulator-{port} "
		"shell am start -a android.intent.action.VIEW -d "
		"'https://www.google.com' com.android.chrome"
	)
    _ = subprocess.run(command, text=True, shell=True)

    time.sleep(30)
    
    command = (
		f"adb -s emulator-{port} "
		"shell am force-stop com.android.chrome"
	)
    _ = subprocess.run(command, text=True, shell=True)