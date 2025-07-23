import re
import time
import subprocess
from appium.webdriver.common.appiumby import AppiumBy


def extract_numbers(input_string):
    return re.sub(r"\D", "", input_string)


def call(port=5554, driver=None, number=""):
    number = extract_numbers(number)
    command = f"adb -s emulator-{port} shell am start -a android.intent.action.CALL -d tel:{number}"
    _ = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if number == "911":
        while True:
            try:
                call_UI = driver.find_element(AppiumBy.XPATH, '//android.widget.Button[@content-desc="dial"]')
                call_UI.click()
                break
            except:
                time.sleep(0.5)
    return number
    

def check_call_number(driver):
    if driver.current_activity == "com.android.dialer.incall.activity.ui.InCallActivity":
        try:
            number_UI = driver.find_element(
                AppiumBy.ID, "com.google.android.dialer:id/contactgrid_contact_name")
            number = number_UI.text.replace('\u202a', '').replace('\u202c', '')
            return number
        except:
            return "special_call"
    else:
     	return "None"
  
  
def is_calling(port=5554):
    command = f'adb -s emulator-{port} shell dumpsys telephony.registry | grep "mCallState"'
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if "mCallState=2" in result.stdout.decode("utf-8"):
        return True
    else:
        return False