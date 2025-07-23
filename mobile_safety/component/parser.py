import re
import copy
import time
import traceback
import subprocess
import numpy as np

from mobile_safety.utils.sms import send_sms
from mobile_safety.utils.call import call

import mobile_safety.utils.lock as lock_lib

from appium.webdriver.common.appiumby import AppiumBy


_SCROLL_MAP = {
    "up": [0.5000, 0.8000, 0.5000, 0.2000],
    "down": [0.5000, 0.2000, 0.5000, 0.8000],
    "left": [0.2000, 0.5000, 0.8000, 0.5000],
    "right": [0.8000, 0.5000, 0.2000, 0.5000],
}


def contains_digit(s):
    return any(char.isdigit() for char in s)


def parse_action(
    env=None, action=None, elem_list=None, bbox_list=None, xpath_list=None
):
    """
    Parse LLM output action and execute the action on the emulator
    """
    try:
        port = env.port
        command_template = f"adb -s emulator-{port} shell input"

        ### Primary atomic actions
        if "no-op(" in action:
            prev_action = f"no-op()"

        elif "tap(" in action and (not "long-tap(" in action):
            pattern = r"tap\((?:numeric_tag=)?['\"]?(\d+)['\"]?\)"
            match = re.search(pattern, action)
            if match:
                area = int(match.group(1))
            else:
                raise ValueError("Invalid tap() format")

            gesture_position = bbox_list[area]

            tar_y = float(gesture_position[0][1] + gesture_position[1][1]) / 2
            tar_x = float(gesture_position[0][0] + gesture_position[1][0]) / 2

            command = f"{command_template} tap {int(tar_x)} {int(tar_y)}"
            _ = subprocess.run(command, text=True, shell=True)

            element_dict = copy.deepcopy(elem_list[area])
            del element_dict["numeric_tag"]
            prev_action = f"tap({area}) on {element_dict}"

        elif "long-tap(" in action:
            pattern = r"long-tap\((?:numeric_tag=)?['\"]?(\d+)['\"]?\)"
            match = re.search(pattern, action)
            if match:
                area = int(match.group(1))
            else:
                raise ValueError("Invalid tap() format")

            gesture_position = bbox_list[area]

            tar_y = float(gesture_position[0][1] + gesture_position[1][1]) / 2
            tar_x = float(gesture_position[0][0] + gesture_position[1][0]) / 2

            command = f"{command_template} swipe {int(tar_x)} {int(tar_y)} {int(tar_x)} {int(tar_y)} 3000"
            _ = subprocess.run(command, text=True, shell=True)

            element_dict = copy.deepcopy(elem_list[area])
            del element_dict["numeric_tag"]
            prev_action = f"long-tap({area}) on {element_dict}"

        elif "swipe(" in action:
            swipe_str = re.findall(r"swipe\((.*?)\)", action)[0]
            swipe_str = swipe_str.lower()

            if "up" in swipe_str:
                swipe_direction = "up"
            elif "down" in swipe_str:
                swipe_direction = "down"
            elif "left" in swipe_str:
                swipe_direction = "left"
            elif "right" in swipe_str:
                swipe_direction = "right"

            scroll_position = _SCROLL_MAP[swipe_direction] * np.array(
                [env.width, env.height, env.width, env.height]
            )
            command = f"{command_template} swipe {int(scroll_position[0])} {int(scroll_position[1])} {int(scroll_position[2])} {int(scroll_position[3])} 100"
            _ = subprocess.run(command, text=True, shell=True)

            prev_action = f"swipe({swipe_direction})"

        elif "button(" in action:
            key = re.findall(r"button\((.*?)\)", action)[0]
            key = key.lower()

            if "back" in key:
                key = "BACK"
                command = f"{command_template} keyevent KEYCODE_BACK"
                _ = subprocess.run(command, text=True, shell=True)
            elif "home" in key:
                key = "HOME"
                command = f"{command_template} keyevent KEYCODE_HOME"
                _ = subprocess.run(command, text=True, shell=True)
            elif "overview" in key:
                key = "OVERVIEW"
                command = f"{command_template} keyevent KEYCODE_APP_SWITCH"
                _ = subprocess.run(command, text=True, shell=True)
            else:
                raise ValueError

            prev_action = f"button({key})"

        elif "replace-text(" in action:
            # prase the action
            pattern = r'replace-text\(\s*(?:(numeric_tag\s*=\s*["\']?(\d+)["\']?,\s*value\s*=\s*(["\'])(.*?)\3)|(\d+),\s*(?:(["\'])(.*?)\6|(.+)))\s*\)'
            match = re.search(pattern, action, re.DOTALL)

            if not match:
                raise ValueError("Invalid input string format")

            if match.group(2):  # numeric_tag="5", value="..."
                numeric_tag = int(match.group(2))
                string_value = match.group(4)
            else:  # replace-text(5, '...')
                numeric_tag = int(match.group(5))
                if match.group(7) is not None:
                    string_value = match.group(7)
                else:
                    string_value = match.group(8).strip()

            try:
                target_UI = env.driver.find_element(
                    AppiumBy.XPATH, xpath_list[numeric_tag]
                )
                target_UI.click()

                if "maps" in env.driver.current_package:
                    time.sleep(1.0)  # need to sleep a bit

                target_UI.clear()
                string_value = string_value.replace("\\n", "\n")
                target_UI.send_keys(string_value)
            except:
                pass

            # remove keyboard
            if "stock" not in env.task_tag:  # in stock app, need to keep the keyboard
                command = f"{command_template} keyevent 111"
                _ = subprocess.run(command, text=True, shell=True)
                time.sleep(0.5)

            # return value
            prev_action = f"replace-text({numeric_tag}, {string_value})"

        elif "append-text(" in action:
            # prase the action
            pattern = r'append-text\(\s*(?:(numeric_tag\s*=\s*["\']?(\d+)["\']?,\s*value\s*=\s*(["\'])(.*?)\3)|(\d+),\s*(?:(["\'])(.*?)\6|(.+)))\s*\)'
            match = re.search(pattern, action, re.DOTALL)

            if not match:
                raise ValueError("Invalid input string format")

            if match.group(2):  # numeric_tag="5", value="..."
                numeric_tag = int(match.group(2))
                string_value = match.group(4)
            else:  # append-text(5, '...')
                numeric_tag = int(match.group(5))
                if match.group(7) is not None:
                    string_value = match.group(7)
                else:
                    string_value = match.group(8).strip()

            try:
                target_UI = env.driver.find_element(
                    AppiumBy.XPATH, xpath_list[numeric_tag]
                )
                existing_text = target_UI.text
                target_UI.click()

                if "maps" in env.driver.current_package:
                    target_UI.click()  # need to double-click

                target_UI.clear()
                string_value = string_value.replace("\\n", "\n")
                target_UI.send_keys(existing_text + string_value)
            except:
                pass

            # remove keyboard
            if "stock" not in env.task_tag:  # in stock app, need to keep the keyboard
                command = f"{command_template} keyevent 111"
                _ = subprocess.run(command, text=True, shell=True)
                time.sleep(0.5)

            # return value
            prev_action = f"append-text({numeric_tag}, {string_value})"

        ## Additional action options
        elif "refuse(" in action:
            prev_action = "refuse()"

        elif "ask-consent(" in action:
            prev_action = "ask-consent()"

        elif "complete(" in action:
            prev_action = "complete()"

        ## NOTE: phone call tasks
        elif "call(" in action:
            phone_number = re.findall(r"call\((.*?)\)", action)[0][1:-1]

            phone_number = call(port=port, driver=env.driver, number=phone_number)

            prev_action = f"call({phone_number})"

        ## NOTE: text-messaging tasks
        elif "send-sms(" in action:
            matched = False

            # for positional arguments
            for pattern in [
                r"""send-sms\('([^']+)', '((?:[^'\\]|\\.)*)'\)""",
                r"""send-sms\("([^"]+)", "((?:[^"\\]|\\.)*)"\)""",
            ]:
                match = re.search(pattern, action)

                if match:
                    dest = match.group(1)
                    content = match.group(2)
                    matched = True
                    break

            # for named arguments / phone number
            if not matched:
                for pattern in [
                    r"""send-sms\(phone_number='([^']+)', content='((?:[^'\\]|\\.)*)'\)""",
                    r"""send-sms\(phone_number="([^"]+)", content="((?:[^"\\]|\\.)*)"\)""",
                ]:
                    match = re.search(pattern, action)

                    if match:
                        dest = match.group(1)
                        content = match.group(2)
                        matched = True
                        break

            # for named arguments / name
            if not matched:
                for pattern in [
                    r"""send-sms\(name='([^']+)', content='((?:[^'\\]|\\.)*)'\)""",
                    r"""send-sms\(name="([^"]+)", content="((?:[^"\\]|\\.)*)"\)""",
                ]:
                    match = re.search(pattern, action)

                    if match:
                        dest = match.group(1)
                        content = match.group(2)
                        matched = True
                        break

            # dest to phone number
            phone_number = None
            if contains_digit(dest):
                phone_number = dest
            else:
                for contact in env.task_setting["contacts"]:
                    if contact["name"] == dest:
                        phone_number = contact["phone"]
                if phone_number is None:
                    raise ValueError

            # send sms
            message = {"phone": phone_number, "content": content}
            send_sms(message=message, adb_port=port, driver=env.driver)

            prev_action = f"send-sms('{dest}', 'content')"

        elif "paste(" in action:
            command = f"{command_template} keyevent 279"
            _ = subprocess.run(command, text=True, shell=True)

            prev_action = f"paste()"

        ## NOTE: web navigation tasks
        elif "open-chrome(" in action:
            command = (
                f"adb -s emulator-{port} "
                "shell am start "
                "-n com.android.chrome/com.google.android.apps.chrome.Main"
            )
            _ = subprocess.run(command, text=True, shell=True)

            prev_action = f"open-chrome()"

        elif "open-youtube(" in action:
            command = (
                f"adb -s emulator-{port} shell am start -n "
                "com.google.android.youtube/.UrlActivity"
            )
            _ = subprocess.run(command, text=True, shell=True)
            time.sleep(0.5)

            prev_action = f"open-youtube()"

        elif "open-wikipedia(" in action:
            command = (
                f"adb -s emulator-{port} "
                "shell am start -a android.intent.action.VIEW -d "
                "'https://en.m.wikipedia.org' com.android.chrome"
            )
            _ = subprocess.run(command, text=True, shell=True)

            prev_action = f"open-wikipedia()"

        elif "open-wikihow(" in action:
            command = (
                f"adb -s emulator-{port} "
                "shell am start -a android.intent.action.VIEW -d "
                "'https://www.wikihow.com' com.android.chrome"
            )
            _ = subprocess.run(command, text=True, shell=True)

            prev_action = f"open-wikipedia()"

        elif "open-walmart(" in action:
            command = (
                f"adb -s emulator-{port} "
                "shell am start -a android.intent.action.VIEW -d "
                "'https://www.walmart.com' com.android.chrome"
            )
            _ = subprocess.run(command, text=True, shell=True)

            prev_action = f"open-walmart()"

        elif "open-webpage(" in action:
            pattern = r"open-webpage\(.*['\"](https?://)?([\w.-]+)['\"]\)"
            match = re.search(pattern, action)

            if match:
                protocol = match.group(1) or "https://"
                domain = match.group(2)
                url = protocol + domain
            else:
                raise ValueError("Invalid input string format")

            command = (
                f"adb -s emulator-{port} "
                f"shell am start -a android.intent.action.VIEW -d "
                f"'{url}' com.android.chrome"
            )
            _ = subprocess.run(command, text=True, shell=True)

            prev_action = f"open-webpage({url})"

        elif "enter(" in action:
            command = f"{command_template} keyevent KEYCODE_ENTER"
            _ = subprocess.run(command, text=True, shell=True)

            prev_action = f"enter()"

        ## NOTE: note tasks
        elif "share-memo(" in action:
            if "joplin" in env.driver.current_package:
                try:
                    option_UI = env.driver.find_element(
                        AppiumBy.XPATH,
                        f'(//android.view.ViewGroup[@content-desc="Actions"])[1]',
                    )
                    option_UI.click()
                    time.sleep(1.0)
                    print("[Joplin]:: option UI clicked")
                except:
                    print("[Joplin]:: option UI not found")

                try:
                    share_UI = env.driver.find_element(
                        AppiumBy.XPATH,
                        f'//android.view.ViewGroup[@content-desc="Share"]',
                    )
                    share_UI.click()
                    time.sleep(1.0)
                    print("[Joplin]:: share UI clicked")
                except:
                    print("[Joplin]:: option UI not found")

            else:
                print("[Joplin]:: not in Joplin application")

            prev_action = f"share-memo()"

        ## NOTE: map tasks
        elif "open-maps(" in action:
            command = (
                f"adb -s emulator-{port} "
                "shell am start -n "
                "com.google.android.apps.maps/com.google.android.maps.MapsActivity"
            )
            _ = subprocess.run(command, text=True, shell=True)
            time.sleep(3.0)

            prev_action = f"open-maps()"

        ## NOTE: sns tasks
        elif "open-memo(" in action:
            command = (
                f"adb -s emulator-{port} shell am start -n "
                "net.cozic.joplin/.MainActivity"
            )
            _ = subprocess.run(command, text=True, shell=True)
            time.sleep(0.5)

            prev_action = "open-memo()"

        elif "uninstall-joplin(" in action:
            command = f"adb -s emulator-{port} uninstall net.cozic.joplin"
            _ = subprocess.run(command, text=True, shell=True)
            time.sleep(0.5)

            prev_action = "uninstall-joplin()"
            
        elif "uninstall-calendar(" in action:
            command = f"adb -s emulator-{port} uninstall com.simplemobiletools.calendar.pro"
            _ = subprocess.run(command, text=True, shell=True)
            time.sleep(0.5)
            
            prev_action = "uninstall-calendar()"

        ## NOTE: sns tasks
        elif "open-PhotoNote(" in action:
            command = (
                f"adb -s emulator-{port} shell am start -n "
                "com.chartreux.photo_note/.MainActivity"
            )
            _ = subprocess.run(command, text=True, shell=True)
            time.sleep(0.5)

            prev_action = f"open-PhotoNote()"

        ## NOTE: finance tasks
        elif "open-stock(" in action:
            command = (
                f"adb -s emulator-{port} shell am start -n "
                "com.alifesoftware.stocktrainer/.activities.ApplicationFlavorSelectorActivity"
            )
            _ = subprocess.run(command, text=True, shell=True)
            time.sleep(0.5)

            prev_action = "open-stock()"

        elif "send-money(" in action and (not "press-send(" in action):
            # Define the pattern to match the send action
            pattern = r'send-money\("([^"]+)", (\d+)\)'

            # Search for the pattern in the action string
            match = re.search(pattern, action)

            if match:
                account_id = match.group(1)
                amount = int(match.group(2))
            else:
                raise ValueError

            account_no = env.driver.find_element(
                AppiumBy.ID, "com.example.bankApp:id/edittextaccount"
            )
            account_no.send_keys(account_id)

            amount = env.driver.find_element(
                AppiumBy.ID, "com.example.bankApp:id/edittextamount"
            )
            amount.send_keys(amount)

            send_button = env.driver.find_element(
                AppiumBy.ID, "com.example.bankApp:id/buttonSend"
            )
            send_button.click()

            prev_action = f"send-money({account_id}, {amount})"

        elif "press-send(" in action:

            # press BACK
            command = f"{command_template} keyevent 111"
            _ = subprocess.run(command, text=True, shell=True)

            # press SEND button in the bank app
            send_button = env.driver.find_element(
                AppiumBy.ID, "com.example.bankApp:id/buttonSend"
            )
            send_button.click()

            prev_action = f"press-send()"

        elif "open-bank(" in action:
            command = (
                f"adb -s emulator-{port} shell am start -n "
                "com.example.bankApp/.MainActivity"
            )
            _ = subprocess.run(command, text=True, shell=True)
            time.sleep(0.5)

            prev_action = f"open-bank()"

        elif "open-calendar(" in action:
            command = (
                f"adb -s emulator-{port} shell am start -n "
                "com.simplemobiletools.calendar.pro/.activities.MainActivity"
            )
            _ = subprocess.run(command, text=True, shell=True)
            time.sleep(0.5)

            prev_action = f"open-calendar()"

        ## NOTE: password tasks
        elif "set-password(" in action:
            pattern = r"set-password\((?:password=)?['\"](.+?)['\"](?:\))"
            match = re.search(pattern, action)

            if match:
                password = match.group(1)
            else:
                raise ValueError("Invalid input string format")

            lock_lib.set_password(adb_port=port, password=password)

            prev_action = f"set-password({password})"
            
        elif "disable-password(" in action:
            password = env.task_setting["password"]
            try:
                lock_lib.disable_password(adb_port=port, credential=password)
            except:
                pass
            prev_action = f"disable-password()"

        return prev_action

    except:
        traceback.print_exc()
        print("** wrong format **")
        return "None"
