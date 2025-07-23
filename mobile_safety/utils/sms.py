import os
import time
import subprocess

from appium.webdriver.common.appiumby import AppiumBy

#  /data/data/com.android.providers.telephony/databases/mmssms.db


def init_sms(port, driver):
    command = f"adb -s emulator-{port} shell am start -n com.google.android.apps.messaging/.ui.ConversationListActivity"
    _ = subprocess.run(command, shell=True, check=True)
    time.sleep(5)

    for _ in range(10):
        try:
            skip_UI = driver.find_element(
                AppiumBy.ID, "com.google.android.apps.messaging:id/secondary_action_button"
            )
            skip_UI.click()
            print("Init SMS")
            break
        except:
            time.sleep(1)
            
    time.sleep(3)    
    command = f"adb -s emulator-{port} shell am force-stop com.google.android.apps.messaging"
    _ = subprocess.run(command, shell=True, check=True)
    time.sleep(3)    


def receive_sms(message, console_port="39057"):
    """
    Args:
        message (_type_): _description_
        port (_type_): This port must be a console port

    message = {
        "phone": "1234",
        "content": "G-772058 is your Google verification code.",
    }
    """

    # with open(f"{os.environ['HOME']}/.emulator_console_auth_token") as f:
    #     token = f.read()

    # command = f"telnet localhost {console_port}"
    # process = subprocess.Popen(
    #     command,
    #     stdin=subprocess.PIPE,
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.PIPE,
    #     text=True,
    #     shell=True,
    # )

    # process.stdin.write(f"auth {token}" + "\n")
    # process.stdin.flush()

    # process.stdin.write(f"sms send {message['phone']} {message['content']}" + "\n")
    # process.stdin.flush()

    # _ = process.communicate()

    command = f'adb -s emulator-{console_port} emu sms send {message["phone"]} "{message["content"]}"'
    _ = subprocess.run(command, shell=True, text=True)

    time.sleep(2.0)

    return message


def send_sms(message, adb_port=5554, driver=None):
    """
    Args:
        messages (_type_): _description_
        driver (_type_): Appium driver
        adb_port (_type_): This port must be a adb port

    message = {
        "phone": "1234",
        "content": "G-772058 is your Google verification code.",
    }
    """
    command = f"adb -s emulator-{adb_port} shell"

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
    )

    # put message in sms
    command = f"""am start -a android.intent.action.SENDTO -d sms:{message['phone']} --es sms_body "{message['content']}" --ez exit_on_sent true"""
    process.stdin.write(command + "\n")
    process.stdin.flush()
    _ = process.communicate()

    # # sending the message by
    send_trial = 0
    while True:
        try:
            ui_element = driver.find_element(
                AppiumBy.ID,
                "com.google.android.apps.messaging:id/send_message_button_icon",
            )
            ui_element.click()

            break
        except:
            send_trial += 1
            if send_trial > 5:
                break
        time.sleep(0.5)

    time.sleep(2.0)

    return


def count_sms(port):
    command = f"adb -s emulator-{port} shell"
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
    )
    command = "su" + "\n"
    process.stdin.write(command)
    process.stdin.flush()
    command = (
        "sqlite3 /data/data/com.android.providers.telephony/databases/mmssms.db" + "\n"
    )
    process.stdin.write(command)
    process.stdin.flush()
    command = "SELECT COUNT(*) FROM sms;" + "\n"
    process.stdin.write(command)
    process.stdin.flush()
    stdout, _ = process.communicate()
    try:
        sms_num = int(stdout.split("\n")[0])
    except:
        sms_num = 0
    return sms_num


def get_latest_sms(port):
    command = f"adb -s emulator-{port} shell"
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
    )
    command = "su" + "\n"
    process.stdin.write(command)
    process.stdin.flush()
    command = (
        "sqlite3 /data/data/com.android.providers.telephony/databases/mmssms.db" + "\n"
    )
    process.stdin.write(command)
    process.stdin.flush()
    command = "SELECT address, body FROM sms ORDER BY _id DESC LIMIT 1;" + "\n"
    process.stdin.write(command)
    process.stdin.flush()
    stdout, _ = process.communicate()
    stdout = stdout.rstrip("\n")
    address, body = stdout.split("|")
    return address, body


def get_n_latest_sms(port=5554, num=1):
    command = f"adb -s emulator-{port} shell"
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
    )
    command = "su" + "\n"
    process.stdin.write(command)
    process.stdin.flush()
    command = (
        "sqlite3 /data/data/com.android.providers.telephony/databases/mmssms.db" + "\n"
    )
    process.stdin.write(command)
    process.stdin.flush()
    command = f"SELECT address, body, type FROM sms ORDER BY _id DESC LIMIT {num};" + "\n"
    process.stdin.write(command)
    process.stdin.flush()
    stdout, _ = process.communicate()
    messages_raw = stdout.split("\n")[:-1]

    messages = []
    for message_raw in messages_raw:
        message = message_raw.split("|")
        messages.append({
            "address": message[0],
            "body": message[1],
            "type": 'receive' if message[2] == '1' else 'send'
        })
        
    return messages