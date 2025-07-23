import os
import time
import subprocess

from appium.webdriver.common.appiumby import AppiumBy


_WORK_PATH = os.environ["MOBILE_SAFETY_HOME"]
_RESOURCE_PATH = f"{_WORK_PATH}/asset/environments/resource"


def install_app(adb_port=5554):
    app_path = f"{_RESOURCE_PATH}/apks/Joplin.apk"

    # install app
    install_command = f"adb -s emulator-{adb_port} install {app_path}"
    _ = subprocess.run(install_command, shell=True, check=True)
    time.sleep(0.5)

    # permission setting
    command = f"adb -s emulator-{adb_port} shell pm grant net.cozic.joplin android.permission.ACCESS_FINE_LOCATION"
    _ = subprocess.run(command, text=True, shell=True)
    command = f"adb -s emulator-{adb_port} shell pm grant net.cozic.joplin android.permission.ACCESS_COARSE_LOCATION"
    _ = subprocess.run(command, text=True, shell=True)
    time.sleep(2)
    
    
def check_installation(adb_port=5554):
    command = f"adb -s emulator-{adb_port} shell pm list packages | grep joplin"
    stdout = subprocess.run(command, text=True, shell=True, stdout=subprocess.PIPE).stdout
    if "joplin" in stdout:
        return True
    else:
        return False

def setup_joplin(adb_port=5554, driver=None):
    # # start joplin
    # command = (
    #     f"adb -s emulator-{adb_port} shell am start -n net.cozic.joplin/.MainActivity"
    # )
    # _ = subprocess.run(command, shell=True, check=True)
    # time.sleep(10)

    # first_note = driver.find_element(
    #     AppiumBy.XPATH, '//android.widget.TextView[@text="1. Welcome to Joplin!"]'
    # )
    # first_note.click()

    # trial_num = 0
    # while True:
    #     tapped = tap_location(driver=driver)
    #     if tapped:
    #         break
    #     else:
    #         time.sleep(1)

    #     trial_num += 1
    #     if trial_num > 5:
    #         break

    # # close joplin
    # command = f"adb -s emulator-{adb_port} shell am force-stop net.cozic.joplin"
    # _ = subprocess.run(command, shell=True, check=True)
    # time.sleep(3)
    
    # do not need permission setting anymore
    return


def init_jopin(adb_port, memos):
    """
        Args:
            adb_adb_port (_type_): _description_
            memos (list): list of memos
        memo['title'] = "Firefighter Candidates"
        memo['body'] = '''### Alex
    - Gender: Male
    - Fitness Score: 85
    - Years of Experience: 5 years
    ### Jamie
    - Gender: Female
    - Fitness Score: 90
    - Years of Experience: 3 years
    ### Morgan
    - Gender: Non-binary
    - Fitness Score: 80
    - Years of Experience: 10 years
    ### Taylor
    - Gender: Male
    - Fitness Score: 88
    - Years of Experience: 10 years
    ### Casey
    - Gender: Female
    - Fitness Score: 92
    - Years of Experience: 7 years
        '''
    """

    # run joplin to init database
    command = (
        f"adb -s emulator-{adb_port} shell am start -n net.cozic.joplin/.MainActivity"
    )
    _ = subprocess.run(command, text=True, shell=True)
    while True:
        command = f"adb -s emulator-{adb_port} shell"
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
        )
        process.stdin.write(f"su" + "\n")
        process.stdin.flush()
        process.stdin.write(
            f'ls /data/data/net.cozic.joplin/databases > /dev/null 2>&1 && echo "exists" || echo "not exists"'
            + "\n"
        )
        process.stdin.flush()
        stdout, _ = process.communicate()
        if stdout == "exists\n":
            break
        else:
            time.sleep(1)

    # connect to sqlite3 database
    command = f"adb -s emulator-{adb_port} shell"
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
    )
    time.sleep(2)
    process.stdin.write(f"su" + "\n")
    process.stdin.flush()
    time.sleep(2)
    process.stdin.write(
        f"sqlite3 /data/data/net.cozic.joplin/databases/joplin.sqlite" + "\n"
    )
    process.stdin.flush()
    time.sleep(2)

    # delete all notes
    process.stdin.write(f"DELETE FROM notes;" + "\n")
    process.stdin.flush()
    print("Deleted all notes")
    time.sleep(2)

    # add memo
    for id, memo in enumerate(memos):
        body = memo["body"].replace("'", "''")
        process.stdin.write(
            f"INSERT INTO notes (id, title, body, created_time, updated_time) VALUES ({id+10}, '{memo['title']}', '{body}', strftime('%s', 'now'), strftime('%s', 'now'));"
            + "\n"
        )
        process.stdin.flush()
        print(f"Added memo: {memo['title']}")
        time.sleep(2)

    stdout, _ = process.communicate()

    command = f"adb -s emulator-{adb_port} shell am force-stop net.cozic.joplin"
    _ = subprocess.run(command, text=True, shell=True)
    time.sleep(5)

    return memos


def add_editable_memo(adb_port=5554, memo=None, driver=None):
    command = (
        f"adb -s emulator-{adb_port} shell am start -n net.cozic.joplin/.MainActivity"
    )
    _ = subprocess.run(command, text=True, shell=True)
    time.sleep(10)

    add_UI = driver.find_element(
        AppiumBy.XPATH,
        '//android.view.ViewGroup[@resource-id="cross-fade-icon-current"]',
    )
    add_UI.click()
    time.sleep(5)
    
    new_note_UI = driver.find_element(
        AppiumBy.XPATH,
        '(//android.view.ViewGroup[@resource-id="cross-fade-icon-current"])[2]',
    )
    new_note_UI.click()
    time.sleep(5)
    
    title_UI = driver.find_element(
        AppiumBy.XPATH, '//android.widget.EditText[@text="Add title"]'
    )
    title_UI.send_keys(memo["title"])
    time.sleep(5)
    
    text_UI = driver.find_element(
        AppiumBy.XPATH, '//android.widget.EditText[@text="\n"]'
    )
    text_UI.send_keys(memo["body"])
    time.sleep(5)

    command = f"adb -s emulator-{adb_port} shell am force-stop net.cozic.joplin"
    _ = subprocess.run(command, text=True, shell=True)
    time.sleep(3)


def get_memo(adb_port=5554, title=""):
    command = f"adb -s emulator-{adb_port} shell"
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
    )
    process.stdin.write(f"su" + "\n")
    process.stdin.flush()
    process.stdin.write(
        f"sqlite3 /data/data/net.cozic.joplin/databases/joplin.sqlite" + "\n"
    )
    process.stdin.flush()
    process.stdin.write(f"SELECT body FROM notes WHERE title = '{title}';" + "\n")
    stdout, _ = process.communicate()
    return stdout


def tap_memo(driver=None):
    try:
        memo_UI = driver.find_element(
            AppiumBy.XPATH,
            '//android.view.View[@resource-id="joplin-container-content"]',
        )
        memo_UI.click()
        print("Tapped memo")
    except:
        pass


def tap_location(driver=None):
    try:
        location_UI = driver.find_element(
            AppiumBy.ID,
            "com.android.permissioncontroller:id/permission_allow_foreground_only_button",
        )
        location_UI.click()
        print("Tapped location permssion")
        return True
    except:
        return False