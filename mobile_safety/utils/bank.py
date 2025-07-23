import os
import time
import sqlite3
import subprocess
import pandas as pd

from appium.webdriver.common.appiumby import AppiumBy


_WORK_PATH = os.environ["MOBILE_SAFETY_HOME"]
_RESOURCE_PATH = f"{_WORK_PATH}/asset/environments/resource"


def install_bank(adb_port=5554):
    app_path = f"{_RESOURCE_PATH}/apks/Bank.apk"
    install_command = f"adb -s emulator-{adb_port} install {app_path}"

    _ = subprocess.run(install_command, shell=True, check=True)
    time.sleep(3.0)

    return


def login_bank(
    adb_port=5554, driver=None, account_id="a@a.com", account_pw="aa", initialize=False
):
    """_summary_
    Try login to create bank database

    Args:
        port (_type_): _description_

    Returns:
        _type_: _description_
    """
    # run bank app
    command = f"adb -s emulator-{adb_port} shell am start -n com.example.bankApp/.MainActivity"
    _ = subprocess.run(command, text=True, shell=True)
    time.sleep(3.0)

    # try login
    trial = 0
    while True:
        try:
            email_id = driver.find_element(
                AppiumBy.ID, "com.example.bankApp:id/textInputEditTextEmail"
            )
            email_id.send_keys(account_id)
            time.sleep(1.0)

            email_pw = driver.find_element(
                AppiumBy.ID, "com.example.bankApp:id/textInputEditTextPassword"
            )
            email_pw.send_keys(account_pw)
            time.sleep(1.0)

            login_button = driver.find_element(
                AppiumBy.ID, "com.example.bankApp:id/appCompatButtonLogin"
            )
            login_button.click()
            time.sleep(1.0)

            break
        except:
            trial += 1
            if trial > 5:
                return None

    # Exit app
    if initialize:
        command = f"adb -s emulator-{adb_port} shell am force-stop com.example.bankApp"
        _ = subprocess.run(command, text=True, shell=True)
        time.sleep(0.3)

    return None


def auto_login(
    adb_port=5554, driver=None, account_id="Jeffrey@gmail.com", account_pw="1234"
):
    try:
        _ = driver.find_element(AppiumBy.XPATH, '//android.widget.TextView[@text="We pursue a relaxed banking experience"]')
        print("Auto login")
        login_bank(adb_port, driver, account_id, account_pw, initialize=False)
    except:
        pass

def reset_bank(adb_port=5554):
    """
    Reset all accounts in bank

    Args:
        adb_port (_type_): _description_

    Returns:
        _type_: _description_
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
    process.stdin.write("su\n")
    process.stdin.flush()

    command = f"sqlite3 /data/data/com.example.bankApp/databases/DBbank.db"
    process.stdin.write(command + "\n")
    process.stdin.flush()

    command = f"DELETE FROM users;"
    process.stdin.write(command + "\n")
    process.stdin.flush()

    process.stdin.write(".exit\n")  # Exit sqlite3
    process.stdin.flush()
    process.stdin.close()
    process.wait()

    time.sleep(3.0)

    return None


def get_bank_account(adb_port=5554):
    """
    Get all accounts info in bank

    Args:
        adb_port (_type_): _description_

    Returns:
        _type_: _description_
    """
    # get db data
    command = f"adb -s emulator-{adb_port} shell"
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
    )
    process.stdin.write("su\n")
    process.stdin.flush()

    command = f"sqlite3 /data/data/com.example.bankApp/databases/DBbank.db"
    process.stdin.write(command + "\n")
    process.stdin.flush()

    command = f"SELECT * FROM users;"
    process.stdin.write(command + "\n")
    process.stdin.flush()

    stdout, _ = process.communicate()
    raw_data = stdout.split("\n")[:-1]

    # parse data
    account_data = []
    for raw in raw_data:
        raw = raw.split("|")
        account_data.append(
            {
                "id": int(raw[0]),
                "accountID": int(raw[1]),
                "balance": int(raw[2]),
                "name": raw[3],
                "phoneNumber": raw[4],
                "email": raw[5],
                "password": raw[6],
            }
        )

    return account_data


def init_bank_account(account_data, adb_port=5554):
    """
    Init accounts info to given data

    Args:
        adb_port (_type_): _description_
        account_data (_type_): _description_

    account_data = [
        {
            'accountID': 111111,
            'balance': 50000,
            'name': 'John',
            'phoneNumber': '123456',
            'email': "John@gmail.com",
            'password': '1234'
        },
        {
            'accountID': 111112,
            'balance': 50000,
            'name': 'Boss',
            'phoneNumber': '123123',
            'email': "Boss@gmail.com",
            'password': '1234'
        }
    ]

    Returns:
        _type_: _description_
    """
    # delete account
    reset_bank(adb_port)

    # add account
    command = f"adb -s emulator-{adb_port} shell"
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
    )
    process.stdin.write("su\n")
    process.stdin.flush()
    process.stdin.write(
        "sqlite3 /data/data/com.example.bankApp/databases/DBbank.db" + "\n"
    )
    process.stdin.flush()

    # add rows
    for id, row in enumerate(account_data):
        command = f"INSERT INTO users (id, accountID, balance, name, phoneNumber, email, password) VALUES ({id+1}, {row['accountID']}, {row['balance']}, '{row['name']}', '{row['phoneNumber']}', '{row['email']}', '{row['password']}');"
        print(command)
        process.stdin.write(command + "\n")
        process.stdin.flush()

    # exit sqlite3
    process.stdin.write(".exit\n")
    process.stdin.flush()
    process.stdin.close()
    process.wait()

    return get_bank_account(adb_port)


def get_transactions(adb_port=5554):
    """
    Get trasaction data

    Args:
        adb_port (_type_): _description_

    Returns:
        _type_: _description_
    """
    # get db data
    command = f"adb -s emulator-{adb_port} shell"
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
    )
    process.stdin.write("su\n")
    process.stdin.flush()
    command = f"sqlite3 /data/data/com.example.bankApp/databases/DBbank.db"
    process.stdin.write(command + "\n")
    process.stdin.flush()
    command = f"SELECT * FROM transactions;"
    process.stdin.write(command + "\n")
    process.stdin.flush()
    stdout, _ = process.communicate()
    raw_data = stdout.split("\n")[:-1]

    # parse data
    transaction_data = []
    for raw in raw_data:
        raw = raw.split("|")
        transaction_data.append(
            {
                "id": int(raw[0]),
                "sent": int(raw[1]),
                "received": int(raw[2]),
                "amount": int(raw[3]),
            }
        )

    return transaction_data

def bank_activated(port):
    try:
        output = subprocess.check_output(
            f"adb -s emulator-{port} shell dumpsys activity activities | grep com.example.bankApp",
            shell=True,
        )
        if "mFocusedApp" in str(output):
            return True
        else:
            return False
    except:
        return False