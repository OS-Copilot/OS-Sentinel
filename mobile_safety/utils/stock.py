import os
import time
import sqlite3
import subprocess
import pandas as pd

from appium import webdriver
from appium.options.common.base import AppiumOptions
from appium.webdriver.common.appiumby import AppiumBy


_WORK_PATH = os.environ["MOBILE_SAFETY_HOME"]


def install_stock_app(adb_port=5554):
    stock_app_path = f"{_WORK_PATH}/asset/environments/resource/apks/StockTrainer.apk"
    command = f"adb -s emulator-{adb_port} install {stock_app_path}"
    _ = subprocess.run(command, text=True, shell=True)
    time.sleep(2)
    

def open_stock_app(adb_port=5554):
    command = f"adb -s emulator-{adb_port} shell am start -n com.alifesoftware.stocktrainer/.activities.ApplicationFlavorSelectorActivity"
    _ = subprocess.run(command, text=True, shell=True)
    
    
def close_stock_app(adb_port=5554):
    command = f"adb -s emulator-{adb_port} shell am force-stop com.alifesoftware.stocktrainer"
    _ = subprocess.run(command, text=True, shell=True)


def init_stock_app(adb_port=5554, driver=None):

    # appium connection
    # options = AppiumOptions()
    # options.load_capabilities(
    #     {
    #         "platformName": "Android",
    #         "appium:platformVersion": "10",
    #         "appium:udid": f"emulator-{adb_port}",
    #         "appium:automationName": "UiAutomator2",
    #         "appium:appPackage": "com.alifesoftware.stocktrainer",
    #         "appium:appActivity": ".activities.ApplicationFlavorSelectorActivity",
    #         "appium:ensureWebviewsHavePages": True,
    #         "appium:nativeWebScreenshot": True,
    #         "appium:newCommandTimeout": 90000,
    #         "appium:connectHardwareKeyboard": True,
    #     }
    # )
    # driver = webdriver.Remote("http://127.0.0.1:4723", options=options)
    # time.sleep(10)
    
    open_stock_app(adb_port)
    time.sleep(15)

    # UI control
    while True:
        try:
            accept_UI = driver.find_element(
                AppiumBy.ID, "com.alifesoftware.stocktrainer:id/md_buttonDefaultPositive"
            )
            accept_UI.click()
            break
        except:
            time.sleep(1)
    time.sleep(1)

    while True:
        try:
            anon_login_UI = driver.find_element(
                AppiumBy.ID, "com.alifesoftware.stocktrainer:id/anonymousLoginButton"
            )
            anon_login_UI.click()
            break
        except:
            time.sleep(1)
    time.sleep(5)

    # select United States
    for i in range(17):
        command = f"adb -s emulator-{adb_port} shell input keyevent 20"
        _ = subprocess.run(command, text=True, shell=True)
        time.sleep(0.2)
    command = f"adb -s emulator-{adb_port} shell input keyevent 66"
    _ = subprocess.run(command, text=True, shell=True)
    time.sleep(1)

    select_UI = driver.find_element(
        AppiumBy.ID, "com.alifesoftware.stocktrainer:id/md_buttonDefaultPositive"
    )
    select_UI.click()
    time.sleep(1)

    ok_UI = driver.find_element(
        AppiumBy.ID, "com.alifesoftware.stocktrainer:id/md_buttonDefaultPositive"
    )
    ok_UI.click()
    time.sleep(20)
    
    # go back home
    command = f"adb -s emulator-{adb_port} shell input keyevent KEYCODE_HOME"
    _ = subprocess.run(command, text=True, shell=True)
    time.sleep(1)

    return None


def get_stockdb_filename(adb_port=5554):
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
    process.stdin.write("cd /data/data/com.alifesoftware.stocktrainer/databases" + "\n")
    process.stdin.flush()
    process.stdin.write("ls\n")
    process.stdin.flush()
    stdout, _ = process.communicate()
    raw_file_names = stdout.split("\n")[:-1]
    db_file = [
        filename
        for filename in raw_file_names
        if filename.endswith("usercomstocktrainerapp.db")
    ][0]
    return db_file


def set_stock_db(db_data, adb_port=5554):
    """_summary_

    Args:
        port (_type_): _description_
        db_data (_type_): _description_

    db_data = [
        {'symbol': 'NVDA',
        'name': 'NVIDIA Corporation',
        'numberstocks': 1,
        'price': 100,
        'date': '2024-07-12'},
        {'symbol': 'AAPL',
        'name': 'APPLE Inc.',
        'numberstocks': 1,
        'price': 200,
        'date': '2024-07-12'},
    ]

    Returns:
        _type_: _description_
    """
    # run sqlite
    db_name = get_stockdb_filename(adb_port)
    print(db_name)
    command = f"adb -s emulator-{adb_port} shell"
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
    )
    process.stdin.write("su" + "\n")
    process.stdin.flush()
    command = (
        f"sqlite3 /data/data/com.alifesoftware.stocktrainer/databases/{db_name}" + "\n"
    )
    print(command)
    process.stdin.write(command)
    process.stdin.flush()

    # clear table
    command = "DELETE FROM stocksowned;"
    process.stdin.write(command)
    process.stdin.flush()
    
    # add rows
    for id, row in enumerate(db_data):
        command = f"INSERT INTO stocksowned (_id, symbol, name, numberstocks, price, date) VALUES ({id+1}, '{row['symbol']}', '{row['name']}', {row['numberstocks']}, {row['price']}, '{row['date']}');"
        print(command)
        process.stdin.write(command)
        process.stdin.flush()

    # get result
    command = "SELECT * FROM stocksowned;"
    process.stdin.write(command)
    process.stdin.flush()
    stdout, _ = process.communicate()
    print(stdout)

    return None


def get_balance(adb_port=5554):
    db_name = get_stockdb_filename(adb_port)
    command = f"adb -s emulator-{adb_port} shell"
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
    )
    process.stdin.write("su" + "\n")
    process.stdin.flush()
    command = (
        f"sqlite3 /data/data/com.alifesoftware.stocktrainer/databases/{db_name}" + "\n"
    )
    print(command)
    process.stdin.write(command)
    process.stdin.flush()

    command = "SELECT balance FROM balance;"
    process.stdin.write(command)
    process.stdin.flush()
    stdout, _ = process.communicate()
    balance = float(stdout[:-2])

    return balance


def get_stock_db(adb_port=5554):
    db_name = get_stockdb_filename(adb_port)
    command = f"adb -s emulator-{adb_port} shell"
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
    )
    process.stdin.write("su" + "\n")
    process.stdin.flush()
    command = (
        f"sqlite3 /data/data/com.alifesoftware.stocktrainer/databases/{db_name}" + "\n"
    )
    print(command)
    process.stdin.write(command)
    process.stdin.flush()

    command = "SELECT * FROM stocksowned;"
    process.stdin.write(command)
    process.stdin.flush()
    stdout, _ = process.communicate()
    raw_data = stdout.split("\n")[:-1]

    db_data = []
    for raw in raw_data:
        raw = raw.split("|")
        db_data.append(
            {
                "id": int(raw[0]),
                "symbol": raw[1],
                "name": raw[2],
                "numberstocks": int(raw[3]),
                "price": float(raw[4]),
                "date": raw[5],
            }
        )

    return db_data


def get_numberstocks(stock_db, symbol):
    for stock in stock_db:
        if stock["symbol"] == symbol:
            return stock["numberstocks"]
    return -1


def get_transaction(adb_port=5554):
	db_name = get_stockdb_filename(adb_port)
	command = f"adb -s emulator-{adb_port} shell"
	process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
	process.stdin.write('su'+'\n')
	process.stdin.flush()
	command = f'sqlite3 /data/data/com.alifesoftware.stocktrainer/databases/{db_name}'+'\n'
	process.stdin.write(command)
	process.stdin.flush()

	command = "SELECT * FROM transactionsdb;"
	process.stdin.write(command)
	process.stdin.flush()
	stdout, _ = process.communicate()
	raw_data = stdout.split('\n')[:-1]

	transaction_data = []
	for raw in raw_data:
		raw = raw.split('|')
		transaction_data.append({
			'id': raw[0],
			'type': raw[2],
			'symbol': raw[3],
			'name': raw[4],
			'numberstocks': int(raw[5]),
			'price': float(raw[6]),
			'date': raw[1],
			'isgain': bool(raw[10]=='loss'),
		})
  
	return transaction_data


def pass_advertisement(driver=None):
    try:
        _ = driver.page_source
        pass_ad_UI = driver.find_element(AppiumBy.XPATH, '//android.view.View[@resource-id="close-button"]')
        pass_ad_UI.click()
        print("pass advertisement")
    except:
        pass

def stock_activated(port):
    try:
        output = subprocess.check_output(
            f"adb -s emulator-{port} shell dumpsys activity activities | grep com.alifesoftware.stocktrainer",
            shell=True,
        )
        if "mFocusedApp" in str(output):
            return True
        else:
            return False
    except:
        return False