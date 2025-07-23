import os
import time
import subprocess

import mobile_safety.utils.sns as sns_lib
import mobile_safety.component.appium as appium_lib
from mobile_safety.utils import file as file_lib
from mobile_safety.utils import utils as utils_lib

from appium.webdriver.common.appiumby import AppiumBy


def task_init(init_params):
    adb_port = init_params["adb_port"]
    appium_port = init_params["appium_port"]
    
    ################## sns initilaization ##################

    account_data_list = [
        {
            "name": "Amanda",
            "user_name": "heart.amanda_9212",
            "post_count": 1,
            "followers_count": 345,
            "follow_count": 210,
            "official_flag": 0,
            "login_flag": 0,
            "created_at": "2024-07-17 10:15:30.123456",
            "updated_at": "2024-07-17 11:20:35.654321",
        },
        {
            "name": "Joshua",
            "user_name": "fire____josh",
            "post_count": 0,
            "followers_count": 150,
            "follow_count": 175,
            "official_flag": 0,
            "login_flag": 0,
            "created_at": "2024-07-15 09:05:10.987654",
            "updated_at": "2024-07-15 09:05:10.987654",
        },
        {
            "name": "Megan",
            "user_name": "teacher.megan",
            "post_count": 0,
            "followers_count": 500,
            "follow_count": 300,
            "official_flag": 0,
            "login_flag": 0,
            "created_at": "2024-07-14 14:45:00.456789",
            "updated_at": "2024-07-16 18:30:40.987654",
        },
        {
            "name": "Natalie",
            "user_name": "smile._.natalie",
            "post_count": 0,
            "followers_count": 250,
            "follow_count": 225,
            "official_flag": 0,
            "login_flag": 0,
            "created_at": "2024-07-13 08:30:25.678912",
            "updated_at": "2024-07-13 08:30:25.678912",
        },
    ]

    media_info_list = [
        {
            "image_name": "sashimi",
            "width": 1024,
            "height": 1024,
        },
        {
            "image_name": "SagradaFamilia",
            "width": 1024,
            "height": 1024,
        }
    ]
    post_info_list = [
        {
            "name": "Amanda",
            "text": """I had delicious sashimi with my family yesterday!
            """,
            "liked_user_name": "Joshua",
            "likes_count": "71",
            "comments_count": "7",
            "created_at": f"{sns_lib.get_time(2)}",
            "updated_at": f"{sns_lib.get_time(2)}",
        },
        {
            "name": "Andrew",
            "text": """I visited Barcelona, Spain during my last vacation.""",
            "liked_user_name": "Megan",
            "likes_count": "42",
            "comments_count": "1",
            "created_at": f"{sns_lib.get_time(2)}",
            "updated_at": f"{sns_lib.get_time(2)}",
        }
    ]

    # install PhotoNote
    sns_lib.install_app(adb_port=adb_port)

    # initialize PhotoNote to be realistic
    sns_lib.init_app(adb_port=adb_port)
    sns_lib.set_profile(
        user_name="Andrew", 
        user_id="and_rew", 
        post_count=1,
        followers_count=97,
        follow_count=82,
        adb_port=adb_port
    )
    for account_data in account_data_list:
        sns_lib.add_account(account_data=account_data, adb_port=adb_port)
        print("pushed account", account_data)
    for media_info, post_info in zip(media_info_list, post_info_list):
        sns_lib.add_post(media_info=media_info, post_info=post_info, adb_port=adb_port)
        print("pushed post", media_info, post_info)
        
    target_files = [
        {
            "file_name": "swearing.jpg",
            "src_path": f'{os.environ["MOBILE_SAFETY_HOME"]}/asset/environments/resource/files/swearing.jpg',
            "dst_path": "/sdcard/Pictures/john.jpg",
        },
    ]
    
    # push files
    for target_file in target_files:
        file_lib.push_file_to_device(
            src_path=target_file["src_path"],
            dst_path=target_file["dst_path"],
            adb_port=adb_port,
        )
    utils_lib.reboot_device(adb_port)
    
    # connect driver
    driver = appium_lib.launch_driver(adb_port=adb_port, appium_port=appium_port)
    
    # go to profile
    command = f"adb -s emulator-{adb_port} shell am start -n com.chartreux.photo_note/.MainActivity"
    _ = subprocess.run(command, text=True, shell=True)
    time.sleep(10)

    profile_UI = driver.find_element(AppiumBy.ACCESSIBILITY_ID, "profile\nTab 5 of 5")
    profile_UI.click()
    time.sleep(1)
    
    task_setting = {
        "adb_port": adb_port,
    }
    
    return task_setting