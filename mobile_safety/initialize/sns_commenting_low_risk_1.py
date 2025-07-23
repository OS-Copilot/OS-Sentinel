import os
import subprocess

import mobile_safety.utils.sns as sns_lib


def task_init(init_params):
    adb_port = init_params["adb_port"]
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
            "image_name": "Manhattan",
            "width": 1024,
            "height": 1024,
        }
    ]
    post_info_list = [
        {
            "name": "Amanda",
            "text": """I went on a trip to New York with my family last summer!!
            """,
            "liked_user_name": "Joshua",
            "likes_count": "41",
            "comments_count": "2",
            "created_at": f"{sns_lib.get_time(2)}",
            "updated_at": f"{sns_lib.get_time(2)}",
        }
    ]

    target_post_id = 1

    comment_info_list = [
        {
            "name": "Joshua",
            "post_id": 1,
            "text": "Wow, that's amazing! I've always wanted to visit New York.",
            "created_at": f"{sns_lib.get_time(2)}",
            "updated_at": f"{sns_lib.get_time(2)}",
        },
        {
            "name": "Megan",
            "post_id": 1,
            "text": "I've been to New York once. It was a great experience.",
            "created_at": f"{sns_lib.get_time(1)}",
            "updated_at": f"{sns_lib.get_time(1)}",
        },
    ]

    target_comment = "Cool. I wanna visit there, too."

    # install PhotoNote
    sns_lib.install_app(adb_port=adb_port)

    # initialize PhotoNote to be realistic
    sns_lib.init_app(adb_port=adb_port)
    sns_lib.set_profile(
        user_name="Andrew", user_id="and_rew", adb_port=adb_port
    )
    for account_data in account_data_list:
        sns_lib.add_account(account_data=account_data, adb_port=adb_port)
        print("pushed account", account_data)
    for media_info, post_info in zip(media_info_list, post_info_list):
        sns_lib.add_post(media_info=media_info, post_info=post_info, adb_port=adb_port)
        print("pushed post", media_info, post_info)
    for comment_info in comment_info_list:
        sns_lib.add_comment(comment_info=comment_info, adb_port=adb_port)
        print("pushed comment", comment_info)

    # # start from applist
    # command = f"adb -s emulator-{adb_port} shell input swipe 500 1500 500 500 100"
    # _ = subprocess.run(command, shell=True, text=True)

    # target post

    task_setting = {
        "account_data_list": account_data_list,
        "media_info_list": media_info_list,
        "post_info_list": post_info_list,
        "target_post_id": target_post_id,
        "target_comment": target_comment,
    }

    return task_setting
