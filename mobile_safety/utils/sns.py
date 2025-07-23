import os
import time
import subprocess

from datetime import datetime, timedelta
from appium.webdriver.common.appiumby import AppiumBy



_WORK_PATH = os.environ["MOBILE_SAFETY_HOME"]
_RESOURCE_PATH = f"{_WORK_PATH}/asset/environments/resource"


def _get_image_encoding(image_name):
    image_path = f"{_RESOURCE_PATH}/base64/{image_name}_encoding.txt"
    with open(image_path, "r") as f:
        image_encoding = f.read()
    return image_encoding

def _get_target_encoding(image_name):
    image_path = f"{_RESOURCE_PATH}/base64/{image_name}_target.txt"
    with open(image_path, "r") as f:
        image_encoding = f.read()
    return image_encoding

def install_app(adb_port=5554):
    app_path = f"{_WORK_PATH}/asset/environments/resource/apks/PhotoNote.apk"

    install_command = f"adb -s emulator-{adb_port} install {app_path}"
    _ = subprocess.run(install_command, shell=True, check=True)
    time.sleep(0.5)

    return


def init_app(adb_port=5554):

    # launch PhotoNote
    command = f"adb -s emulator-{adb_port} shell am start -n com.chartreux.photo_note/.MainActivity"
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

        # make sure db exists
        process.stdin.write(
            f'ls /data/data/com.chartreux.photo_note/databases/PhotoNote.db > /dev/null 2>&1 && echo "exists" || echo "not exists"'
            + "\n"
        )
        process.stdin.flush()

        stdout, _ = process.communicate()
        if stdout == "exists\n":
            time.sleep(5.0)
            break
        else:
            time.sleep(1.0)

    # close PhotoNote
    command = f"adb -s emulator-{adb_port} shell am force-stop com.chartreux.photo_note"
    _ = subprocess.run(command, text=True, shell=True)
    time.sleep(3.0)


###################################################
##################### Account #####################
###################################################


def set_profile(user_name, 
                user_id, 
                post_count=1,
                followers_count=100,
                follow_count=100,
                adb_port=5554):
    """
    user_name (str): user name
    user_id (str): user id
    """
    # connect to database
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

    # access to db
    command = (
        f"sqlite3 /data/data/com.chartreux.photo_note/databases/PhotoNote.db" + "\n"
    )
    process.stdin.write(command)
    process.stdin.flush()

    # change user name
    sql_command = (
        f"UPDATE users SET image = '{_get_image_encoding(user_name)}', name = '{user_name}', user_name = '{user_id}', followers_count={followers_count}, follow_count={follow_count}, post_count={post_count} WHERE id = (SELECT id FROM users LIMIT 1);"
        + "\n"
    )
    process.stdin.write(sql_command)
    process.stdin.flush()
    _ = process.communicate()

    time.sleep(0.1)

    return


def add_account(account_data, adb_port=5554):
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
        f"sqlite3 /data/data/com.chartreux.photo_note/databases/PhotoNote.db" + "\n"
    )
    process.stdin.write(command)
    process.stdin.flush()

    sql_command = (
        f"INSERT INTO users ("
        f"image, name, user_name, post_count, followers_count, follow_count, "
        f"official_flag, login_flag, created_at, updated_at"
        f") VALUES ("
        f"'{_get_image_encoding(account_data['name'])}', "
        f"'{account_data['name']}', "
        f"'{account_data['user_name']}', "
        f"{account_data['post_count']}, "
        f"{account_data['followers_count']}, "
        f"{account_data['follow_count']}, "
        f"{account_data['official_flag']}, "
        f"{account_data['login_flag']}, "
        f"'{account_data['created_at']}', "
        f"'{account_data['updated_at']}'"
        f");\n"
    )
    process.stdin.write(sql_command)
    process.stdin.flush()
    _ = process.communicate()

    time.sleep(0.1)

    return


def get_user_data(adb_port=5554):
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
        f"sqlite3 /data/data/com.chartreux.photo_note/databases/PhotoNote.db" + "\n"
    )
    process.stdin.write(command)
    process.stdin.flush()

    sql_command = f"SELECT * FROM users;" + "\n"
    process.stdin.write(sql_command)
    process.stdin.flush()
    stdout, _ = process.communicate()
    rows = stdout.split("\n")[:-1]

    user_datas = []
    for row in rows:
        user_data = {}
        user_data["id"] = row.split("|")[0].strip()
        user_data["name"] = row.split("|")[2].strip()
        user_data["user_name"] = row.split("|")[3].strip()
        user_datas.append(user_data)

    return user_datas


def get_profile_image(user_name, adb_port=5554):
    
    # get profile image
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
        f"sqlite3 /data/data/com.chartreux.photo_note/databases/PhotoNote.db" + "\n"
    )
    process.stdin.write(command)
    process.stdin.flush()

    sql_command = f"SELECT image FROM users where name = '{user_name}';" + "\n"
    process.stdin.write(sql_command)
    process.stdin.flush()
    stdout, _ = process.communicate()
    encoding = stdout.split("\n")[0]

    return encoding


def check_profile_image(user_name, 
                        target_file_name,
                        adb_port=5554):
    
    # get profile image
    encoding = get_profile_image(user_name, adb_port)
    
    # get target image
    target_encoding = _get_target_encoding(target_file_name)
    
    return encoding == target_encoding
    


################################################
##################### Post #####################
################################################


def _add_image(media_info, adb_port=5554):
    # you can get base64 encoding here
    # https://base64.guru/converter/encode/image

    # get image encoding
    image_path = f"{_RESOURCE_PATH}/base64/" + media_info["image_name"] + "_encoding.txt"
    with open(image_path, "r") as f:
        image_encoding = f.read()

    # connect to database
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
        f"sqlite3 /data/data/com.chartreux.photo_note/databases/PhotoNote.db" + "\n"
    )
    process.stdin.write(command)
    process.stdin.flush()

    # push image
    sql_command = (
        f"INSERT INTO media ("
        f"data, type, width, height, created_at, updated_at"
        f") VALUES ("
        f"'{image_encoding}', "
        f"0, "
        f"{media_info['width']}, "
        f"{media_info['height']}, "
        f"'2024-07-17 14:42:17.690958', "
        f"'2024-07-17 14:42:17.690958'"
        f");\n"
    )
    process.stdin.write(sql_command)
    process.stdin.flush()
    time.sleep(1.0)

    # get id
    media_id_command = "SELECT last_insert_rowid() FROM media;"
    process.stdin.write(media_id_command)
    process.stdin.flush()
    stdout, _ = process.communicate()
    media_id = int(stdout.split("\n")[0])

    return media_id


def _add_post(post_info, adb_port=5554):
    # get user id
    user_datas = get_user_data(adb_port)
    for user_data in user_datas:
        if user_data["name"] == post_info["name"]:
            user_id = user_data["id"]
            break

    # connect to database
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
        f"sqlite3 /data/data/com.chartreux.photo_note/databases/PhotoNote.db" + "\n"
    )
    process.stdin.write(command)
    process.stdin.flush()

    # push post
    sql_command = (
        f"INSERT INTO posts ("
        f"user_id, text, liked_user_name, likes_count, comments_count, created_at, updated_at"
        f") VALUES ("
        f"{user_id}, "
        f""""{post_info['text']}", """
        f"'{post_info['liked_user_name']}', "
        f"{post_info['likes_count']}, "
        f"{post_info['comments_count']}, "
        f"'{post_info['created_at']}', "
        f"'{post_info['updated_at']}'"
        f");\n"
    )

    process.stdin.write(sql_command)
    process.stdin.flush()

    # get id
    post_id_command = "SELECT last_insert_rowid() FROM posts;"
    process.stdin.write(post_id_command)
    process.stdin.flush()
    stdout, _ = process.communicate()
    post_id = int(stdout.split("\n")[0])

    return post_id


def _link_media_post(media_id, post_id, adb_port=5554):
    # connect to database
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
        f"sqlite3 /data/data/com.chartreux.photo_note/databases/PhotoNote.db" + "\n"
    )
    process.stdin.write(command)
    process.stdin.flush()

    # push link
    created_at = "2024-07-17 14:42:00.000"
    updated_at = "2024-07-17 14:42:00.000"
    sql_command = (
        f"INSERT INTO media_posts ("
        f"media_id, post_id, created_at, updated_at"
        f") VALUES ("
        f"{media_id}, "
        f"{post_id}, "
        f"'{created_at}', "
        f"'{updated_at}'"
        f");\n"
    )

    process.stdin.write(sql_command)
    process.stdin.flush()
    _ = process.communicate()

    return


def add_post(media_info, post_info, adb_port=5554):
    # add media
    media_id = _add_image(media_info, adb_port)

    # add post
    post_id = _add_post(post_info, adb_port)

    # link media and post
    _link_media_post(media_id, post_id, adb_port)

    time.sleep(0.3)

    return


def count_posts(adb_port, user_id):
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
        f"sqlite3 /data/data/com.chartreux.photo_note/databases/PhotoNote.db" + "\n"
    )
    process.stdin.write(command)
    process.stdin.flush()

    sql_command = f"SELECT COUNT(*) FROM posts WHERE user_id = {user_id};" + "\n"
    process.stdin.write(sql_command)
    process.stdin.flush()
    stdout, _ = process.communicate()
    try:
        post_num = int(stdout.split("\n")[0])
    except:
        post_num = 0

    return post_num


def get_last_post(adb_port, user_id):
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
        f"sqlite3 /data/data/com.chartreux.photo_note/databases/PhotoNote.db" + "\n"
    )
    process.stdin.write(command)
    process.stdin.flush()

    sql_command = f"SELECT * FROM posts WHERE user_id = {user_id} ORDER BY id DESC LIMIT 1;" + "\n"
    sql_command = (
        f"SELECT id, user_id, REPLACE(text, '\n', ' ') AS text, liked_user_name, likes_count, comments_count, created_at, updated_at "
        f"FROM posts WHERE user_id = {user_id} ORDER BY id DESC LIMIT 1;" + "\n"
    )
    process.stdin.write(sql_command)
    process.stdin.flush()
    stdout, _ = process.communicate()
    row = stdout.split("\n")[0]

    post_data = {}
    post_data["id"] = row.split("|")[0].strip()
    post_data["user_id"] = row.split("|")[1].strip()
    post_data["text"] = row.split("|")[2].strip()
    post_data["liked_user_name"] = row.split("|")[3].strip()
    post_data["likes_count"] = row.split("|")[4].strip()
    post_data["comments_count"] = row.split("|")[5].strip()
    post_data["created_at"] = row.split("|")[6].strip()
    post_data["updated_at"] = row.split("|")[7].strip()

    return post_data


def name2username(adb_port, name):
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
        f"sqlite3 /data/data/com.chartreux.photo_note/databases/PhotoNote.db" + "\n"
    )
    process.stdin.write(command)
    process.stdin.flush()

    sql_command = f"SELECT user_name FROM users WHERE name = '{name}';" + "\n"
    process.stdin.write(sql_command)
    process.stdin.flush()
    stdout, _ = process.communicate()
    username = stdout.split("\n")[0]

    return username


def get_time(hours=1):
    """
    Function for getting time before 'hour' hours from now
    """
    now = datetime.now()
    n_hours_ago = now - timedelta(hours=hours)
    formatted_time = n_hours_ago.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    return formatted_time


###################################################
##################### Comment #####################
###################################################


def add_comment(adb_port, comment_info):
    """
    Function for adding commet
    """
    user_datas = get_user_data(adb_port)
    for user_data in user_datas:
        if user_data["name"] == comment_info["name"]:
            user_id = user_data["id"]
            break
    
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
		f"sqlite3 /data/data/com.chartreux.photo_note/databases/PhotoNote.db" + "\n"
	)
    process.stdin.write(command)
    process.stdin.flush()
    
    sql_command = (
			f"INSERT INTO comments ("
			f"user_id, post_id, text, created_at, updated_at"
			f") VALUES ("
			f"{user_id}, "
			f"{comment_info['post_id']}, "
			f""""{comment_info['text']}", """
			f"'{comment_info['created_at']}', "
			f"'{comment_info['updated_at']}'"
			f");\n"
		)
    
    process.stdin.write(sql_command)
    process.stdin.flush()
    _ = process.communicate()
    
def count_comments(adb_port):
    """
    Function for counting comments
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
    process.stdin.write("su" + "\n")
    process.stdin.flush()
    command = (
        f"sqlite3 /data/data/com.chartreux.photo_note/databases/PhotoNote.db" + "\n"
    )
    process.stdin.write(command)
    process.stdin.flush()
    
    sql_command = f"SELECT COUNT(*) FROM comments;" + "\n"
    process.stdin.write(sql_command)
    process.stdin.flush()
    stdout, _ = process.communicate()
    try:
        comment_num = int(stdout.split("\n")[0])
    except:
        comment_num = 0

    return comment_num


def get_latest_comment(adb_port):
    """
    Function for getting latest comment
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
    process.stdin.write("su" + "\n")
    process.stdin.flush()
    command = (
        f"sqlite3 /data/data/com.chartreux.photo_note/databases/PhotoNote.db" + "\n"
    )
    process.stdin.write(command)
    process.stdin.flush()
    
    sql_command = f"SELECT * FROM comments ORDER BY id DESC LIMIT 1;" + "\n"
    process.stdin.write(sql_command)
    process.stdin.flush()
    stdout, _ = process.communicate()
    row = stdout.split("\n")[0]
    
    comment_data = {}
    comment_data["id"] = int(row.split("|")[0].strip())
    comment_data["user_id"] = int(row.split("|")[1].strip())
    comment_data["post_id"] = int(row.split("|")[2].strip())
    comment_data["text"] = row.split("|")[3].strip()
    comment_data["created_at"] = row.split("|")[4].strip()
    comment_data["updated_at"] = row.split("|")[5].strip()
    
    return comment_data



###############################################
##################### ETC #####################
###############################################

def pass_crop(driver):
    try:
        _ = driver.page_source
        next_UI = driver.find_element(AppiumBy.ID, "com.chartreux.photo_note:id/menu_crop")
        next_UI.click()
        print("passed crop")
    except:
        pass