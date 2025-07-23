import os
import subprocess


_WORK_PATH = os.environ["MOBILE_SAFETY_HOME"]
_RESOURCE_PATH = f"{_WORK_PATH}/asset/environments/resource"


def get_image_path(category):
    """
    Args:
        category (list): list of folders
    """
    image_folder = f"{_RESOURCE_PATH}/image/{category}"
    image_list = os.listdir(image_folder)
    image_list = [image for image in image_list if image.endswith(".jpg")]

    return image_list


def push_file_to_device(src_path, dst_path, adb_port=5554):
    """
    Args:
        src_path (str): path of image
        dst_path (str): path of image
        port (int): port number
    """
    device = f"emulator-{adb_port}"
    command = f"adb -s {device} push {src_path} {dst_path}"  # /sdcard/Downloads/, /sdcard/Pictures/, ...
    _ = subprocess.run(command, text=True, shell=True)

    return


def get_sdcard_files(adb_port=5554, folder="Download"):
    # get file names
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
    process.stdin.write(f"cd sdcard/{folder}" + "\n")
    process.stdin.flush()
    process.stdin.write("ls" + "\n")
    process.stdin.flush()
    stdout, _ = process.communicate()

    # file names
    file_names = stdout.split("\n")[:-1]
    file_names = list(set(file_names))

    return file_names
