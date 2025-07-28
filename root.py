import os
import subprocess
import requests

from dataclasses import dataclass
from minimal_task_runner import _find_adb_directory

def _init_env():
    repo_path = os.path.split(__file__)[0]
    os.environ["MOBILE_SAFETY_HOME"] = repo_path

    assert "JAVA_HOME" in os.environ
    assert "APPIUM_BIN" in os.environ

    if "ANDROID_SDK_ROOT" not in os.environ:
        sdk_path = os.path.split(os.path.split(_find_adb_directory())[0])[0]
        sdk_path = os.path.normpath(sdk_path)
        os.environ["ANDROID_SDK_ROOT"] = sdk_path

        os.environ["PATH"] = os.path.join(sdk_path, "platform-tools") \
            + os.pathsep \
            + os.environ["PATH"]
        os.environ["PATH"] = os.path.join(sdk_path, "emulator") \
            + os.pathsep \
            + os.environ["PATH"]

    return True

def _init_apk():
    apk_path = lambda filename: os.path.join(
        os.environ["MOBILE_SAFETY_HOME"],
        "asset/environments/resource/apks",
        filename
    )

    def download(filename, url):
        if os.path.exists(apk_path(filename)):
            return

        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(apk_path(filename), "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

    download("Joplin.apk", "...")
    download("PhotoNote.apk", "...")
    download("SimpleCalendarPro.apk", "...")
    download("StockTrainer.apk", "...")

    return True

@dataclass
class EnvParam:
    avd_name: str
    port: int

def _check_snapshot(avd_name):
    snapshots = subprocess.run(
        "emulator -snapshot-list",
        shell=True,
        capture_output=True,
        text=True
    ).stdout.split("\n")

    snapshot = [item for item in snapshots if item.startswith(avd_name)][0]
    return snapshot.split(": ")[1].split(", ")[:-1]

def _init_avd():
    from asset.environments import set_up as setup

    jpg_path = f"{setup._RESOURCE_PATH}/wallpapers_jpg"
    bmp_path = f"{setup._RESOURCE_PATH}/wallpapers_bmp"
    if len(os.listdir(jpg_path)) != len(os.listdir(bmp_path)):
        setup.convert_jpg_to_bmp()

    avd_name = "AndroidWorldAvd"
    env_builder = setup.EnvBuilder(EnvParam(avd_name=avd_name, port=5554))

    if "init" not in _check_snapshot(avd_name):
        env_builder.build_devices()

    if not any([
        item.startswith(f"{env_builder.mode}_env_")
        for item in _check_snapshot(avd_name)
    ]):
        env_builder.build_environments()

    return True

def init():
    return all([_init_env(), _init_apk(), _init_avd()])

if __name__ == "__main__":
    _init_env()
    _init_apk()
    _init_avd()
