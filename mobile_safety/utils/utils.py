import os
import re
import enum
import time
import json
import random
import portpicker
import subprocess
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# import torch


_WORK_PATH = os.environ["MOBILE_SAFETY_HOME"]


def chunk_text(text, chunk_size=16):
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def reboot_device(adb_port=5554):
    device = f"emulator-{adb_port}"
    subprocess.run(f"adb -s {device} reboot", text=True, shell=True)
    subprocess.run(f"adb -s {device} wait-for-device", text=True, shell=True)

    while True:
        output = subprocess.run(
            f"adb -s emulator-{adb_port} shell getprop sys.boot_completed",
            text=True,
            shell=True,
            capture_output=True,
        )
        if output.stdout.strip() == "1":
            break
        time.sleep(1.0)
    time.sleep(2.0)


def run_emulator(avd_name, adb_port=5554, gui=False):
    if gui:
        script_path = f"{_WORK_PATH}/asset/environments/script/emulator_on_gui.sh"
    else:
        script_path = f"{_WORK_PATH}/asset/environments/script/emulator_on.sh"
    print(script_path)
    command = [script_path, str(avd_name), str(adb_port)]
    print(f"run_emulator called with: avd_name={avd_name}, port={adb_port}, gui={gui}")
    print("Before executing command:", " ".join(command))
    try:
        subprocess.run(command, check=True)
        print("Command executed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def emulator_snaphot_load(
    snapshot_name,
    adb_port=5554,
):
    device = f"emulator-{adb_port}"
    script_path = f"{_WORK_PATH}/asset/environments/script/snapshot_load.sh"
    command = [script_path, snapshot_name, str(adb_port)]
    try:
        subprocess.run(command, check=True)
        print("Command executed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def get_port(start=5554, end=6000, interval=2):
    for port in range(start, end, interval):
        if portpicker.is_port_free(port):
            return port
    return portpicker.pick_unused_port()


def get_empty_ports(start=5556, end=6000, interval=2, port_num=1):
    ports = []
    while len(ports) < port_num:
        for port in range(start, end, interval):
            if portpicker.is_port_free(port):
                ports.append(port)
            else:
                ports.append(portpicker.pick_unused_port())
            if len(ports) == port_num:
                break
    return ports


def run_emulators(avd_list, gui=False):
    ports = get_empty_ports(port_num=len(avd_list))
    print(avd_list)
    print(ports)
    with ProcessPoolExecutor(max_workers=len(avd_list)) as executor:
        futures = [
            executor.submit(run_emulator, avd_name, port, gui)
            for avd_name, port in zip(avd_list, ports)
        ]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred while running emulator: {e}")

    return ports


def get_running_emulators():
    command = "adb devices"
    output = subprocess.run(command, text=True, shell=True, capture_output=True)
    devices = output.stdout.split("\n")[1:-2]
    devices = [device.split("\t")[0] for device in devices]
    return devices


def close_emulator(emulator_name):
    command = f"adb -s {emulator_name} emu kill"
    _ = subprocess.run(command, text=True, shell=True)
    
    while True:
        result = subprocess.run("adb devices", text=True, shell=True, capture_output=True)
        if emulator_name not in result.stdout:
            break
        time.sleep(2)
    time.sleep(5)


def get_time():
	return time.strftime("%m%d%H%M%Y.%S", time.localtime())


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    

class StepType(enum.IntEnum):
    """Defines the status of a `TimeStep` within a sequence."""

    # Denotes the first `TimeStep` in a sequence.
    FIRST = 0
    # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
    MID = 1
    # Denotes the last `TimeStep` in a sequence.
    LAST = 2
    # Denotes a `TimeStep` that is not part of a sequence.
    ERROR = -1

    def first(self) -> bool:
        return self is StepType.FIRST

    def mid(self) -> bool:
        return self is StepType.MID

    def last(self) -> bool:
        return self is StepType.LAST
    
    def error(self) -> bool:
        return self is StepType.ERROR
