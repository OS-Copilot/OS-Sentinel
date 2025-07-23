import re
import subprocess
import time
import sys

def start_emulator(avd_name, port, max_attempts=20):
    attempts = 0

    # Start the emulator
    emulator_command = [
        "emulator", 
        "-port", str(port), 
        "-avd", avd_name, 
        "-no-audio", 
        "-no-skin", 
        "-no-snapshot-load", 
        "-gpu", "swiftshader_indirect"
    ]
    subprocess.Popen(emulator_command)

    # Wait for the device to be ready
    adb_command = ["adb", "-s", f"emulator-{port}", "wait-for-device"]
    subprocess.run(adb_command, check=True)

    # Wait for the emulator to fully boot
    while attempts < max_attempts:
        result = subprocess.run(
            ["adb", "-s", f"emulator-{port}", "shell", "getprop", "sys.boot_completed"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        
        if "1" in result.stdout:
            print("Emulator turn on complete")
            return
        
        print("Waiting for emulator to fully boot...")
        time.sleep(5)
        attempts += 1
    
    print(f"Failed to start emulator after {max_attempts} attempts.")
    sys.exit(1)
    

def get_screen_size(port):
    try:
        command = f"adb -s emulator-{port} shell wm size"
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        if result.returncode != 0:
            raise Exception(f"Failed to execute adb command: {result.stderr}")
        output = result.stdout.strip()

        # parse width, height
        match = re.search(r'Physical size: (\d+)x(\d+)', output)
        if match:
            width = int(match.group(1))
            height = int(match.group(2))
            return width, height
        else:
            raise ValueError("Invalid wm size output format")

    except Exception as e:
        print(e)
        return None, None
    
    
def kill_emulator(port):
    command = f"adb -s emulator-{port} emu kill"
    _ = subprocess.run(command, shell=True, text=True)
    time.sleep(20)
    
    
def set_time(port, curr_time):
    
    command = f"adb -s emulator-{port} shell su 0 settings put global auto_time 0"
    _ = subprocess.run(command, shell=True, text=True)
    time.sleep(0.3)
    
    command = f"adb -s emulator-{port} shell su 0 settings put global auto_time_zone 0"
    _ = subprocess.run(command, shell=True, text=True)
    time.sleep(0.3)
    
    command = f"adb -s emulator-{port} shell su 0 date {curr_time}"
    _ = subprocess.run(command, shell=True, text=True)
    time.sleep(0.3)
    
    command = f"adb -s emulator-{port} shell su 0 toybox date {curr_time}"
    _ = subprocess.run(command, shell=True, text=True)
    time.sleep(0.3)
    

if __name__ == "__main__":
    get_screen_size(5554)
