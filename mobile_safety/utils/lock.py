import time
import subprocess

# Only can set password while no password is set

def set_PIN(adb_port=5554, PIN=1234, credential=None):
    if credential is None:
        command = f"adb -s emulator-{adb_port} shell locksettings set-pin {PIN}"
    else:
        command = f"adb -s emulator-{adb_port} shell locksettings set-pin --old {credential} {PIN}"
    print(command)
    _ = subprocess.run(command, text=True, shell=True)
    time.sleep(1)
    
    return PIN

def set_password(adb_port=5554, password="password", credential=None):
    if credential is None:
        command = f"adb -s emulator-{adb_port} shell locksettings set-password {password}"
    else:
        command = f"adb -s emulator-{adb_port} shell locksettings set-password --old {credential} {password}"
    print(command)
    _ = subprocess.run(command, text=True, shell=True)
    time.sleep(1)
    
    return password

def set_pattern(adb_port=5554, pattern="password", credential=None):
    if credential is None:
        command = f"adb -s emulator-{adb_port} shell locksettings set-pattern {pattern}"
    else:
        command = f"adb -s emulator-{adb_port} shell locksettings set-pattern --old {credential} {pattern}"
    print(command)
    _ = subprocess.run(command, text=True, shell=True)
    time.sleep(1)
    
    return pattern

def verify_credential(adb_port=5554, credential="password"):
    command = f"adb -s emulator-{adb_port} shell locksettings verify --old {credential}"
    stdout, _ = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True).communicate()
    stdout = stdout.decode('utf-8')
    if 'successfully' in stdout:
        return True
    else:
        return False
    
def disable_password(adb_port=5554, credential="password"):
    command = f"adb -s emulator-{adb_port} shell locksettings clear --old {credential}"
    _ = subprocess.run(command, text=True, shell=True)
    time.sleep(1)
    
    return True
    
def get_lock_type(adb_port=5554):
    command = f'adb -s emulator-{adb_port} shell dumpsys lock_settings | grep "CredentialType"'
    stdout, _ = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True).communicate()
    stdout = stdout.decode('utf-8')
    if 'none' in stdout:
        return 'none'
    elif 'NONE' in stdout:
        return 'none'
    elif 'PASSWORD' in stdout:
        return 'PASSWORD'
    elif 'PIN' in stdout:
        return 'PIN'
    elif 'PATTERN' in stdout:
        return 'PATTERN'
    else:
        return None