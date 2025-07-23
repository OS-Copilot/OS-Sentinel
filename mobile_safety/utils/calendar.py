import os
import re
import pytz
import time
import datetime
import subprocess


_WORK_PATH = os.environ["MOBILE_SAFETY_HOME"]


def init_calendar(adb_port=5554):
    # disable google calendar
    command = f"adb -s emulator-{adb_port} shell pm disable-user --user 0 com.google.android.calendar"
    _ = subprocess.run(command, text=True, shell=True)
    time.sleep(2)

    # app install
    stock_app_path = f"{_WORK_PATH}/asset/environments/resource/apks/SimpleCalendarPro.apk"
    command = f"adb -s emulator-{adb_port} install {stock_app_path}"
    _ = subprocess.run(command, text=True, shell=True)
    time.sleep(2)
    
    # permission setting
    command = f"adb -s emulator-{adb_port} shell pm grant com.simplemobiletools.calendar.pro android.permission.POST_NOTIFICATIONS"
    _ = subprocess.run(command, text=True, shell=True)
    command = f"adb -s emulator-{adb_port} shell pm grant com.simplemobiletools.calendar.pro android.permission.WRITE_CALENDAR"
    _ = subprocess.run(command, text=True, shell=True)
    command = f"adb -s emulator-{adb_port} shell pm grant com.simplemobiletools.calendar.pro android.permission.POST_NOTIFICATIONS"
    _ = subprocess.run(command, text=True, shell=True)
    time.sleep(2)
    
    # launch Calendar
    command = f"adb -s emulator-{adb_port} shell am start -n com.simplemobiletools.calendar.pro/.activities.MainActivity"
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
            f'ls /data/data/com.simplemobiletools.calendar.pro/databases/events.db > /dev/null 2>&1 && echo "exists" || echo "not exists"'
            + "\n"
        )
        process.stdin.flush()

        stdout, _ = process.communicate()
        if stdout == "exists\n":
            time.sleep(5.0)
            break
        else:
            time.sleep(1.0)
                   
    # close Calendar
    command = f"adb -s emulator-{adb_port} shell am force-stop com.simplemobiletools.calendar.pro"
    _ = subprocess.run(command, text=True, shell=True)
    time.sleep(3.0)
    
    # set shared preferences
    command = f"adb -s emulator-{adb_port} push {_WORK_PATH}/asset/environments/resource/files/calendar_pref.xml /sdcard/Download/Prefs.xml"
    _ = subprocess.run(command, text=True, shell=True)
    
    command = f"adb -s emulator-{adb_port} shell"
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
    )

    command = f"su" + "\n"
    process.stdin.write(command)
    process.stdin.flush()
    
    command = f"cp /sdcard/Download/Prefs.xml /data/data/com.simplemobiletools.calendar.pro/shared_prefs/Prefs.xml" + "\n"
    process.stdin.write(command)
    process.stdin.flush()
    
    
def check_installation(adb_port=5554):
    command = f"adb -s emulator-{adb_port} shell pm list packages | grep com.simplemobiletools.calendar.pro"
    stdout = subprocess.run(command, text=True, shell=True, stdout=subprocess.PIPE).stdout
    if "com.simplemobiletools.calendar.pro" in stdout:
        return True
    else:
        return False
    
    
def get_timezone(adb_port):
	command = f"adb -s emulator-{adb_port} shell getprop persist.sys.timezone"
	process = subprocess.Popen(
		command,
		stdin=subprocess.PIPE,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True,
		shell=True,
	)
	timezone, _ = process.communicate()
	timezone = timezone.strip()
	return timezone


def get_current_time(adb_port=5554):
	# get current time of android emulator
	# output: current time in unix timestamp
	command = f"adb -s emulator-{adb_port} shell date +%s"
	process = subprocess.Popen(
		command,
		stdin=subprocess.PIPE,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True,
		shell=True,
	)
	stdout, _ = process.communicate()
	current_time = int(stdout)
	return current_time


def convert_timestamp_to_timezone(timestamp, timezone_str):
    """converts a unix timestamp to a timezone aware datetime object

    Args:
        timestamp (_type_): _description_
        timezone_str (_type_): _description_

    Returns:
        localized_time: datetime object in the specified timezone
    """
    timezone = pytz.timezone(timezone_str)
    if type(timestamp) == str:
        timestamp = int(timestamp)
    utc_time = datetime.datetime.utcfromtimestamp(timestamp)
    localized_time = utc_time.replace(tzinfo=pytz.utc).astimezone(timezone)
    
    return localized_time


def modify_timestamp(timestamp, days=0, hours=0):
    """_summary_

    Args:
        tumestamp (_type_): _description_
        days (_type_): _description_
        hours (_type_): _description_
        minutes (_type_): _description_

    Returns:
        _type_: _description_
    """
    dt_object = datetime.datetime.fromtimestamp(timestamp) \
        + datetime.timedelta(days=days)
    dt_new_time = dt_object.replace(hour=hours, minute=0, second=0, microsecond=0)
    new_timestamp = int(dt_new_time.timestamp())
    
    return new_timestamp
        


def get_events(adb_port):
    """_summary_

    Args:
        adb_port (_type_): _description_

    Returns:
        _type_: _description_
        
    [{'start_ts': 1724932800,
    'end_ts': 1724936400,
    'title': 'Tt',
    'location': '',
    'description': '',
    'repeat_interval': '0',
    'repeat_rule': '0',
    'repeat_limit': '0',
    'time_zone': 'Asia/Seoul'}]
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
        f"sqlite3 /data/data/com.simplemobiletools.calendar.pro/databases/events.db" + "\n"
    )
    process.stdin.write(command)
    process.stdin.flush()

    command = "SELECT start_ts, end_ts, title, location, description, repeat_interval, repeat_rule, repeat_limit, time_zone FROM events;"
    process.stdin.write(command)
    process.stdin.flush()
    stdout, _ = process.communicate()
    
    event_pattern = re.compile(r'(\d{10})\|(\d{10})\|(.+?)(?=(\d{10}\|\d{10})|$)', re.DOTALL)
    events = event_pattern.findall(stdout)

    keys = ['start_ts', 'end_ts', 'title', 'location', 'description', 'repeat_interval', 'repeat_rule', 'repeat_limit', 'time_zone']
    parsed_events = []

    for event in events:
        start_ts, end_ts, details = event[0], event[1], event[2]
        event_fields = details.strip().split('|')

        event_dict = {
            'start_ts': start_ts,
            'end_ts': end_ts,
            'title': event_fields[0],
            'location': event_fields[1],
            'description': event_fields[2],
            'repeat_interval': event_fields[3],
            'repeat_rule': event_fields[4],
            'repeat_limit': event_fields[5],
            'time_zone': event_fields[6]
        }
        
        parsed_events.append(event_dict)
        
    return parsed_events


def add_event(adb_port=5554, event_info=None):
    """_summary_

    Args:
        adb_port (_type_): _description_
        event_info (_type_): _description_

    Returns:
        _type_: _description_
    """
    curr_timestamp = event_info["curr_timestamp"]
    
    days = event_info["days"]
    start_hour = event_info["start_hour"]
    end_hour = event_info["end_hour"]
    start_ts = modify_timestamp(curr_timestamp, days, start_hour)
    end_ts = modify_timestamp(curr_timestamp, days, end_hour)
    
    title = event_info["title"]
    location = event_info["location"]
    description = event_info["description"]
    
    timezone = get_timezone(adb_port)
    
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
        f"sqlite3 /data/data/com.simplemobiletools.calendar.pro/databases/events.db" + "\n"
    )
    process.stdin.write(command)
    process.stdin.flush()

    command = (
        f"INSERT INTO events (start_ts, end_ts, title, location, description, repeat_interval, repeat_rule, repeat_limit, time_zone, reminder_1_minutes, reminder_2_minutes, reminder_3_minutes, reminder_1_type, reminder_2_type, reminder_3_type, repetition_exceptions, attendees, import_id, flags, event_type, parent_id, last_updated, source, availability, color, type) "
        f"VALUES ( {start_ts}, {end_ts}, '{title}', '{location}', '{description}', 0, 0, 0, '{timezone}', 10, -1, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0, {curr_timestamp}, 'simple-calendar', 0, 0, 0);"
    )
    process.stdin.write(command)
    process.stdin.flush()
    stdout, stderr = process.communicate()
    
    return stderr == ""