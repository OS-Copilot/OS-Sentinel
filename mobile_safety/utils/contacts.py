import os
import time
import subprocess


def set_contact(contact, adb_port=5554):
    """_summary_

    Args:
        port (_type_): _description_
        contacts (_type_): _description_

    contacts = [
        {"id": 1, "name": "Google", "phone": "1234"},
        {"id": 2, "name": "John", "phone": "1357"},
    ]
    """

    command = f"adb -s emulator-{adb_port} shell content insert --uri content://contacts/people --bind name:s:{contact['name']}"
    _ = subprocess.run(command, text=True, shell=True)
    command = f"adb -s emulator-{adb_port} shell content insert --uri content://com.android.contacts/data --bind raw_contact_id:i:{contact['id']} --bind mimetype:s:vnd.android.cursor.item/phone_v2 --bind data1:s:{contact['phone']} --bind data2:s:Mobile --bind data3:s:Else"
    _ = subprocess.run(command, text=True, shell=True)

    time.sleep(0.1)

    return
