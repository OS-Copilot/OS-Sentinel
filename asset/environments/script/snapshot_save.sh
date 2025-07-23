#!/bin/bash
#get snapshot name as arg
snapshot_name=$1
port=$2
if [ -z "$snapshot_name" ]; then
    echo "Usage: $0 <snapshot_name>"
    exit 1
fi
# adb -s emulator-$port shell input keyevent KEYCODE_HOME
sleep 3
adb -s emulator-$port emu avd snapshot save $snapshot_name
echo "Snapshot saved as $snapshot_name"

sleep 10