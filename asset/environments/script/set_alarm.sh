#!/bin/bash
db_location=$1
port=$2
attempts=0
max_attempts=20

# adb -s emulator-$port root
# sleep 5
adb -s emulator-$port push $db_location /data/user_de/0/com.google.android.deskclock/databases/alarms.db
sleep 2
adb -s emulator-$port reboot
sleep 10
until adb -s emulator-$port shell getprop sys.boot_completed | grep -m 1 "1"; do
    if [ $attempts -ge $max_attempts ]; then
        echo "Failed to start emulator after $max_attempts attempts."
        exit 1
    fi

    echo "Waiting for emulator to fully boot..."
    sleep 5
    attempts=$((attempts + 1))
done
sleep 10