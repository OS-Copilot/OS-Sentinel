#!/bin/bash

avd_name=$1
port=$2
max_attempts=20
attempts=0

emulator -port $port -avd $avd_name -no-audio -no-window -no-skin -no-snapshot -gpu "swiftshader_indirect" &
# emulator -port $port -avd $avd_name -no-audio -no-skin -no-snapshot -gpu "swiftshader_indirect" &
adb -s emulator-$port wait-for-device

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

echo "Emulator turn on complete"
