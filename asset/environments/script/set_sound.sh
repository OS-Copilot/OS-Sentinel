#!/bin/bash
sound=$1
port=$2

adb -s emulator-$port shell media volume --show --stream 1 --set $sound
sleep 2
adb -s emulator-$port shell media volume --show --stream 3 --set $sound
sleep 2
adb -s emulator-$port shell media volume --show --stream 4 --set $((sound + 1))