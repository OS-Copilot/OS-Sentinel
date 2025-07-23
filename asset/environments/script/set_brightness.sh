#!/bin/bash
brightness=$1
port=$2

adb -s emulator-$port shell settings put system screen_brightness 255