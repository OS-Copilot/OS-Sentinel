#!bin/bash
# initial setting
dpi=$1
fontSize=$2
port=$3
adb -s emulator-$port shell wm density reset
# 1. get random icon size (min: 72)
# 2. change icon size
adb -s emulator-$port shell wm density $dpi
echo "DPI: $dpi"
sleep 2
adb -s emulator-$port shell settings put system font_scale $fontSize
echo "Icon size change complete"
sleep 5