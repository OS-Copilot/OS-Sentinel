#!/bin/bash
#grant wallpaper permission
port=$1
adb -s emulator-$port shell am start \
    -a android.intent.action.ATTACH_DATA \
    -c android.intent.category.DEFAULT \
    -t 'image/*' \
    -e mimeType 'image/*'
sleep 15
adb -s emulator-$port shell input touchscreen tap 558 1785
sleep 5
adb -s emulator-$port shell input touchscreen tap 963 1920
sleep 10
adb -s emulator-$port shell am force-stop com.google.android.apps.photos
sleep 10
echo "permission complete"