#!/bin/bash
bmp_location=$1
port=$2
# #1. List images and select on randomly
# echo "Selected Image: $image_name"
# # 2. Change wallpaper in to the avd
# adb -s shell am start \
#     -a android.intent.action.ATTACH_DATA \
#     -c android.intent.category.DEFAULT \
#     -d file:///sdcard/Download/background.jpeg \
#     -t 'image/*' \
#     -e mimeType 'image/*'

# sleep 10
# adb -s emulator-$port shell input touchscreen tap 808 2090
# sleep 10
adb -s emulator-$port push $bmp_location /data/system/users/0/wallpaper
sleep 4