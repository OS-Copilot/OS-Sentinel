#!/bin/bash
port=$2
image_dir=$1
#1. List images and select on randomly
echo "Push Image: $image_dir"

# 2. Put image in to the avd
dest=/sdcard/Download/
adb -s emulator-$port push "$image_dir" "$dest"
sleep 1
