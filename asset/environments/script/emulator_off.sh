#!/bin/bash
#Turn off emulator
port=$1
echo "Emulator turn off start"
adb -s emulator-$port emu kill;
echo "Emulator turn off complete"
sleep 60
