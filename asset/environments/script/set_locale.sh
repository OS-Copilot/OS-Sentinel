#!/bin/bash
locale=$1
port=$2


# adb -s emulator-$port shell content delete --uri content://settings/system --where "name=\'system_locales\'"
# adb -s emulator-$port shell content insert --uri content://settings/system --bind name:s:system_locales --bind value:s:$locale
# adb -s emulator-$port shell content query --uri content://settings/system --where "name=\'system_locales\'"
# sleep 5

adb -s emulator-$port shell "setprop persist.sys.locale $locale; setprop ctl.restart zygote"
