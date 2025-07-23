#!/bin/bash
#create as pixel_3_test (change to input later)
avdmanager delete avd -n $1
echo "$1 avd deleted"
# source permission_set.sh
