#!/bin/bash

avdmanager create avd -n $1 -k "$2" -d $3
echo "$1 avd created"
# source permission_set.sh
