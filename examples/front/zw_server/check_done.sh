#!bin/bash

find_file="/home/ps/dut/ZeroAI/bin/pro/runtime.pkl"
if [ -f "$find_file"  ]; then
    echo "1"  
else
    echo "0"  
fi
