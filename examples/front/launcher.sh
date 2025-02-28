#!/bin/bash

# zeroai runtime file

run_file="/home/jetson/dut/ZeroAI/bin/pro/runtime.pkl"

log_file="/home/jetson/dut/front_reboot_algorithm.log"

if [ -f "$run_file" ]; then

        rm -f "$run_file"

        echo $(date)" [shutdown algorithm]"

        echo $(date)" [shutdown algorithm]" >> $log_file

fi


sleep 5


echo $(date)" [launch algorithm]"

echo $(date)" [launch algorithm]" >> $log_file

cd /home/jetson/dut/ZeroAI

nohup python bin/main-pro.py > /home/jetson/dut/nohup_zero.log 2>&1 &

cd ~
