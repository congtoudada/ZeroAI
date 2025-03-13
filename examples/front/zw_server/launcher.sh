#!/bin/bash



# zeroai runtime file

run_file="/home/ps/dut/ZeroAI/bin/pro/runtime.pkl"

log_file="/home/ps/dut/front_reboot_algorithm.log"

if [ -f "$run_file" ]; then

        rm -f "$run_file"

        echo $(date)" [shutdown algorithm]"

        echo $(date)" [shutdown algorithm]" >> $log_file

fi



sleep 5



export LD_LIBRARY_PATH=/home/ps/dut/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH

export LIBRARY_PATH=/home/ps/dut/TensorRT-8.6.1.6/lib:$LIBRARY_PATH



echo $(date)" [launch algorithm]"

echo $(date)" [launch algorithm]" >> $log_file

cd /home/ps/dut/ZeroAI

nohup /home/ps/miniconda3/envs/zeroai38/bin/python bin/main-pro.py > /home/ps/dut/nohup_zero.log 2>&1 &

cd ~
