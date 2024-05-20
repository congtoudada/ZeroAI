#!/bin/bash

cd /data2/czy/mianyang/ZeroAI
source /data2/czy/anaconda3/bin/activate zeroai39

CUDA_VISIBLE_DEVICES=0 python bin/main.py --application conf/application-dev0.yaml &
CUDA_VISIBLE_DEVICES=1 python bin/main.py --application conf/application-dev1.yaml &
CUDA_VISIBLE_DEVICES=2 python bin/main.py --application conf/application-dev2.yaml