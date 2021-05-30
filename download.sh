#!/bin/bash

wget https://www.dropbox.com/sh/nf532hddgdt68ix/AABGLUiPRyXv6UL2YAcHmAFqa?dl=1 -O data.zip
unzip data.zip -d data

gdown --id 1svEiRgQWMDk-mO8NVMhJCrwvkpQy-l4f -O distance.zip

if [ $? -gt 0 ]; then
    echo "[Error] Download of the distance matrices failed. This script may have output \"Permission denied. Maybe you need to change permission over 'Anyone with the link'?\" This is not a permission problem but a bandwith problem. In that case, please download the file from https://drive.google.com/uc?id=1svEiRgQWMDk-mO8NVMhJCrwvkpQy-l4f via a web browser and unzip it to the distance/ directory. There should be, for example, ./distance/bbcsport-emd_tr_te_split.mat.npy after this process." 1>&2
    exit 1
fi

unzip distance.zip
