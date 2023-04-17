import os
import time
import sys

flag = sys.argv[1]

if flag == 'train':
    # for train
    os.system('python train.py --datasets_folder DataForPytorch --lr 0.00005 \
                               --img_h 150 --img_w 150 --img_s 150 --gap_h 60 --gap_w 60 --gap_s 60 \
                               --n_epochs 20 --GPU 0 --normalize_factor 1 \
                               --train_datasets_size 1200 --select_img_num 10000')

if flag == 'test':
    # for test
    os.system('python test.py --denoise_model ModelForPytorch \
                              --datasets_folder DataForPytorch \
                              --test_datasize 6000')

if flag == 'all':
    # train and then test
    os.system('python train.py --datasets_folder DataForPytorch --lr 0.00005 \
                               --img_h 150 --img_w 150 --img_s 150 --gap_h 60 --gap_w 60 --gap_s 60 \
                               --n_epochs 20 --GPU 0 --normalize_factor 1 \
                               --train_datasets_size 1200 --select_img_num 10000')

    os.system('python test.py --denoise_model ModelForPytorch \
                              --datasets_folder DataForPytorch \
                              --test_datasize 6000')
