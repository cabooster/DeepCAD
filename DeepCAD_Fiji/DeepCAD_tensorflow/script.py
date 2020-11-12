import os
import time

# for train
os.system('python main2.py --GPU 0 --img_h 64 --img_w 64 --img_s 320 --train_epochs 30 --datasets_folder DataForPytorch --normalize_factor 1 --lr 0.00005 --train_datasets_size 10000')

# for test
os.system('python test_pb2.py --GPU 3 --denoise_model pb_unet3d_10AMP_0.3_0001_20201108-2139 \
    --datasets_folder 10AMP_0.3_0001 --model_name 25_1000')

