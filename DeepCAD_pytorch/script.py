import os
import time

# for train
os.system('python train.py --datasets_folder DataForPytorch --lr 0.00005 \
    --img_h 64 --img_w 64 --img_s 464 --gap_h 64 --gap_w 64 --gap_s 150 --n_epochs 20 --GPU 0 --normalize_factor 1 --train_datasets_size 1200 --select_img_num 10000')

# for test
os.system('python test.py --denoise_model ModelForPytorch \
    --datasets_folder DataForPytorch --datasets_path datasets --pth_path pth --output_dir results --test_datasize 2000')