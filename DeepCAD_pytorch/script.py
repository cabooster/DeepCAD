import os
import time

# os.system('python train.py --datasets_folder test --img_h 64 --img_w 64 --img_s 256 --gap_h 56 --gap_w 56 --gap_s 56 --n_epochs 3 --GPU 2')

os.system('python test.py --denoise_model test_20201001-0003 --test_datasize 256')