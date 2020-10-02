import os
import time

# os.system('python main.py --GPU 0 --img_h 64 --img_w 64 --img_s 256 --train_epochs 30 --datasets_folder test')
os.system('python test_pb.py --GPU 0 --gap_h 56 --gap_w 56 --gap_s 128 --denoise_model pb_unet3d_test_20201001-1339')

