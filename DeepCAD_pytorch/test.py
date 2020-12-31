import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import time
import datetime
import sys
import math
import scipy.io as scio
from network import Network_3D_Unet
from tensorboardX import SummaryWriter
import numpy as np
from utils import save_yaml, read_yaml, name2index
from data_process import shuffle_datasets, train_preprocess, test_preprocess, test_preprocess_lessMemory,test_preprocess_lessMemoryNoTail
from skimage import io
#############################################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100, help="number of training epochs")
parser.add_argument('--GPU', type=int, default=0, help="the index of GPU you will use for computation")
parser.add_argument('--cuda', action='store_true', help='use GPU computation')

parser.add_argument('--output_dir', type=str, default='./results', help="the output folder")

parser.add_argument('--batch_size', type=int, default=1, help="size of the batchs")
parser.add_argument('--img_s', type=int, default=464, help="the slices of image sequence")
parser.add_argument('--img_w', type=int, default=64, help="the width of image sequence")
parser.add_argument('--img_h', type=int, default=64, help="the height of image sequence")
parser.add_argument('--gap_h', type=int, default=48, help='the height of image gap')
parser.add_argument('--gap_w', type=int, default=48, help='the width of image gap')
parser.add_argument('--gap_s', type=int, default=400, help='the slices of image gap')

parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument('--normalize_factor', type=int, default=65535, help='normalize factor')

parser.add_argument('--datasets_path', type=str, default='datasets', help="datasets path")
parser.add_argument('--pth_path', type=str, default='pth', help="pth file path")
parser.add_argument('--datasets_folder', type=str, default='3_1.0zoom_depth70um_dendrite_0.3_low_MC', help="datasets folder")
parser.add_argument('--denoise_model', type=str, default='3_1.0zoom_depth70um_dendrite_0.3_low_MC_20201012-0703', help='denoise model')
parser.add_argument('--test_datasize', type=int, default=800, help='test dataset size')
parser.add_argument('--train_datasets_size', type=int, default=1000, help='train datasets size')
opt = parser.parse_args()
print('the parameter of your training ----->')
print(opt)
########################################################################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.GPU)
model_path = opt.pth_path+'//'+opt.denoise_model
# print(model_path)
model_list = list(os.walk(model_path, topdown=False))[-1][-1]
# print(model_list)

for i in range(len(model_list)):
    aaa = model_list[i]
    if '.yaml' in aaa:
        yaml_name = model_list[i]
print(yaml_name)
read_yaml(opt, model_path+'//'+yaml_name)
# print(opt.datasets_folder)

name_list, noise_img, coordinate_list= test_preprocess_lessMemoryNoTail(opt)
# name_list, noise_img, coordinate_list = test_preprocess_lessMemory(opt)
# trainX = np.expand_dims(np.array(train_raw),4)
num_h = (math.floor((noise_img.shape[1]-opt.img_h)/opt.gap_h)+1)
num_w = (math.floor((noise_img.shape[2]-opt.img_w)/opt.gap_w)+1)
num_s = (math.floor((noise_img.shape[0]-opt.img_s)/opt.gap_s)+1)
# print(num_h, num_w, num_s)
# print(coordinate_list)

if not os.path.exists(opt.output_dir): 
    os.mkdir(opt.output_dir)
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
output_path1 = opt.output_dir + '//' + opt.datasets_folder + '_' + current_time  + '_' + opt.denoise_model
if not os.path.exists(output_path1): 
    os.mkdir(output_path1)

yaml_name = output_path1+'//para.yaml'
save_yaml(opt, yaml_name)
denoise_generator = Network_3D_Unet(in_channels = 1,
                                    out_channels = 1,
                                    final_sigmoid = True)
if torch.cuda.is_available():
    print('Using GPU.')
for pth_index in range(len(model_list)):
    aaa = model_list[pth_index]
    if '.pth' in aaa:
        pth_name = model_list[pth_index]
        output_path = output_path1 + '//' + pth_name.replace('.pth','')
        if not os.path.exists(output_path): 
            os.mkdir(output_path)
        denoise_generator.load_state_dict(torch.load(opt.pth_path+'//'+opt.denoise_model+'//'+pth_name))

        denoise_generator.cuda()
        prev_time = time.time()
        time_start=time.time()
        denoise_img = np.zeros(noise_img.shape)
        input_img = np.zeros(noise_img.shape)
        for index in range(len(name_list)):
            single_coordinate = coordinate_list[name_list[index]]
            init_h = single_coordinate['init_h']
            end_h = single_coordinate['end_h']
            init_w = single_coordinate['init_w']
            end_w = single_coordinate['end_w']
            init_s = single_coordinate['init_s']
            end_s = single_coordinate['end_s']
            noise_patch = noise_img[init_s:end_s,init_h:end_h,init_w:end_w]
            # print(noise_patch.shape)
            real_A = torch.from_numpy(np.expand_dims(np.expand_dims(noise_patch, 3),0)).cuda()
            # print('real_A -----> ',real_A.shape)
            real_A = real_A.permute([0,4,1,2,3])
            input_name = name_list[index]
            print(' input_name -----> ',input_name)
            print(' single_coordinate -----> ',single_coordinate)
            print('real_A -----> ',real_A.shape)
            real_A = Variable(real_A)
            fake_B = denoise_generator(real_A)
            ################################################################################################################
            # Determine approximate time left
            batches_done = index
            batches_left = 1 * len(name_list) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            prev_time = time.time()
            ################################################################################################################
            if index%1 == 0:
                time_end=time.time()
                time_cost=datetime.timedelta(seconds= (time_end - time_start))
                sys.stdout.write("\r [Batch %d/%d] [Time Left: %s] [Time Cost: %s]"
                % (index,
                len(name_list),
                time_left,
                time_cost,))
            ################################################################################################################
            output_image = np.squeeze(fake_B.cpu().detach().numpy())
            raw_image = np.squeeze(real_A.cpu().detach().numpy())
            stack_start_w = int(single_coordinate['stack_start_w'])
            stack_end_w = int(single_coordinate['stack_end_w'])
            patch_start_w = int(single_coordinate['patch_start_w'])
            patch_end_w = int(single_coordinate['patch_end_w'])

            stack_start_h = int(single_coordinate['stack_start_h'])
            stack_end_h = int(single_coordinate['stack_end_h'])
            patch_start_h = int(single_coordinate['patch_start_h'])
            patch_end_h = int(single_coordinate['patch_end_h'])

            stack_start_s = int(single_coordinate['stack_start_s'])
            stack_end_s = int(single_coordinate['stack_end_s'])
            patch_start_s = int(single_coordinate['patch_start_s'])
            patch_end_s = int(single_coordinate['patch_end_s'])

            aaaa = output_image[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]
            bbbb = raw_image[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]

            denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
            = output_image[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]*(np.sum(bbbb)/np.sum(aaaa))**0.5
            input_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
            = raw_image[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]

        # del noise_img
        output_img = denoise_img.squeeze().astype(np.float32)*opt.normalize_factor
        del denoise_img
        # output_img = output_img1[0:raw_noise_img.shape[0],0:raw_noise_img.shape[1],0:raw_noise_img.shape[2]]
        output_img = output_img-output_img.min()
        output_img = output_img/output_img.max()*65535
        output_img = np.clip(output_img, 0, 65535).astype('uint16')
        output_img = output_img-output_img.min()
        # output_img = output_img.astype('uint16')
        input_img = input_img.squeeze().astype(np.float32)*opt.normalize_factor
        # input_img = input_img1[0:raw_noise_img.shape[0],0:raw_noise_img.shape[1],0:raw_noise_img.shape[2]]
        input_img = np.clip(input_img, 0, 65535).astype('uint16')
        result_name = output_path + '//' +pth_name.replace('.pth','')+ '_output.tif'
        input_name = output_path + '//' +pth_name.replace('.pth','')+ '_input.tif'
        io.imsave(result_name, output_img)
        io.imsave(input_name, input_img)





