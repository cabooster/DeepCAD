import numpy as np
import argparse
import os
import tifffile as tiff
import time
import datetime
import random
from skimage import io
import logging


def shuffle_datasets(train_raw, train_GT, name_list):
    index_list = list(range(0, len(name_list)))
    # print('index_list -----> ',index_list)
    random.shuffle(index_list)
    random_index_list = index_list
    # print('index_list -----> ',index_list)
    new_name_list = list(range(0, len(name_list)))
    train_raw = np.array(train_raw)
    # print('train_raw shape -----> ',train_raw.shape)
    train_GT = np.array(train_GT)
    # print('train_GT shape -----> ',train_GT.shape)
    new_train_raw = train_raw
    new_train_GT = train_GT
    for i in range(0,len(random_index_list)):
        # print('i -----> ',i)
        new_train_raw[i,:,:,:] = train_raw[random_index_list[i],:,:,:]
        new_train_GT[i,:,:,:] = train_GT[random_index_list[i],:,:,:]
        new_name_list[i] = name_list[random_index_list[i]]
    # new_train_raw = np.expand_dims(new_train_raw, 4)
    # new_train_GT = np.expand_dims(new_train_GT, 4)
    return new_train_raw, new_train_GT, new_name_list

def train_preprocess(args):
    img_h = args.img_h
    img_w = args.img_w
    img_s2 = args.img_s*2
    gap_h = args.gap_h
    gap_w = args.gap_w
    gap_s2 = args.gap_s*2
    im_folder = 'datasets//'+args.datasets_folder

    name_list = []
    train_raw = []
    train_GT = []
    for im_name in list(os.walk(im_folder, topdown=False))[-1][-1]:
        print('im_name -----> ',im_name)
        im_dir = im_folder+'//'+im_name
        noise_im = tiff.imread(im_dir)
        print('noise_im shape -----> ',noise_im.shape)
        # print('noise_im max -----> ',noise_im.max())
        # print('noise_im min -----> ',noise_im.min())
        noise_im = (noise_im-noise_im.min()).astype(np.float32)/args.normalize_factor

        whole_w = noise_im.shape[2]
        whole_h = noise_im.shape[1]
        whole_s = noise_im.shape[0]
        print('int((whole_h-img_h+gap_h)/gap_h) -----> ',int((whole_h-img_h+gap_h)/gap_h))
        print('int((whole_w-img_w+gap_w)/gap_w) -----> ',int((whole_w-img_w+gap_w)/gap_w))
        print('int((whole_s-img_s2+gap_s2)/gap_s2) -----> ',int((whole_s-img_s2+gap_s2)/gap_s2))
        for x in range(0,int((whole_h-img_h+gap_h)/gap_h)):
            for y in range(0,int((whole_w-img_w+gap_w)/gap_w)):
                for z in range(0,int((whole_s-img_s2+gap_s2)/gap_s2)):
                    init_h = gap_h*x
                    end_h = gap_h*x + img_h
                    init_w = gap_w*y
                    end_w = gap_w*y + img_w
                    init_s = gap_s2*z
                    end_s = gap_s2*z + img_s2
                    noise_patch1 = noise_im[init_s:end_s:2,init_h:end_h,init_w:end_w]
                    noise_patch2 = noise_im[init_s+1:end_s:2,init_h:end_h,init_w:end_w]
                    patch_name = args.datasets_folder+'_x'+str(x)+'_y'+str(y)+'_z'+str(z)
                    train_raw.append(noise_patch1.transpose(1,2,0))
                    train_GT.append(noise_patch2.transpose(1,2,0))
                    name_list.append(patch_name)
    return train_raw, train_GT, name_list, noise_im

def test_preprocess(args):
    img_h = args.img_h
    img_w = args.img_w
    img_s2 = args.img_s
    gap_h = args.gap_h
    gap_w = args.gap_w
    gap_s2 = args.gap_s
    im_folder = 'datasets//'+args.datasets_folder

    name_list = []
    train_raw = []
    for im_name in list(os.walk(im_folder, topdown=False))[-1][-1]:
        # print('im_name -----> ',im_name)
        im_dir = im_folder+'//'+im_name
        noise_im = tiff.imread(im_dir)
        # print('noise_im shape -----> ',noise_im.shape)
        # print('noise_im max -----> ',noise_im.max())
        # print('noise_im min -----> ',noise_im.min())
        noise_im = (noise_im-noise_im.min()).astype(np.float32)/args.normalize_factor

        whole_w = noise_im.shape[2]
        whole_h = noise_im.shape[1]
        whole_s = noise_im.shape[0]
        # print('int((whole_h-img_h+gap_h)/gap_h) -----> ',int((whole_h-img_h+gap_h)/gap_h))
        # print('int((whole_w-img_w+gap_w)/gap_w) -----> ',int((whole_w-img_w+gap_w)/gap_w))
        # print('int((whole_s-img_s2+gap_s2)/gap_s2) -----> ',int((whole_s-img_s2+gap_s2)/gap_s2))
        for x in range(0,int((whole_h-img_h+gap_h)/gap_h)):
            for y in range(0,int((whole_w-img_w+gap_w)/gap_w)):
                for z in range(0,int((whole_s-img_s2+gap_s2)/gap_s2)):
                    init_h = gap_h*x
                    end_h = gap_h*x + img_h
                    init_w = gap_w*y
                    end_w = gap_w*y + img_w
                    init_s = gap_s2*z
                    end_s = gap_s2*z + img_s2
                    noise_patch1 = noise_im[init_s:end_s,init_h:end_h,init_w:end_w]
                    patch_name = args.datasets_folder+'_x'+str(x)+'_y'+str(y)+'_z'+str(z)
                    train_raw.append(noise_patch1.transpose(1,2,0))
                    name_list.append(patch_name)
    return train_raw, name_list, noise_im

def test_preprocess_lessMemory (args):
    img_h = args.img_h
    img_w = args.img_w
    img_s2 = args.img_s
    gap_h = args.gap_h
    gap_w = args.gap_w
    gap_s2 = args.gap_s
    im_folder = 'datasets//'+args.datasets_folder

    name_list = []
    # train_raw = []
    coordinate_list={}
    for im_name in list(os.walk(im_folder, topdown=False))[-1][-1]:
        # print('im_name -----> ',im_name)
        im_dir = im_folder+'//'+im_name
        noise_im = tiff.imread(im_dir)
        # print('noise_im shape -----> ',noise_im.shape)
        # print('noise_im max -----> ',noise_im.max())
        # print('noise_im min -----> ',noise_im.min())
        if noise_im.shape[0]>args.test_datasize:
            noise_im = noise_im[0:args.test_datasize,:,:]
        noise_im = (noise_im-noise_im.min()).astype(np.float32)/args.normalize_factor

        whole_w = noise_im.shape[2]
        whole_h = noise_im.shape[1]
        whole_s = noise_im.shape[0]
        # print('int((whole_h-img_h+gap_h)/gap_h) -----> ',int((whole_h-img_h+gap_h)/gap_h))
        # print('int((whole_w-img_w+gap_w)/gap_w) -----> ',int((whole_w-img_w+gap_w)/gap_w))
        # print('int((whole_s-img_s2+gap_s2)/gap_s2) -----> ',int((whole_s-img_s2+gap_s2)/gap_s2))
        for x in range(0,int((whole_h-img_h+gap_h)/gap_h)):
            for y in range(0,int((whole_w-img_w+gap_w)/gap_w)):
                for z in range(0,int((whole_s-img_s2+gap_s2)/gap_s2)):
                    single_coordinate={'init_h':0, 'end_h':0, 'init_w':0, 'end_w':0, 'init_s':0, 'end_s':0}
                    init_h = gap_h*x
                    end_h = gap_h*x + img_h
                    init_w = gap_w*y
                    end_w = gap_w*y + img_w
                    init_s = gap_s2*z
                    end_s = gap_s2*z + img_s2
                    single_coordinate['init_h'] = init_h
                    single_coordinate['end_h'] = end_h
                    single_coordinate['init_w'] = init_w
                    single_coordinate['end_w'] = end_w
                    single_coordinate['init_s'] = init_s
                    single_coordinate['end_s'] = end_s
                    # noise_patch1 = noise_im[init_s:end_s,init_h:end_h,init_w:end_w]
                    patch_name = args.datasets_folder+'_x'+str(x)+'_y'+str(y)+'_z'+str(z)
                    # train_raw.append(noise_patch1.transpose(1,2,0))
                    name_list.append(patch_name)
                    # print(' single_coordinate -----> ',single_coordinate)
                    coordinate_list[patch_name] = single_coordinate
    return  name_list, noise_im, coordinate_list

def train_preprocess_lessMemory(args):
    img_h = args.img_h
    img_w = args.img_w
    img_s2 = args.img_s*2
    gap_h = args.gap_h
    gap_w = args.gap_w
    gap_s2 = args.gap_s*2
    im_folder = 'datasets//'+args.datasets_folder

    name_list = []
    # train_raw = []
    coordinate_list={}
    for im_name in list(os.walk(im_folder, topdown=False))[-1][-1]:
        # print('im_name -----> ',im_name)
        im_dir = im_folder+'//'+im_name
        noise_im = tiff.imread(im_dir)
        # print('noise_im shape -----> ',noise_im.shape)
        # print('noise_im max -----> ',noise_im.max())
        # print('noise_im min -----> ',noise_im.min())
        noise_im = (noise_im-noise_im.min()).astype(np.float32)/args.normalize_factor

        whole_w = noise_im.shape[2]
        whole_h = noise_im.shape[1]
        whole_s = noise_im.shape[0]
        # print('int((whole_h-img_h+gap_h)/gap_h) -----> ',int((whole_h-img_h+gap_h)/gap_h))
        # print('int((whole_w-img_w+gap_w)/gap_w) -----> ',int((whole_w-img_w+gap_w)/gap_w))
        # print('int((whole_s-img_s2+gap_s2)/gap_s2) -----> ',int((whole_s-img_s2+gap_s2)/gap_s2))
        for x in range(0,int((whole_h-img_h+gap_h)/gap_h)):
            for y in range(0,int((whole_w-img_w+gap_w)/gap_w)):
                for z in range(0,int((whole_s-img_s2+gap_s2)/gap_s2)):
                    single_coordinate={'init_h':0, 'end_h':0, 'init_w':0, 'end_w':0, 'init_s':0, 'end_s':0}
                    init_h = gap_h*x
                    end_h = gap_h*x + img_h
                    init_w = gap_w*y
                    end_w = gap_w*y + img_w
                    init_s = gap_s2*z
                    end_s = gap_s2*z + img_s2
                    single_coordinate['init_h'] = init_h
                    single_coordinate['end_h'] = end_h
                    single_coordinate['init_w'] = init_w
                    single_coordinate['end_w'] = end_w
                    single_coordinate['init_s'] = init_s
                    single_coordinate['end_s'] = end_s
                    # noise_patch1 = noise_im[init_s:end_s,init_h:end_h,init_w:end_w]
                    patch_name = args.datasets_folder+'_x'+str(x)+'_y'+str(y)+'_z'+str(z)
                    # train_raw.append(noise_patch1.transpose(1,2,0))
                    name_list.append(patch_name)
                    # print(' single_coordinate -----> ',single_coordinate)
                    coordinate_list[patch_name] = single_coordinate
    return  name_list, noise_im, coordinate_list


def shuffle_datasets_lessMemory(name_list):
    index_list = list(range(0, len(name_list)))
    # print('index_list -----> ',index_list)
    random.shuffle(index_list)
    random_index_list = index_list
    # print('index_list -----> ',index_list)
    new_name_list = list(range(0, len(name_list)))
    for i in range(0,len(random_index_list)):
        new_name_list[i] = name_list[random_index_list[i]]
    return new_name_list