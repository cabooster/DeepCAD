import numpy as np
from keras.models import load_model
import argparse
import os
import tifffile as tiff
import time
import datetime
import random
from skimage import io
import tensorflow as tf
import logging
import time
from data_process import test_preprocess_lessMemory
from utils import read_yaml, name2index
import math


def main(args):
    # train_3d_new(args)
    test(args)

def test(args):
    model_path = args.CBSDeep_model_folder+'//'+args.denoise_model
    # print(list(os.walk(model_path, topdown=False))[-1])
    # print(list(os.walk(model_path, topdown=False))[-1][-1][0])
    # print(list(os.walk(model_path, topdown=False))[-1][-2])
    model_list = list(os.walk(model_path, topdown=False))[-1][-2]
    yaml_name = list(os.walk(model_path, topdown=False))[-1][-1][0]
    print(yaml_name)
    read_yaml(args, model_path+'//'+yaml_name)

    name_list, noise_img, coordinate_list = test_preprocess_lessMemory(args)
    num_h = (math.floor((noise_img.shape[1]-args.img_h)/args.gap_h)+1)
    num_w = (math.floor((noise_img.shape[2]-args.img_w)/args.gap_w)+1)
    num_s = (math.floor((noise_img.shape[0]-args.img_s)/args.gap_s)+1)

    TIME = args.datasets_folder+'_'+datetime.datetime.now().strftime("%Y%m%d-%H%M")
    results_path = args.results_folder+'//'+'unet3d_'+TIME+'//'
    if not os.path.exists(args.results_folder): 
        os.mkdir(args.results_folder)
    if not os.path.exists(results_path): 
        os.mkdir(results_path)
    for model_index in range(len(model_list)):
        model_name = model_list[model_index]
        print('model_name -----> ',model_name)
        output_path = results_path + '//' + model_name
        if not os.path.exists(output_path): 
            os.mkdir(output_path)

        output_graph_path = args.CBSDeep_model_folder+'//'+args.denoise_model+'//'+model_name+'//'
        start_time=time.time()
        sess = tf.Session()
        with sess.as_default():
            meta_graph_def = tf.saved_model.loader.load(sess, ['3D_N2N'], output_graph_path)
            signature = meta_graph_def.signature_def
            in_tensor_name = signature['my_signature'].inputs['input0'].name
            out_tensor_name = signature['my_signature'].outputs['output0'].name
            input = sess.graph.get_tensor_by_name(in_tensor_name)
            output = sess.graph.get_tensor_by_name(out_tensor_name)
            # sess.run(tf.global_variables_initializer())
            '''
            variable_names = [v.name for v in tf.trainable_variables()]
            values = sess.run(variable_names)
            for k,v in zip(variable_names, values):
                if len(v.shape)==5:
                    print("Variable: ", k, "Shape: ", v.shape,"value: ",v[0][0][0][0][0])
                if len(v.shape)==1:
                    print("Variable: ", k, "Shape: ", v.shape,"value: ",v[0])
            '''
            denoise_img = np.zeros(noise_img.shape)
            input_img = np.zeros(noise_img.shape)
            for index in range(len(name_list)):
                input_name = name_list[index]
                single_coordinate = coordinate_list[name_list[index]]
                init_h = single_coordinate['init_h']
                end_h = single_coordinate['end_h']
                init_w = single_coordinate['init_w']
                end_w = single_coordinate['end_w']
                init_s = single_coordinate['init_s']
                end_s = single_coordinate['end_s']
                noise_patch1 = noise_img[init_s:end_s,init_h:end_h,init_w:end_w]
                train_input = np.expand_dims(np.expand_dims(noise_patch1.transpose(1,2,0), 3),0)
                print('train_input -----> ',train_input.shape)
                data_name = name_list[index]
                train_output = sess.run(output, feed_dict={input: train_input})

                train_input = np.squeeze(train_input).transpose(2,0,1)
                train_output = np.squeeze(train_output).transpose(2,0,1)
                stack_start_w ,stack_end_w ,patch_start_w ,patch_end_w ,\
                stack_start_h ,stack_end_h ,patch_start_h ,patch_end_h ,\
                stack_start_s ,stack_end_s ,patch_start_s ,patch_end_s = name2index(args, input_name, num_h, num_w, num_s)

                denoise_img[stack_start_s:stack_end_s, stack_start_w:stack_end_w, stack_start_h:stack_end_h] \
                = train_output[patch_start_s:patch_end_s, patch_start_w:patch_end_w, patch_start_h:patch_end_h]
                input_img[stack_start_s:stack_end_s, stack_start_w:stack_end_w, stack_start_h:stack_end_h] \
                = train_input[patch_start_s:patch_end_s, patch_start_w:patch_end_w, patch_start_h:patch_end_h]
                # print('output_img shape -----> ',output_img.shape)

            output_img = denoise_img.squeeze().astype(np.float32)*args.normalize_factor
            output_img = np.clip(output_img, 0, 65535).astype('uint16')
            input_img = input_img.squeeze().astype(np.float32)*args.normalize_factor
            input_img = np.clip(input_img, 0, 65535).astype('uint16')
            result_name = output_path + '//' + 'output.tif'
            input_name = output_path + '//' + 'input.tif'
            io.imsave(result_name, output_img)
            io.imsave(input_name, input_img)







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_h', type=int, default=16, help='the height of patch stack')
    parser.add_argument('--img_w', type=int, default=16, help='the width of patch stack')
    parser.add_argument('--img_s', type=int, default=16, help='the image number of patch stack')
    parser.add_argument('--img_c', type=int, default=1, help='the channel of image')
    parser.add_argument('--gap_h', type=int, default=56, help='actions: train or predict')
    parser.add_argument('--gap_w', type=int, default=56, help='actions: train or predict')
    parser.add_argument('--gap_s', type=int, default=128, help='actions: train or predict')
    parser.add_argument('--normalize_factor', type=int, default=65535, help='actions: train or predict')
    parser.add_argument('--datasets_folder', type=str, default='test2', help='actions: train or predict')
    parser.add_argument('--model_folder', type=str, default='log', help='actions: train or predict')
    parser.add_argument('--model_epoch', type=int, default=0, help='actions: train or predict')
    parser.add_argument('--CBSDeep_model_folder', type=str, default='DeepCAD_model', help='actions: train or predict')
    parser.add_argument('--results_folder', type=str, default='results', help='actions: train or predict')
    parser.add_argument('--GPU', type=int, default=3, help='the index of GPU you will use for computation')

    parser.add_argument('--denoise_model', type=str, default='unet3d_test2_20200924-1707', help='actions: train or predict')
    parser.add_argument('--test_datasize', type=int, default=1000, help='epoch for denoising')
    args = parser.parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
    main(args)
