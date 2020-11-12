import numpy as np
from keras.models import load_model
import argparse
import os
import tifffile as tiff
import time
import datetime
import random
from skimage import io
from network import autoencoder
import tensorflow as tf
import logging
import time
from utils import save_yaml
from data_process import train_preprocess_lessMemory, shuffle_datasets_lessMemory,train_preprocess_lessMemoryMulStacks


def main(args):
    train(args)

def train(args):
    TIME = args.datasets_folder+'_'+datetime.datetime.now().strftime("%Y%m%d-%H%M")
    DeepCAD_model_path = args.DeepCAD_model_folder+'//'+'pb_unet3d_'+TIME+'//'
    if not os.path.exists(args.DeepCAD_model_folder): 
        os.mkdir(args.DeepCAD_model_folder)
    if not os.path.exists(DeepCAD_model_path): 
        os.mkdir(DeepCAD_model_path)
    yaml_name = DeepCAD_model_path+'//para.yaml'
    save_yaml(args, yaml_name)
    results_path = args.results_folder+'//'+'unet3d_'+TIME+'//'
    if not os.path.exists(args.results_folder): 
        os.mkdir(args.results_folder)
    if not os.path.exists(results_path): 
        os.mkdir(results_path)

    name_list, noise_img, coordinate_list = train_preprocess_lessMemoryMulStacks(args)
    data_size = len(name_list)

    sess = tf.Session()
    input_shape = [1, args.img_h, args.img_w, args.img_s, args.img_c]
    input = tf.placeholder(tf.float32, shape=input_shape, name='input')
    # output = tf.placeholder(tf.float32, shape=input_shape, name='output')
    output_GT = tf.placeholder(tf.float32, shape=input_shape, name='output_GT')
    # net = Network(training = args.is_training)
    output = autoencoder(input, height=args.img_h, width=args.img_w, length=args.img_s)

    L2_loss = tf.reduce_mean(tf.square(output -  output_GT))
    L1_loss = tf.reduce_sum(tf.losses.absolute_difference(output, output_GT))
    loss = tf.add(L1_loss, L2_loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
    train_step = optimizer.minimize(loss)
    start_time=time.time()
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        for i in range(args.train_epochs):
            name_list = shuffle_datasets_lessMemory(name_list)
            for index in range(data_size):
                single_coordinate = coordinate_list[name_list[index]]
                init_h = single_coordinate['init_h']
                end_h = single_coordinate['end_h']
                init_w = single_coordinate['init_w']
                end_w = single_coordinate['end_w']
                init_s = single_coordinate['init_s']
                end_s = single_coordinate['end_s']
                noise_patch1 = noise_img[init_s:end_s:2,init_h:end_h,init_w:end_w]
                noise_patch2 = noise_img[init_s+1:end_s:2,init_h:end_h,init_w:end_w]
                train_input = np.expand_dims(np.expand_dims(noise_patch1.transpose(1,2,0), 3),0)
                train_GT = np.expand_dims(np.expand_dims(noise_patch2.transpose(1,2,0), 3),0)
                # print(train_input.shape)
                data_name = name_list[index]
                sess.run(train_step, feed_dict={input: train_input, output_GT: train_GT})

                if index % 100 == 0:
                    output_img, L1_loss_va, L2_loss_va = sess.run([output, L1_loss, L2_loss], feed_dict={input: train_input, output_GT: train_GT})
                    print('--- Epoch ',i,' --- Step ',index,'/',data_size,' --- L1_loss ', L1_loss_va,' --- L2_loss ', L2_loss_va,' --- Time ',(time.time()-start_time))
                    print('train_input ---> ',train_input.max(),'---> ',train_input.min())
                    print('output_img ---> ',output_img.max(),'---> ',output_img.min())
                    train_input = train_input.squeeze().astype(np.float32)*args.normalize_factor
                    train_GT = train_GT.squeeze().astype(np.float32)*args.normalize_factor
                    output_img = output_img.squeeze().astype(np.float32)*args.normalize_factor
                    train_input = np.clip(train_input, 0, 65535).astype('uint16')
                    train_GT = np.clip(train_GT, 0, 65535).astype('uint16')
                    output_img = np.clip(output_img, 0, 65535).astype('uint16')
                    result_name = results_path+str(i)+'_'+str(index)+'_'+data_name+'_output.tif'
                    noise_img1_name = results_path+str(i)+'_'+str(index)+'_'+data_name+'_noise1.tif'
                    noise_img2_name = results_path+str(i)+'_'+str(index)+'_'+data_name+'_noise2.tif'
                    io.imsave(result_name, output_img.transpose(2,0,1))
                    io.imsave(noise_img1_name, train_input.transpose(2,0,1))
                    io.imsave(noise_img2_name, train_GT.transpose(2,0,1))
                    '''
                    variable_names = [v.name for v in tf.trainable_variables()]
                    values = sess.run(variable_names)
                    for k,v in zip(variable_names, values):
                        if len(v.shape)==5:
                            print("Variable: ", k, "Shape: ", v.shape,"value: ",v[0][0][0][0][0])
                        if len(v.shape)==1:
                            print("Variable: ", k, "Shape: ", v.shape,"value: ",v[0])
                    '''
                    '''
                    aaaaa=0
                    for op in tf.get_default_graph().get_operations():
                        aaaaa=aaaaa+1
                        if aaaaa<50:
                            # print('-----> ',op.name)
                            print('-----> ',op.values())
                    '''
                    
                if index % 1000 == 0:
                    DeepCAD_model_name=DeepCAD_model_path+'//'+str(i)+'_'+str(index)+'//' 
                    builder = tf.saved_model.builder.SavedModelBuilder(DeepCAD_model_name)
                    input0 = {'input0': tf.saved_model.utils.build_tensor_info(input)}
                    output0 = {'output0': tf.saved_model.utils.build_tensor_info(output)}
                    method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                    my_signature = tf.saved_model.signature_def_utils.build_signature_def(input0, output0, method_name)
                    builder.add_meta_graph_and_variables(sess, ["3D_N2N"], signature_def_map={'my_signature': my_signature})
                    builder.add_meta_graph(["3D_N2N"], signature_def_map={'my_signature': my_signature})
                    builder.save()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_h', type=int, default=64, help='the height of patch stack')
    parser.add_argument('--img_w', type=int, default=64, help='the width of patch stack')
    parser.add_argument('--img_s', type=int, default=320, help='the image number of patch stack')
    parser.add_argument('--img_c', type=int, default=1, help='the channel of image')
    parser.add_argument('--gap_h', type=int, default=56, help='the height of patch gap')
    parser.add_argument('--gap_w', type=int, default=56, help='the width of patch gap')
    parser.add_argument('--gap_s', type=int, default=128, help='the image number of patch gap')
    parser.add_argument('--normalize_factor', type=int, default=1, help='Image normalization factor')
    parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--datasets_path', type=str, default='datasets', help="the name of your project")
    parser.add_argument('--datasets_folder', type=str, default='3', help='the folders for datasets')
    parser.add_argument('--DeepCAD_model_folder', type=str, default='DeepCAD_model', help='the folders for DeepCAD(pb) model')
    parser.add_argument('--results_folder', type=str, default='results', help='the folders for results')
    parser.add_argument('--GPU', type=int, default=3, help='the index of GPU you will use for computation')
    parser.add_argument('--is_training', type=bool, default=True, help='train or test')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--train_datasets_size', type=int, default=1000, help='actions: train or predict')
    parser.add_argument('--select_img_num', type=int, default=6000, help='actions: train or predict')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
    main(args)
