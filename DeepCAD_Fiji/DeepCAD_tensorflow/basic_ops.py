import tensorflow as tf
from keras.layers import UpSampling3D
import numpy as np
"""This script defines basic operations.
"""
def Pool3d(inputs, name):
	layer = tf.layers.max_pooling3d(inputs=inputs,
                                    pool_size=2,
			                        strides=2,
									name=name,
			                        padding='same')
	print(name,'.get_shape -----> ',str(layer.get_shape()))
	return layer


def Deconv3D(inputs, filters, name):
	layer = tf.layers.conv3d_transpose(inputs=inputs,
			                            filters=filters,
			                            kernel_size=2,
			                            strides=2,
			                            padding='same',
			                            use_bias=True,
										name=name,
			                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.05, seed=1))
	print(name,'.get_shape -----> ',str(layer.get_shape()))
	return layer


def Deconv3D_upsample(inputs, filters, name):
	print(name,'.inputs -----> ',str(inputs.get_shape()))
	kernel = tf.constant(1.0, shape=[2,2,2,filters,filters])
	inputs_shape = inputs.get_shape().as_list()
	layer = tf.nn.conv3d_transpose(value=inputs, filter=kernel,
								output_shape=[inputs_shape[0], inputs_shape[1]*2, inputs_shape[2]*2, inputs_shape[3]*2, filters],
								strides=[1, 2, 2, 2, 1],
								padding="SAME")
	print(name,'.get_shape -----> ',str(layer.get_shape()))
	return layer


def Conv3D(inputs, filters, name):
	inputs_shape = inputs.get_shape().as_list()
	fan_in =3*3*inputs_shape[-1]*filters
	std = np.sqrt(2) / np.sqrt(fan_in)
	layer = tf.layers.conv3d(inputs=inputs,
                            filters=filters,
			                kernel_size=3,
			                strides=1,
			                padding='same',
			                use_bias=True,
							name=name,
			                kernel_initializer=tf.truncated_normal_initializer(stddev=std))
	print(name,'.get_shape -----> ',str(layer.get_shape()),' std -----> ',std)
	return layer

def ReLU(inputs, name):
	layer = tf.nn.relu(inputs, name)
	print(name,'.get_shape -----> ',str(layer.get_shape()))
	return layer

def leak_ReLU(inputs, name):
	layer = tf.nn.leaky_relu(inputs, alpha=0.1, name=name) #tf.nn.relu(inputs, name)
	print(name,'.get_shape -----> ',str(layer.get_shape()))
	return layer

def Deconv3D_keras(inputs):
	layer = UpSampling3D(size=2)(inputs)
	print('.get_shape -----> ',str(layer.get_shape()))
	return layer

def up_conv3d(input, conv_filter_size, num_input_channels, num_filters, feature_map_size, feature_map_len, train=True, padding='SAME',relu=True):
    # num_input_channels 
    # num_filters 
    # feature_map_size 
    # feature_map_len 
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, conv_filter_size, num_filters, num_input_channels])
    biases = create_biases(num_filters)
    if train:
        batch_size_0 = 1 #batch_size
    else:
        batch_size_0 = 1
    layer = tf.nn.conv3d_transpose(value=input, filter=weights,
                                   output_shape=[batch_size_0, feature_map_size, feature_map_size, feature_map_len, num_filters],
                                   strides=[1, 2, 2, 2, 1],
                                   padding=padding)
    layer += biases
    if relu:
        layer = tf.nn.relu(layer)
    return layer



def BN_ReLU(inputs, training, name):
	"""Performs a batch normalization followed by a ReLU6."""
	inputs = tf.layers.batch_normalization(inputs=inputs,
                                            axis=-1,
			                                momentum=0.997,
				                            epsilon=1e-5,
				                            center=True,
				                            scale=True,
				                            training=training, 
				                            fused=True)
	return tf.nn.relu(inputs)

def GN_ReLU(inputs, name):
	"""Performs a batch normalization followed by a ReLU6."""
	inputs = group_norm(inputs, name=name)
	return tf.nn.relu(inputs)

def GN_leakReLU(inputs, name):
	"""Performs a batch normalization followed by a ReLU6."""
	# inputs = group_norm(inputs, name=name)
	inputs = tf.nn.relu(inputs) #tf.nn.relu(inputs, name)
	layer = group_norm(inputs, name=name)
	return layer

def group_norm(x, name, G=8, eps=1e-5, scope='group_norm') :
    with tf.variable_scope(scope+name, reuse=tf.AUTO_REUSE) :
        N, H, W, S, C = x.get_shape().as_list()
        G = min(G, C)
        # [N, H, W, G, C // G]
        x = tf.reshape(x, [N, H, W, S, G, C // G])
        mean, var = tf.nn.moments(x, [1, 2, 3, 5], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + eps)

        # print(' -----> GroupNorm ',x.get_shape())
        gamma = tf.get_variable('gamma_'+name, [1, 1, 1, 1, C], initializer=tf.constant_initializer(1))
        beta = tf.get_variable('beta_'+name, [1, 1, 1, 1, C], initializer=tf.constant_initializer(0))
        x = tf.reshape(x, [N, H, W, S, C]) * gamma + beta
    return x